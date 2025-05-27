import argparse
import torch

from multi_modality_model.multi_modality_v1.constants import IGNORE_INDEX, DEFAULT_SEQ_TOKEN_INDEX, DEFAULT_SEQ_TOKEN, SEQ_PLACEHOLDER
from multi_modality_model.multi_modality_v1.conversation import conv_vicuna_v0, conv_vicuna_v1
from multi_modality_model.multi_modality_v1.utils import disable_torch_init
from multi_modality_model.multi_modality_v1.model.builder import load_pretrained_model,return_cstp_path
from multi_modality_model.multi_modality_v1.mm_utils import (
    tokenizer_seq_token,
    get_model_name_from_path,
)
def is_protein_sequence(seq):
    standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa in standard_amino_acids for aa in seq.upper())

def eval_model(args):
    # Model
    disable_torch_init()
    if args.system_version == 'v0':
        conv = conv_vicuna_v0.copy()
    else:
        conv = conv_vicuna_v1.copy()
    model_name = get_model_name_from_path(args.model_base_path)
    print(f'args:{vars(args)}')
    cstp_path = return_cstp_path(args.opus_pllm_weights_path,'modality_encoder/modality_encoding_adapter.ckpt')
    tokenizer, model, context_len = load_pretrained_model(
        args.model_base_path, args.opus_pllm_weights_path, model_name, args.load_8bit, args.load_4bit, accelerator = None, switch_projector_type=args.switch_projector_type, cstp_path=cstp_path
    )

    # Begin interactive loop
    while True:
        instruction = input("Enter your instruction: ")
        while True:
            seq = input("Enter the protein sequence (or leave empty to skip): ").strip()
            if not seq or is_protein_sequence(seq):
                print("Valid protein sequence:", seq)
                break
            else:
                print(
                    "Invalid sequence!")

        header = f"{conv.system}\n\n"
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        human_name = conv.roles[0]
        gpt_name = conv.roles[1]

        if len(seq) == 0:
            seq = None
            prompt = (header + BEGIN_SIGNAL + human_name + ": " + instruction + END_SIGNAL + BEGIN_SIGNAL + 'Professor:')
            inputs = tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
        else:
            if DEFAULT_SEQ_TOKEN not in instruction:
                instruction = DEFAULT_SEQ_TOKEN + '\n' + instruction
            prompt = (header + BEGIN_SIGNAL + human_name + ": " + instruction + END_SIGNAL + BEGIN_SIGNAL + 'Professor:')
            input_ids = (
                tokenizer_seq_token(prompt, tokenizer, DEFAULT_SEQ_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

        # Evaluate the model
        model.eval()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                seq,
                attention_mask=None,
                pad_token_id=tokenizer.eos_token_id,
                seq_embedding=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        # Decode the output and display the result
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        try:
            index = outputs.index(conv.sep, 2)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, 2)

        outputs = outputs[:index].strip()

        print('----------------------------')
        print(f'Instruction: {instruction}')
        print(f'Sequence: {seq}')
        print(f'Output: {outputs}')
        print('----------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--opus-pllm-weights-path", type=str, default=None, required=True)
    parser.add_argument("--is_json", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--switch_projector_type", type=str, default='mlp2x_gelu')
    parser.add_argument("--load-4bit", type=bool, default=True)
    parser.add_argument("--load-8bit", type=bool, default=False)
    parser.add_argument("--system_version", type=str, default='v0')
    args = parser.parse_args()

    eval_model(args)
