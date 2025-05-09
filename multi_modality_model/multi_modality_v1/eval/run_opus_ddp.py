import argparse
import torch

from multi_modality_model.multi_modality_v1.constants import IGNORE_INDEX, DEFAULT_SEQ_TOKEN_INDEX, DEFAULT_SEQ_TOKEN, SEQ_PLACEHOLDER
from multi_modality_model.multi_modality_v1.conversation import conv_vicuna_v0, conv_vicuna_v1
from multi_modality_model.multi_modality_v1.utils import disable_torch_init
from multi_modality_model.multi_modality_v1.model.builder import load_pretrained_model
from multi_modality_model.multi_modality_v1.mm_utils import (
    tokenizer_seq_token,
    get_model_name_from_path,
)
import json
from tqdm import tqdm
import sys
from accelerate import Accelerator
from accelerate.utils import gather_object
import time
from metrics_computing_opi import return_opi_metrics
from metrics_computing_ours import return_our_metrics
def after_process_output(outputs, conv):
    outputs = outputs.strip()
    try:
        index = outputs.index(conv.sep, 0)
    except ValueError:
        outputs += conv.sep
        index = outputs.index(conv.sep, 0)
    # print(f'index:{index}')
    outputs = outputs[:index].strip()
    return outputs


def left_pad_sequence(sequences, padding_value, batch_first=False):
    # 计算所有序列中的最大长度
    max_len = max([seq.size(0) for seq in sequences])

    # 创建填充后的张量
    padded_sequences = []
    for seq in sequences:
        # 先在序列前添加 bos_token
        padding_length = max_len - seq.size(0)
        # 创建填充部分
        padding = torch.full((padding_length,), padding_value, dtype=seq.dtype, device=seq.device)
        # 拼接填充部分和原始序列
        padded_seq = torch.cat([padding, seq], dim=0)
        padded_sequences.append(padded_seq)


    # 将列表转换为张量
    if batch_first:
        return torch.stack(padded_sequences)
    else:
        return torch.stack(padded_sequences).transpose(0, 1)


def eval_model(args):
    # Model
    if args.system_version == 'v0':
        conv = conv_vicuna_v0.copy()
    else:
        conv = conv_vicuna_v1.copy()
    accelerator = Accelerator()
    model_name = get_model_name_from_path(args.model_base_path)
    #print(f'args:{vars(args)}')
    print(args)
    tokenizer, model, context_len = load_pretrained_model(
        args.model_base_path, args.adapter_path, model_name, args.load_8bit, args.load_4bit, accelerator = accelerator, switch_projector_type=args.switch_projector_type, cstp_path=args.cstp_path
    )
    start = 0
    if args.is_json is not None:
        if args.json_path is not None:
            qs = json.load(open(args.json_path, "r"))
            if type(qs) is list:
                length = len(qs)
     #           if len(qs) == 1:
                qs = [item for item in qs if item['input']!=None]
                seqs = [item['input'] for item in qs[:length]]
                instructions = [item['instruction'] for item in qs[start:start+length]]
                ground_truthes = [item['output'] for item in qs[start:start+length]]
              #  else:
              #      raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


    #print(f'prompt:{instruction}, seq:{seq}, model_name:{model_name}')
    batch_size = 8
    total_ground_truthes = ground_truthes
    with accelerator.split_between_processes(instructions) as instructions, \
         accelerator.split_between_processes(ground_truthes) as ground_truthes, \
         accelerator.split_between_processes(seqs) as seqs:
        results = dict(outputs=[], questions=[])
        total_length = len(instructions)
        batch_instructions = [instructions[i: min(i+batch_size, total_length)] for i in range(0, total_length, batch_size)]
        batch_ground_truthes = [ground_truthes[i: min(i+batch_size, total_length)] for i in range(0, total_length, batch_size)]
        batch_seqs = [seqs[i: min(i + batch_size, total_length)] for i in
                                range(0, total_length, batch_size)]
        #print(batch_instructions)
        start_time = time.time()
        for i in tqdm(range(len(batch_instructions)),position=0):
            instruction, seq, ground_truth = batch_instructions[i], batch_seqs[i], batch_ground_truthes[i]
            new_instruction = []
            for instruct in instruction:
                if DEFAULT_SEQ_TOKEN not in instruct:
                    if 'localization' in args.json_path:
                        instruct = DEFAULT_SEQ_TOKEN + '\n' + instruct + 'Kindly reply with only one word.'
                        args.max_new_tokens = 32
                    elif 'keywords' in args.json_path:
                        instruct = DEFAULT_SEQ_TOKEN + '\n' + instruct
                        args.max_new_tokens = 128
                    else:
                        instruct = DEFAULT_SEQ_TOKEN + '\n' + instruct
                        args.max_new_tokens = 256
                header = f"{conv.system}\n\n"
                BEGIN_SIGNAL = "### "
                END_SIGNAL = "\n"
                human_name = conv.roles[0]
                gpt_name = conv.roles[1]
                #print(gpt_name)
                instruct = (header + BEGIN_SIGNAL + human_name +": " +  instruct + END_SIGNAL + BEGIN_SIGNAL+'Professor:')
                new_instruction.append(instruct)
            input_ids = [
                tokenizer_seq_token(item, tokenizer, DEFAULT_SEQ_TOKEN_INDEX, return_tensors="pt")
                .cuda() for item in new_instruction
            ]
            input_ids = left_pad_sequence(
                input_ids,
                padding_value = tokenizer.pad_token_id,
                batch_first=True)
            attention_mask = (input_ids != tokenizer.pad_token_id)
            model.eval()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    seq,
                    attention_mask = attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    seq_embedding = None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [after_process_output(item,conv) for item in outputs]
            results["outputs"].extend(outputs)
            results["questions"].extend(new_instruction)
        final_results = [results['outputs']]
    results_gathered = gather_object(final_results)
    if accelerator.is_main_process:
        timediff = time.time() - start_time
        ans_list = [item for sublist in results_gathered for item in sublist]
        result_to_save = [{'ground_truth':gt,'generated':ans} for (gt, ans) in zip(total_ground_truthes,ans_list)]
        print(f"entries/sec: {length / timediff}, time elapsed: {timediff}")
        print(f'Inferece Score of Dataset: {args.json_path}')
        print(f'Saving Inference Result of model: {args.adapter_path} to {args.save_path}...')
        with open(args.save_path, 'w') as f:
           json.dump(result_to_save, f)
        return_opi_metrics(result_to_save, args.json_path)
        return_our_metrics(result_to_save, args.json_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--is_json", type=bool, default=False, required=True)
    parser.add_argument("--json_path", type=str, default=False)
    parser.add_argument("--save_path", type=str, default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--switch_projector_type", type=str, default='mlp2x_gelu')
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--load-8bit", type=bool, default=False)
    parser.add_argument("--cstp_path", type=str, default=None)
    parser.add_argument("--system_version", type=str, default='v1')
    args = parser.parse_args()

    eval_model(args)
