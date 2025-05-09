import argparse
import torch

from multi_modality_model.multi_modality_v1.constants import IGNORE_INDEX, DEFAULT_SEQ_TOKEN_INDEX, DEFAULT_SEQ_TOKEN, \
    SEQ_PLACEHOLDER
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
import multi_modality_model.multi_modality_v1.conversation as conversation_lib
import time
import os
def rank0_print(*args):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args)

def after_process_output(outputs, sep):
    outputs = outputs.strip()
    try:
        index = outputs.index(sep, 0)
    except ValueError:
        outputs += sep
        index = outputs.index(sep, 0)
    outputs = outputs[:index].strip()
    return outputs


def left_pad_sequence(sequences, padding_value, batch_first=False):
    max_len = max([seq.size(0) for seq in sequences])
    padded_sequences = []
    for seq in sequences:
        padding_length = max_len - seq.size(0)
        padding = torch.full((padding_length,), padding_value, dtype=seq.dtype, device=seq.device)
        padded_seq = torch.cat([padding, seq], dim=0)
        padded_sequences.append(padded_seq)

    if batch_first:
        return torch.stack(padded_sequences)
    else:
        return torch.stack(padded_sequences).transpose(0, 1)


def eval_model(args):
    # Model
    accelerator = Accelerator()
    model_name = get_model_name_from_path(args.model_base_path)
    tokenizer, model, context_len = load_pretrained_model(
        args.model_base_path, args.adapter_path, model_name, args.load_8bit, args.load_4bit, accelerator=accelerator,
        switch_projector_type=args.switch_projector_type,cstp_path=args.cstp_path
    )
    chat_template = """
    {% for message in messages %}
        {% if message['role'] == 'system' %}
            <|im_start|>system\n{{ message['content'] }}<|im_end|>\n
        {% elif message['role'] == 'user' %}
            <|im_start|>user\n{{ message['content'] }}<|im_end|>\n
        {% elif message['role'] == 'assistant' %}
            <|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n
        {% endif %}
    {% endfor %}
    {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}
    """
    tokenizer.chat_template = chat_template if tokenizer.chat_template == None else tokenizer.chat_template
    start = 0

    def return_prompt(question, options):
        return f"""Question: {question}

        Options:
        {options}

        Please carefully read the question and select the single correct answer from A-D.
        You can only output one option from A), B), C), D) with format 'The correct answer is' without explanation."""

    if args.is_json is not None:
        if args.json_path is not None:
            qs = json.load(open(args.json_path, "r"))
            if type(qs) is list:
                length = len(qs)
                qs = qs[:length]

                #           if len(qs) == 1:
                def return_options(option_list):
                    return "\n".join(option_list)

                instructions = [return_prompt(item['question'], return_options(item['options'])) for item in qs]
                seqs = [item['input'] for item in qs]
                ground_truthes = [item['answer'] for item in qs]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    batch_size = 8
    total_ground_truthes = ground_truthes
    with accelerator.split_between_processes(instructions) as instructions, \
            accelerator.split_between_processes(ground_truthes) as ground_truthes, \
            accelerator.split_between_processes(seqs) as seqs:
        results = dict(outputs=[], questions=[])
        total_length = len(instructions)
        batch_instructions = [instructions[i: min(i + batch_size, total_length)] for i in
                              range(0, total_length, batch_size)]
        batch_ground_truthes = [ground_truthes[i: min(i + batch_size, total_length)] for i in
                                range(0, total_length, batch_size)]
        batch_seqs = [seqs[i: min(i + batch_size, total_length)] for i in
                      range(0, total_length, batch_size)]
        # print(batch_instructions)
        start_time = time.time()
        for i in tqdm(range(len(batch_instructions)), position=0):
            instruction, seqs, ground_truth = batch_instructions[i], batch_seqs[i], batch_ground_truthes[i]
            new_instruction = []
            sep = ''
            for instruct, seq in zip(instruction, seqs):
                conv = conversation_lib.conv_vicuna_v3.copy()
                conv.tokenizer = tokenizer
                sep = conv.sep
                conv.append_message('system', conv.system)
                if len(seq) == 0:
                    conv.append_message('user', instruct)
                    prompt = conv.get_prompt_eval()
                else:
                    conv.append_message('user', DEFAULT_SEQ_TOKEN + '\n' + instruct)
                    prompt = conv.get_prompt_eval()
                new_instruction.append(prompt)
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
            # attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device='cuda')
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    seqs,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    seq_embedding=None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [after_process_output(item, sep) for item in outputs]
            results["outputs"].extend(outputs)
            results["questions"].extend(new_instruction)
        final_results = [results['outputs']]
        final_questions = [results['questions']]

    results_gathered = gather_object(final_results)
    question_gathered = gather_object(final_questions)
    if accelerator.is_main_process:
        timediff = time.time() - start_time
        ans_list = [item for sublist in results_gathered for item in sublist]
        option_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0,'None':0}
        result_to_save = [{'ground_truth': gt, 'generated': ans} for (gt, ans) in zip(total_ground_truthes, ans_list)]

        def extract_option_letter(text):
            import re
            match = re.search(
                r"\b([A-Da-d])[\s]*[).\.）\]】]|answer\s*:\s*([A-Da-d])|答案是\s*([A-Da-d])",
                text,
                re.IGNORECASE
            )
            if not match:
                return text

            letter = next((g for g in match.groups() if g is not None), None)
            return letter.upper() if letter else None

        def compare_answers(data):
            correct_count = 0
            total = len(data)

            for item in data:
                ground_truth = item['ground_truth']
                generated = item['generated']
                generated_option = extract_option_letter(item['generated'])
                ground_truth_option = extract_option_letter(item['ground_truth'])
                if generated_option == ground_truth_option:
                    correct_count += 1
                    print(f"✅ Correct: Ground Truth = {ground_truth}, Generated = {generated}")
                else:
                    print(f"❌ Incorrect: Ground Truth = {ground_truth}, Generated = {generated}")
                if generated_option is not None and generated_option.strip('') in option_count.keys():
                    option_count[generated_option] += 1
                else:
                    option_count['None']+=1

            # 计算准确率
            print(option_count)
            accuracy = correct_count / total
            print(f"\n{correct_count}/{total}:Accuracy: {accuracy * 100:.2f}%")

        compare_answers(result_to_save)
        print(f"entries/sec: {length / timediff}, time elapsed: {timediff}")
        print(f'Inferece Score of Dataset: {args.json_path}')
        print(f'Saving Inference Result of model: {args.adapter_path} to {args.save_path}...')
        with open(args.save_path, 'w') as f:
            json.dump(result_to_save, f)
        '''
        return_opi_metrics(result_to_save, args.json_path)
        return_our_metrics(result_to_save, args.json_path)
        '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--is_json", type=bool, default=True, required=False)
    parser.add_argument("--json_path", type=str, default=False)
    parser.add_argument("--save_path", type=str, default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--switch_projector_type", type=str, default='mlp2x_gelu')
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--load-8bit", type=bool, default=False)
    parser.add_argument("--cstp_path", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)