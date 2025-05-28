from multi_modality_model.multi_modality_v1.model.protein_encoder.builder import build_protein_encoder
from multi_modality_model.multi_modality_v1.model.protein_projector.builder import build_protein_projector
import json
import tqdm
import torch

def generate_esm_embedding(args):
    encoder = build_protein_encoder(None)

    print('文件读取中....')
    with open(args.file_path, 'r') as f:
        data = json.load(f)
    print(len(data))
    output_data = []
    data = [{'instruction':item['instruction'],'input':item['input'], 'output':item['output']} for item in data]
    seq_embedding = json.load(open(args.dict_path, 'r')) if args.dict_path!=None else {}
    for item in tqdm.tqdm(data, position=0):
        seq = item['input']
        if len(seq)>4000:
            continue
        new_item = {
            "instruction": item['instruction'],
            "input": seq,
            "output": item['output'],
            "input_embed":seq_embedding[item['input']] if seq in seq_embedding.keys() else encoder.get_protein_seq_embeddings([seq]).cpu().numpy().tolist()[0]
        }
        output_data.append(new_item)
    print('文件写入中....')
    with open(args.save_path, 'w') as file:
        for item in output_data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='/path/to/file')
    parser.add_argument("--save_path", type=str, default='/path/to/save') #jsonl
    parser.add_argument("--dict_path", type=str, default='/path/to/dict')
    args = parser.parse_args()
    generate_esm_embedding(args)