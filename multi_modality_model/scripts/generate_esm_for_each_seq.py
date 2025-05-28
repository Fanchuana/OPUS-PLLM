from multi_modality_model.multi_modality_v1.model.protein_encoder.builder import build_protein_encoder
from multi_modality_model.multi_modality_v1.model.protein_projector.builder import build_protein_projector
import json
from tqdm import tqdm
import torch

def generate_esm_embedding(args):
    encoder = build_protein_encoder(None)
    with open(args,'r') as f:
        sequence_set = json.load(f)
    print('读取dict字典')
    seq_embedding = json.load(open('seq2embed.json', 'r'))
    sequence_set = list(set([item['input'] for item in sequence_set]))
    seq_2_embed_dict = {}
    for seq in tqdm(sequence_set, position=0):
        if len(seq)<4000:
            seq_2_embed_dict[seq] = seq_embedding[seq] if seq in seq_embedding.keys() else encoder.get_protein_seq_embeddings([seq]).cpu().numpy().tolist()[0]
            torch.cuda.empty_cache()
    json.dump(seq_2_embed_dict, open("/path/to/seq2embed.json",'w'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='/path/to/file')
    parser.add_argument("--save_path", type=str, default='/path/to/save')
    parser.add_argument("--dict_path", type=str, default='/path/to/model')
    args = parser.parse_args()
    generate_esm_embedding(args)