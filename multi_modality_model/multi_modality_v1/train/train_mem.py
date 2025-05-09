from multi_modality_model.multi_modality_v1.train.train import train
from multi_modality_model.multi_modality_v1.train.llama_xformers_attn_monkey_patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

if __name__ == "__main__":
    train()
