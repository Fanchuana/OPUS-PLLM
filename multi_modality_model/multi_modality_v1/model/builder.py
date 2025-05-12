#    Copyright 2024 Xu Yi fan, Ying Lv
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from multi_modality_model.multi_modality_v1.model import *
from peft import PeftModel, PeftConfig
import transformers

class model_args:
    device = 'cuda'
    has_protein_encoder = True
    has_switch_projector = True
    esm_ckpt = None
def load_pretrained_model(
    model_base_path,
    adapter_path,
    model_name,
    load_8bit=False,
    load_4bit=False,
    accelerator=None,
    switch_projector_type='mlp2x_gelu',  # 改为仅关键字参数
    cstp_path = True,
    **kwargs
):
    model_args.pretrain_protein_projector_ckpt = cstp_path
    kwargs['device_map'] = "cuda:0" if accelerator is None else {"": accelerator.process_index}
    model_args.switch_projector_type = switch_projector_type
    # Set up quantization configuration
    quantization_config = None
    if load_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    #print(f'quantization_config:{quantization_config}')
    if model_name is not None and model_base_path:
        if 'llama' in model_base_path.lower():
            model = OpusLlamaForCausalLM.from_pretrained(
                model_base_path,
                use_safetensors=True, quantization_config=quantization_config, low_cpu_mem_usage=True,
                **kwargs
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_base_path, use_fast=False,
            )
            tokenizer.pad_token = tokenizer.unk_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.unk_token_id = tokenizer.eos_token_id
        elif 'opt' in model_base_path.lower() or 'galactica' in model_base_path.lower():
            model = OpusOPTForCausalLM.from_pretrained(
                model_base_path,
                use_safetensors=False, quantization_config=quantization_config,low_cpu_mem_usage=True,
                **kwargs
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_base_path, use_fast=False,
            )
            tokenizer.pad_token = "<pad>"
            tokenizer.unk_token = "<unk>"
            tokenizer.eos_token = "</s>"
        elif 'qwen' in model_base_path.lower():
            model = OpusQwenForCausalLM.from_pretrained(
                model_base_path,
                use_safetensors=True, quantization_config=quantization_config, low_cpu_mem_usage=True,
                **kwargs
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_base_path, use_fast=False,
            )
            if hasattr(model.config, 'use_dynamic_ntk'):
                model.config.use_dynamic_ntk = True
        else:
            raise NotImplementedError
        '''
        print(f'pad_token:{tokenizer.pad_token}, pad_token_id:{tokenizer.pad_token_id}')
        print(f'unk_token:{tokenizer.unk_token}, unk_token_id:{tokenizer.unk_token_id}')
        print(f'eos_token:{tokenizer.eos_token}, eos_token_id:{tokenizer.eos_token_id}')
        '''
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if adapter_path is not None:
            # Initialize the PEFT model with the base model
            peft_model = PeftModel.from_pretrained(model, adapter_path)
            # Merge and unload the LoRA weights into the base model
            model = peft_model.merge_and_unload()
            model_args.hidden_size = model.config.hidden_size
            model_args.pretrain_switch_projector_ckpt = f'{adapter_path}non_lora_trainables.bin'
            print(f'peft model loaded!')
            print(f'Switch Projector path:{model_args.pretrain_switch_projector_ckpt}')
            print(f'Lora Path:{adapter_path}')
        else:
            print(f'No adapter path!')
        # Move the model to the desired device
        if model_args.has_protein_encoder is not None:
            model.get_model().initialize_protein_modules(model_args=model_args, fsdp=None)
    else:
        raise NotImplementedError


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 512

    return tokenizer, model, context_len
