from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)
from .opus_trainer import PrintLossAndOutputCallback
import numpy as np

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import transformers
from multi_modality_model.multi_modality_v1.mm_utils import *
from multi_modality_model.multi_modality_v1.model import *
import multi_modality_model.multi_modality_v1.conversation as conversation_lib
from .opus_trainer import OPUSTrainer
from multi_modality_model.multi_modality_v1.constants import IGNORE_INDEX, DEFAULT_SEQ_TOKEN_INDEX, DEFAULT_SEQ_TOKEN
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from .opus_trainer import get_finetune_state_maybe_zero_3, get_mm_adapter_state_maybe_zero_3, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, find_all_linear_names


default_conversation_annotation, default_conversation_chat = conversation_lib.conv_vicuna_v1, conversation_lib.conv_vicuna_v3
@dataclass
class DataArguments:
    data_path: str = field(default='/path/to/data',
                           metadata={"help": "Path to the training data.Jsonl format"})
    lazy_preprocess: bool = True

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/to/model")
    has_protein_encoder: bool = field(default=True)
    has_switch_projector: bool = field(default= True)
    version: Optional[str] = field(default="llama3")
    freeze_backbone: bool = field(default=False)
    tune_switch_projector: bool = field(default=True)
    protein_encoder: Optional[str] = field(default=True)
    pretrain_protein_projector_ckpt: Optional[str] = field(default=None)
    pretrain_switch_projector_ckpt: Optional[str] = field(default=None)
    device: str = field(default='cuda')
    switch_projector_type: str = field(default='mlp2x_gelu')
    esm_ckpt: str = field(default=None)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    deepspeed: Optional[str] = field(default='/path/to/zero3.json')
    output_dir: Optional[str] = field(default='/path/to/output_dir')
    cache_dir: Optional[str] = field(default='/path/to/cache_dir')
    optim: str = field(default="adamw_torch")
    freeze_switch_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="fp4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    training_module: str = field(default='cuda')
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    switch_projector_lr: Optional[float] = field(default=None)
    fp16: bool = True
    bf16: bool = False
    tf32: bool = False
    fp32: bool = False
    logging_dir: str = './logs'
    logging_steps: int = 500
    report_to: str = "none"



def rank0_print(*args):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args)

def get_rank():
    return int(os.getenv('LOCAL_RANK', '0'))

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_switch_projector", False):
        # Only save Adapter
        keys_to_match = ['switch_projector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                switch_projector_folder = os.path.join(parent_folder, "switch_projector")
                os.makedirs(switch_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(switch_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'switch_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0] #header位置
    tokenized_lens = tokenized_lens[1:] #开始正式统计分词长度
    target[:cur_idx+1] = IGNORE_INDEX #header都要忽略
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == speakers[0]:
            target[cur_idx+3:cur_idx+tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _add_speaker_and_signal(header, source, human_name, gpt_name):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    instruction = (BEGIN_SIGNAL + human_name + ': ' +DEFAULT_SEQ_TOKEN+'\n' + source['instruction']
                    + END_SIGNAL)
    output = (BEGIN_SIGNAL +gpt_name +': ' + source['output'] + END_SIGNAL)
    conversation = header + instruction + output + BEGIN_SIGNAL
    return conversation


def return_system(source):
    if 'task_type' not in source.keys() or source['task_type'] == 'chat':
        default_conversation = default_conversation_chat
    elif source['task_type'] == 'annotation' or source['task_type'] == 'binary':
        default_conversation = default_conversation_annotation
    else:
        default_conversation = default_conversation_annotation
    return f"{default_conversation.system}\n\n", default_conversation.roles[0], default_conversation.roles[1], default_conversation



def preprocess_mix(
        sources: Sequence[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        has_seq: bool = False
):
    _, _, _, default_conv = return_system(sources[0])
    if default_conv == default_conversation_annotation:
        return preprocess(sources, tokenizer, has_seq)
    conversations = []
    sep_token = tokenizer.eos_token
    for source in sources:
        conv = default_conv.copy()
        roles = {'input':conv.roles[0],'output':conv.roles[1]}
        conv.messages = []
        conv.append_message('system',conv.system)
        conv.append_message(roles['input'],DEFAULT_SEQ_TOKEN+'\n' + source['instruction'])
        conv.append_message(roles['output'], source['output'])
        conversations.append(conv.get_prompt().strip('\n'))
    if has_seq:
        input_ids = [tokenizer_seq_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]
    def get_tokenize_len(prompts):
        return [len(tokenizer_seq_token(prompt, tokenizer)) for prompt in prompts]

    targets = copy.deepcopy(input_ids)
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(sep_token)[:-1]  # Last split is empty
        turns = [turns[0] + sep_token + turns[1]] + turns[2:] if len(turns) >= 2 else turns.copy()
        turns_length = get_tokenize_len(turns)
        target[:turns_length[0]-1] = IGNORE_INDEX
        target[turns_length[0]+1:turns_length[0] + 4] = IGNORE_INDEX
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[dict],
    tokenizer: transformers.PreTrainedTokenizer,
    has_seq: bool = False
) -> Dict:
    conversations = []
    for source in sources:
        header, human_name, gpt_name, _ = return_system(source)
        conversation = _add_speaker_and_signal(header, source, human_name, gpt_name)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_seq_token(prompt, tokenizer)) for prompt in prompts]

    if has_seq:
        input_ids = [tokenizer_seq_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    targets = copy.deepcopy(input_ids)
    for i in range(len(targets)):
        target, source, conversation = targets[i], sources[i], conversations[i]
        header, human_name, gpt_name, _ = return_system(source)
        if has_seq:
            tokenized_lens = get_tokenize_len([header] + [(BEGIN_SIGNAL + human_name +": " +DEFAULT_SEQ_TOKEN +'\n'+ source['instruction']
                             + END_SIGNAL)] + [(BEGIN_SIGNAL +gpt_name+": " + source['output'] + END_SIGNAL)])
        else:
            tokenized_lens = _tokenize_fn([header] + [(BEGIN_SIGNAL +  human_name +": " + source['instruction']
                            + END_SIGNAL)] + [(BEGIN_SIGNAL +gpt_name+": " + source['output'] + END_SIGNAL)], tokenizer)["input_ids_lens"]
        speakers = [human_name, gpt_name]
        _mask_targets(target, tokenized_lens, speakers)
    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(data_path, 'r') as f:
            self.total_samples = sum(1 for _ in f)
        print(f'total_sample:{self.total_samples}')
        self._build_index()  # 初始化时构建行索引
        self.file_handle = open(data_path, 'r')  # 保持单一文件句柄

    def __len__(self):
        return self.total_samples

    def _build_index(self):
        self.offsets = [0]
        with open(self.data_path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(f.tell())

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        self.file_handle.seek(self.offsets[i])
        line = self.file_handle.readline()
        source = json.loads(line)
        sources = [source] if isinstance(i, int) else source
        sequence = source.get('input', None)
        input_embed = np.asarray(source['input_embed']) if 'input_embed' in source else None
        data_dict =  preprocess_mix(sources, self.tokenizer, has_seq=(sequence is not None))
        if isinstance(i, int):
            data_dict = {
                'input_ids': data_dict["input_ids"][0],
                'labels': data_dict["labels"][0]
            }

        # 添加额外字段
        if sequence is not None:
            data_dict['seq'] = sequence
        else:
            raise AssertionError('No sequence input')

        data_dict['input_embed'] = input_embed
        return data_dict

    def __len__(self):
        return len(self.offsets) - 1

    def __del__(self):
        # 确保文件句柄被关闭
        if hasattr(self, 'file_handle'):
            self.file_handle.close()


class NoLazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(NoLazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # 这块的数据结构为['image':"xx/xx.jpg", 'id':"xxxx", 'conversation': [ { "from": "human", "value": "Render <image>" }], ..., ]
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        pass

    @property
    def modality_lengths(self):
        pass

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        has_seq = False
        if isinstance(i, int):  # 主要是为了保持形式上的一致
            sources = [sources]
        if 'input' in sources[0]:
            sequence = self.list_data_dict[i]['input']
            has_seq = True
        if 'input_embed' in sources[0]:
            input_embed = self.list_data_dict[i]['input_embed']
        # if 'instruction' in sources[0]:
        #    instruction = self.list_data_dict[i]['instruction']
        # if 'output' in sources[0]:
        #    output = self.list_data_dict[i]['output']
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_seq=has_seq
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        if 'input' in self.list_data_dict[i]:
            data_dict['seq'] = sequence
        else:
            assert 'No sequence input'
        if 'input_embed' in self.list_data_dict[i]:
            data_dict['input_embed'] = np.asarray(input_embed)
        else:
            data_dict['input_embed'] = None
        return data_dict  #{'seq':xxx, 'instruction':xxx, 'input_embed':xxx}

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, input_embed = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "input_embed"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_embed = torch.concat([torch.Tensor(np.asarray(embed)).reshape(1,-1) for embed in input_embed])
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            input_embed = input_embed
        )
        if 'seq' in instances[0]:
            seqs = [instance['seq'] for instance in instances]
            batch['seq'] = seqs
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, lazy=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if lazy:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    else:
        train_dataset = NoLazySupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def create_conversation(tokenizer, model_sep_style):
    if not hasattr(tokenizer, "apply_chat_template"):
        tokenizer.chat_template = conversation_lib.default_chat_template
    default_conversation_annotation = conversation_lib.conv_vicuna_v1
    default_conversation_chat.tokenizer = tokenizer #聊天
    default_conversation_chat.sep_style = model_sep_style
    return default_conversation_annotation, default_conversation_chat

def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    bnb_model_from_pretrained_args = {}

    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["switch_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    if model_args.protein_encoder is not None:
        if 'llama' in model_args.model_name_or_path.lower():
            model = OpusLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
                use_safetensors=True,
                **bnb_model_from_pretrained_args
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.pad_token = tokenizer.unk_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.unk_token_id = tokenizer.eos_token_id
            model_sep_style = conversation_lib.SeparatorStyle.LLAMA_3
        elif 'opt' in model_args.model_name_or_path.lower() or 'galactica' in model_args.model_name_or_path.lower():
            model = OpusOPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
                use_safetensors=False,
                **bnb_model_from_pretrained_args
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
            tokenizer.pad_token = "<pad>"
            tokenizer.unk_token = "<unk>"
            tokenizer.eos_token = "</s>"
            model_sep_style = conversation_lib.SeparatorStyle.SINGLE
        if 'qwen' in model_args.model_name_or_path.lower():
            model = OpusQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                trust_remote_code=True,
                use_safetensors=True,
               **bnb_model_from_pretrained_args
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True  # 必须启用
            )
            if hasattr(model.config, 'use_dynamic_ntk'):
                model.config.use_dynamic_ntk = True
            model_sep_style = conversation_lib.SeparatorStyle.Qwen_2
    else:
        raise NotImplementedError
    create_conversation(tokenizer, model_sep_style)
    model.param_ft_llm_name = set(name for name, _ in model.named_parameters())
    model_args.hidden_size = model.config.hidden_size
    print(f'pad_token:{tokenizer.pad_token}, pad_token_id:{tokenizer.pad_token_id}')
    print(f'unk_token:{tokenizer.unk_token}, unk_token_id:{tokenizer.unk_token_id}')
    print(f'eos_token:{tokenizer.eos_token}, eos_token_id:{tokenizer.eos_token_id}')


    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.require_grads()

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.training_module=='lora':
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    #print(f'model_max_length:{tokenizer.model_max_length}')
    if model_args.protein_encoder is not None:
        model.get_model().initialize_protein_modules(model_args=model_args, fsdp=training_args.fsdp)

        protein_encoder = model.get_protein_encoder()
        protein_encoder.model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_switch_projector = training_args.tune_switch_projector = model_args.tune_switch_projector

        if model_args.tune_switch_projector:
            model.requires_grad_(False)
            for p in model.get_model().switch_projector.parameters():
                p.requires_grad = True

        model.config.freeze_switch_adapter = training_args.freeze_switch_adapter
        if training_args.freeze_switch_adapter:
            for p in model.get_model().switch_projector.parameters():
                p.requires_grad = False

        model.config.switch_projector_lr = training_args.switch_projector_lr
        if training_args.bits in [4, 8]:
            model.get_model().protein_projector.to(dtype=compute_dtype, device=training_args.device)
            model.get_model().switch_projector.to(dtype=compute_dtype, device=training_args.device)
    if training_args.bits == 16:
        if training_args.bf16:
            model.get_protein_encoder().model.to(torch.bfloat16)
            model.get_model().protein_projector.to(torch.bfloat16)
            model.get_model().switch_projector.to(torch.bfloat16)
        if training_args.fp16:
            model.get_protein_encoder().model.to(torch.float16)
            model.get_model().protein_projector.to(torch.float16)
            model.get_model().switch_projector.to(torch.float16)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    rank0_print('Trying to load total data!')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path = data_args.data_path)
    rank0_print('Data Loading Over!')
    try:
        trainer = OPUSTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=[PrintLossAndOutputCallback(tokenizer=tokenizer)], **data_module)
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        model.config.use_cache = True

        if training_args.training_module=='lora':
            state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(training_args.output_dir)
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        elif training_args.training_module=='ft':
            state_dict = get_finetune_state_maybe_zero_3(model.named_parameters(), model.param_ft_llm_name)
            keys_to_match = ['switch_projector']
            pretrain_projector_state = get_mm_adapter_state_maybe_zero_3(model.named_parameters(), keys_to_match)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(pretrain_projector_state, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    except torch.cuda.OutOfMemoryError as e:
        print(f"OutOfMemoryError: {e}")
        print(f"Current batch data: {data_module['train_dataset'][trainer.state.global_step]}"
              f" last batch data: {data_module['train_dataset'][trainer.state.global_step-1]}")
        torch.cuda.empty_cache()
        raise e

if __name__ == "__main__":
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train()