from torch.amp import autocast
from abc import ABC, abstractmethod
import torch
from .protein_encoder.builder import build_protein_encoder
from .protein_projector.builder import build_protein_projector
from .protein_mlp.builder import build_switch_projector
from multi_modality_model.multi_modality_v1.constants import IGNORE_INDEX, DEFAULT_SEQ_TOKEN_INDEX, DEFAULT_SEQ_TOKEN
import os

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



def rank0_print(*args):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(*args)

class OpusMetaModel:
    def __init__(self, config):
        super(OpusMetaModel, self).__init__(config)
        if hasattr(config, "has_protein_encoder") and hasattr(config, 'has_switch_projector'):
            self.protein_encoder = build_protein_encoder(config.esm_ckpt) #load ESM model, ESM model is not a Pytorch Module
            self.switch_projector = build_switch_projector(config) #load projector switch embedding to the text embedding space
            self.protein_projector = None
            if 'limitation' in getattr(config, 'Any_Limitation', ''):
                raise NotImplementedError
    def get_protein_encoder(self):
        #print(f'tring to get protein_encoder...')
        protein_encoder = getattr(self, 'protein_encoder', None)
        return protein_encoder


    def initialize_protein_modules(self, model_args, fsdp=None):
        #print(f'receive protein projector ckpt:{model_args.pretrain_protein_projector_ckpt}')
        #print(f'receive switch projector ckpt:{model_args.pretrain_switch_projector_ckpt}')
        self.config.device = model_args.device
        self.config.has_protein_encoder = model_args.has_protein_encoder
        self.config.has_switch_projector = model_args.has_switch_projector
        if self.get_protein_encoder() is None:
            protein_encoder = build_protein_encoder(model_args.esm_ckpt)
            if fsdp is not None and len(fsdp) > 0:
                self.protein_encoder = [protein_encoder]
            else:
                self.protein_encoder = protein_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:  # Fully Sharded Data Parallel
                protein_encoder = self.protein_encoder[0]
            else:
                protein_encoder = self.protein_encoder
            protein_encoder.load_model()
        #print(f'protein_encoder:{vars(protein_encoder)}')
        #print(f'self:{vars(self)}')
        if model_args.pretrain_protein_projector_ckpt is not None:  # 是否有projector
            #print(f'pretrain_protein_projector_ckpt_path:{model_args.pretrain_protein_projector_ckpt}')
            self.protein_projector = build_protein_projector(model_args.pretrain_protein_projector_ckpt)
            self.protein_projector = self.protein_projector.to(self.config.device)
        else:
            #print('Receive No protein projector, directly use esm embedding!')
            import torch.nn as nn
            class IdentityModule(nn.Module):
                def forward(self, x):
                    return x
                def protein_forward(self, x):
                    return x

            self.protein_projector = IdentityModule()
            self.protein_projector = self.protein_projector.to(self.config.device)
        if getattr(model_args, 'has_switch_projector', False):
            self.switch_projector = build_switch_projector(model_args)
            if model_args.pretrain_switch_projector_ckpt is not None:
                #print(f'pretrain_switch_projector_ckpt_path:{model_args.pretrain_switch_projector_ckpt}')
                switch_projector_weights = torch.load(model_args.pretrain_switch_projector_ckpt, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.switch_projector.load_state_dict(get_w(switch_projector_weights, 'switch_projector'))
            self.switch_projector = self.switch_projector.to(self.config.device)



class OpusMetaModelForCauselLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_protein_encoder(self):
        return self.get_model().get_protein_encoder()

    def encode_seq2embedding(
            self,
            seq
    ):
        with autocast(device_type='cuda'):
            if type(seq) is not list:
                seq = [seq]
            if type(seq[0]) is str:
                extractor_embedding = self.get_protein_encoder().get_protein_seq_embeddings(seq)
            else:
                raise NotImplementedError
            return extractor_embedding
    def encode_projector_embedding(
            self,
            extractor_embedding
    ):
        with autocast(device_type='cuda'):
            seq_embedding = self.get_model().protein_projector.protein_forward(extractor_embedding)
            return seq_embedding
    def switch_projector_embedding(
            self,
            seq_embedding
    ):
        with autocast(device_type='cuda'):
            seq_embedding = self.get_model().switch_projector(seq_embedding)
            batch_sz = seq_embedding.shape[0]
            model_hdsz = self.get_model().config.hidden_size
            seq_embedding = seq_embedding.reshape(batch_sz, -1, model_hdsz)
            return seq_embedding

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, seq, seq_embedding = None, inference_mode = False
    ):
        protein_encoder = self.get_protein_encoder()
        if seq is None or protein_encoder is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if seq_embedding is None:
            seq_embedding = self.encode_seq2embedding(seq)
            seq_embedding = self.encode_projector_embedding(seq_embedding)
            if self.config.has_switch_projector:
                seq_embedding = self.switch_projector_embedding(seq_embedding)

            if seq_embedding.ndimension() == 3:  # batch_sz, max_len, hidden_sz
                pass
            elif seq_embedding.ndimension() == 2:  # batch_sz, 1, hidden_sz
                seq_embedding = seq_embedding.unsqueeze(1)
            else:
                raise NotImplementedError
        else:
            torch.cuda.empty_cache()
            seq_embedding = self.encode_projector_embedding(seq_embedding)
            if self.config.has_switch_projector:
                seq_embedding = self.switch_projector_embedding(seq_embedding)
            if seq_embedding.ndimension() == 3:  # batch_sz, max_len, hidden_sz
                pass
            elif seq_embedding.ndimension() == 2:  # batch_sz, 1, hidden_sz
                seq_embedding = seq_embedding.unsqueeze(1)
            else:
                raise NotImplementedError
        #print(seq_embedding.shape)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [current_input_ids[current_mask] for current_input_ids, current_mask in
                     zip(input_ids, attention_mask)]
        labels = [current_labels[current_mask] for current_labels, current_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        seq_idx = 0
        '''
Here, the text and protein embeddings are concatenated and aligned. 
The process is performed in batches, with each iteration step including the following three steps:
1.Check the total number of protein sequences. If there are none, insert an empty feature and terminate directly.
2.If the sequence is empty, insert an empty feature and terminate directly.
3.If the sequence is not empty, split the text according to the positions of protein tokens, 
concatenate it, and pass it to the text encoder to obtain the embedding. Then, separate the embeddings and alternately 
insert the embeddings encoded by the protein encoder.
        '''
        for batch_idx, cur_input_ids in enumerate(input_ids):
            #rank0_print(f'cur_input_ids_shape:{cur_input_ids.shape}')
            num_protein = (cur_input_ids == DEFAULT_SEQ_TOKEN_INDEX).sum()
            #print(f'num_protein: {num_protein}')
            if num_protein == 0:
                cur_seq_embedding = seq_embedding[seq_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_seq_embedding[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                seq_idx += 1
                continue
            seq_token_indices = [-1] + torch.where(cur_input_ids == DEFAULT_SEQ_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_no_seq = []
            cur_labels = labels[batch_idx]
            cur_labels_no_seq = []
            for i in range(len(seq_token_indices) - 1):
                cur_input_ids_no_seq.append(cur_input_ids[seq_token_indices[i] + 1:seq_token_indices[i + 1]])
                cur_labels_no_seq.append(cur_labels[seq_token_indices[i] + 1:seq_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_input_ids_no_seq]
            cur_input_embeds_temp = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_seq))
            cur_input_embeds_no_seq = torch.split(cur_input_embeds_temp, split_sizes, dim=0)
            cur_input_embeds_has_seq = []
            cur_labels_has_seq = []

            for i in range(num_protein + 1):
                cur_input_embeds_has_seq.append(cur_input_embeds_no_seq[i])
                cur_labels_has_seq.append(cur_labels_no_seq[i])
                if i < num_protein:
                    cur_seq_embedding = seq_embedding[seq_idx]
                    cur_input_embeds_has_seq.append(cur_seq_embedding)
                    seq_idx += 1
                    cur_labels_has_seq.append(
                        torch.full((cur_seq_embedding.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))
            cur_input_embeds_has_seq = [x.to(self.device) for x in cur_input_embeds_has_seq]
            #rank0_print(f'cur_input_embeds_has_shape:{[embeds.shape for embeds in cur_input_embeds_has_seq]}')
            cur_input_embeds_has_seq = torch.cat(cur_input_embeds_has_seq)
            cur_labels_has_seq = torch.cat(cur_labels_has_seq)
            new_input_embeds.append(cur_input_embeds_has_seq)
            new_labels.append(cur_labels_has_seq)
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            #print(f'cur_new_embed.shape[0]:{cur_new_embed.shape[0]} max_len:{max_len}')
            if inference_mode:
                new_input_embeds_padded.append(torch.cat((
                                                          torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                                                      dtype=cur_new_embed.dtype,
                                                                      device=cur_new_embed.device),cur_new_embed
                                                          ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i,max_len - cur_len :] = cur_new_labels
                    attention_mask[i,max_len - cur_len :] = True
                    position_ids[i,max_len - cur_len :] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed,
                                                          torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                                                      dtype=cur_new_embed.dtype,
                                                                      device=cur_new_embed.device)
                                                          ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        '''
        new_input_embeds: batch_sz, max_len, hidden_sz
        new_labels: batch_sz, max_len
        position_ids: batch_sz, max_len
        attention_mask: batch_sz, max_len
        '''
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        #rank0_print(f'new_input_embeds.shape:{new_input_embeds.shape} new_input_embeds:{new_input_embeds}')
        #raise NotImplementedError
        #print(attention_mask, new_input_embeds)
        #print(f'new_labels:{new_labels.shape},new_input_embeds:{new_input_embeds.shape}')
        #raise NotImplementedError
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_seq_tokenizer(self, model_args, tokenizer):
        num_new_tokens = tokenizer.add_tokens([DEFAULT_SEQ_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].float().mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].float().mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


