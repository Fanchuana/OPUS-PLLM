import gc
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         OPTConfig, OPTModel, OPTForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..opus_arch import OpusMetaModel, OpusMetaModelForCauselLM


class OpusOPTConfig(OPTConfig):
    model_type = "opus_opt"


class OpusOPTModel(OpusMetaModel, OPTModel):
    config_class = OpusOPTConfig

    def __init__(self, config: OPTConfig):
        super(OpusOPTModel, self).__init__(config)
        self.embed_tokens = self.decoder.embed_tokens



class OpusOPTForCausalLM(OPTForCausalLM, OpusMetaModelForCauselLM):
    config_class = OpusOPTConfig

    def __init__(self, config):
        super(OPTForCausalLM, self).__init__(config)
        self.model = OpusOPTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            seq: Optional[str] = None,
            input_embed: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        seq_embedding = input_embed
        #print(f'kwargs:{kwargs}')
        if seq is not None: ##training
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                seq,
                seq_embedding
            )
            del input_embed
            del seq_embedding
            del kwargs
            gc.collect()
            torch.cuda.empty_cache()
        elif seq_embedding is None and seq is None: ##inherence
            inputs_embeds = kwargs.pop('inputs_embeds', None)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            seq: Optional[str] = None,
            seq_embedding: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if seq is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                seq,
                inference_mode=True
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        #print(inputs_embeds.shape)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        #print(f'prepare_inputs_for_generation...')
        seq = kwargs.pop("seq", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if seq is not None:
            inputs['seq'] = seq
        return inputs


AutoConfig.register("opus_opt", OpusOPTConfig)
AutoModelForCausalLM.register(OpusOPTConfig, OpusOPTForCausalLM)
