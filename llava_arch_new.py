# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Llava model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import AutoModel, AutoModelForCausalLM
from configuration_llava import LlavaConfig
from transformers import CLIPProcessor, CLIPTextModelWithProjection


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "llava-hf/llava-1.5-7b-hf"


@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size):
        super().__init__()

        self.linear_1 = nn.Linear(input_hidden_size, output_hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(output_hidden_size, output_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_START_DOCSTRING,
)
class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


class LlavaForConditionalGeneration(nn.Module):
    def __init__(
        self, 
        clip_model_name="/root/autodl-tmp/clip-vit-large-patch14", 
        llm_model_name="/root/autodl-tmp/Meta-Llama-3.1-8B",
        multi_model_projector_path=None,
        freeze=True, 
        cache_dir="/root/autodl-tmp/huggingface",
        special_token_id=None,
        clip_pad_token_id=None,
        llm_pad_token_id=None,
    ):
        super().__init__()
        self.clip_model = AutoModel.from_pretrained(clip_model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=cache_dir, trust_remote_code=True)

        self.multi_model_projector = LlavaMultiModalProjector(self.clip_model.config.projection_dim, self.language_model.config.hidden_size)
        if multi_model_projector_path is not None:
            self.multi_model_projector.load_state_dict(torch.load(multi_model_projector_path))
        
        self.vocab_size = self.language_model.config.vocab_size

        self.special_token_id = special_token_id

        self.clip_pad_token_id = clip_pad_token_id
        self.llm_pad_token_id = llm_pad_token_id

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            for param in self.language_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        
        batch_size, seq_length = input_ids.shape

        special_token_positions = (input_ids == self.special_token_id).nonzero(as_tuple=True)[1].to(input_ids.device)
        index_matrix = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        before_mask = (index_matrix < special_token_positions.unsqueeze(1))
        after_mask = (index_matrix >= special_token_positions.unsqueeze(1))

        before_input_ids = torch.where(before_mask, input_ids, torch.full_like(input_ids, self.clip_pad_token_id))
        after_input_ids = torch.where(after_mask, input_ids, torch.full_like(input_ids, self.llm_pad_token_id))

        before_inputs_embeds = self.clip_model.text_model.transformer(before_input_ids, before_mask).last_hidden_state
        before_inputs_embeds = self.multi_model_projector(before_inputs_embeds)
        after_inputs_embeds = self.language_model.get_input_embeddings()(after_input_ids)

        inputs_embeds = before_inputs_embeds * before_mask.float().unsqueeze(-1) + after_inputs_embeds * after_mask.float().unsqueeze(-1)
        inputs_embeds = inputs_embeds.half()

        outputs = self.language_model(
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=None,
        )


    def image_text_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        max_new_tokens=None,
        **kwargs,
    ):
        with torch.no_grad():
            inputs_embeds = self.clip_model.text_model.transformer(input_ids, attention_mask).last_hidden_state
            inputs_embeds = self.multi_model_projector(inputs_embeds)

            if pixel_values is not None:
                vision_embedding = self.clip_model.vision_model(pixel_values, return_all_features=True)
                vision_embedding = self.multi_model_projector(vision_embedding)
                vision_attention_mask = torch.ones(vision_embedding.shape[:2], device=vision_embedding.device)

                inputs_embeds = torch.cat([vision_embedding, inputs_embeds], dim=1)
                attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)

            return self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Trigger the new behavior if we have more than image embeddings seq length tokens for images
        legacy_processing = (
            input_ids is not None
            and (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
        )

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if legacy_processing or cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs