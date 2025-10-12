# We apply token_pruning to clip. The code of this file is based on the source code of clip.
from typing import Tuple, Optional, Union
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPEncoder


class ReduceEncoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        index_matrix: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        reduce_ratio: float = 0.0,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=True,
        )
        hidden_states = residual + hidden_states
        if reduce_ratio==0.0:

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
        else:
            attention_scores=torch.mean(attn_weights, dim=1)[:, 0, :][:, 1:]#bsz,tgt_len-1
            bs=attention_scores.shape[0]
            s=attention_scores.shape[1]
            remove_count = int(s * reduce_ratio)

            _, lowest_attention_indices = torch.topk(attention_scores, remove_count, largest=False, sorted=False, dim=1)
            new_tokens_list = []
            new_indices_list = []

            for i in range(bs):
                mask = torch.ones(s, dtype=torch.bool)

                mask[lowest_attention_indices[i]] = False


                mask=torch.cat((torch.ones(1, dtype=torch.bool),mask),dim=0)

                new_tokens = hidden_states[i][mask]
                new_indices = index_matrix[i][mask]

                new_tokens_list.append(new_tokens)
                new_indices_list.append(new_indices)

            hidden_states = torch.stack(new_tokens_list)
            hidden_states = self.layer_norm2(hidden_states)
            index_matrix = torch.stack(new_indices_list)

            residual = hidden_states


        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        outputs += (index_matrix,)
        return outputs

class ReduceEncoder(nn.Module):
    def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        assert ReduceEncoder.init_index_matrix!=None
        index_matrix = ReduceEncoder.init_index_matrix
        selected_hidden_states_list = []
        for i in range(hidden_states.shape[0]):
            indices = index_matrix[i]
            selected_states = hidden_states[i][indices]
            selected_hidden_states_list.append(selected_states)
        hidden_states = torch.stack(selected_hidden_states_list)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    index_matrix,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                    reduce_ratio=self.calculate_reduce_ratio(idx),
                )

            hidden_states = layer_outputs[0]
            index_matrix=layer_outputs[-1]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions,index_matrix] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def calculate_reduce_ratio(self, layer_num):
        if layer_num in ReduceEncoder.prune_layer_list:
            index_position = ReduceEncoder.prune_layer_list.index(layer_num)
            return ReduceEncoder.prune_ratio_list[index_position]
        else:
            return 0.0





def apply_patch(model,prune_layer_list,prune_ratio_list,index_matrix=None):
    """
    Applies double_efficient mechanism to this transformer.
    """
    BlockClass = None
    EncoderClass=None
    assert len(prune_layer_list)==len(prune_ratio_list)
    # Collect class names


    for module in model.vision_model.modules():
        if module.__class__.__name__ == "CLIPEncoderLayer":
            BlockClass = module.__class__
    for module in model.vision_model.modules():
        if module.__class__.__name__ == "CLIPEncoder":
            EncoderClass = module.__class__


    if BlockClass is None:
        print(
            "Error patching model: this model isn't a ClipModel transformer."
        )
        return
    if EncoderClass is None:
        print(
            "Error patching model: this model isn't a ClipModel transformer."
        )
        return

    ReduceEncoder.prune_layer_list=prune_layer_list
    ReduceEncoder.prune_ratio_list=prune_ratio_list
    if index_matrix!=None:
        ReduceEncoder.init_index_matrix=index_matrix.to(model.device)
    for module in model.vision_model.modules():
        if isinstance(module, BlockClass):
            module.__class__ = ReduceEncoderLayer
        elif isinstance(module, EncoderClass):
            module.__class__ = ReduceEncoder

def reverse_patch(model):
    BlockClass = None
    EncoderClass=None
    # Collect class names
    for module in model.vision_model.modules():
        if module.__class__.__name__ == "ReduceEncoderLayer":
            BlockClass = module.__class__
    for module in model.vision_model.modules():
        if module.__class__.__name__ == "ReduceEncoder":
            EncoderClass = module.__class__
    if BlockClass is None and EncoderClass is None:
        return
    for module in model.vision_model.modules():
        if isinstance(module, BlockClass):
            module.__class__ = CLIPEncoderLayer
        elif isinstance(module, EncoderClass):
            module.__class__ = CLIPEncoder

