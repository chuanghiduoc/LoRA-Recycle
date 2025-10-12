import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from typing import  Optional



class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features,bias_indicator):
        self.bias_indicator=bias_indicator
        super(Linear_fw, self).__init__(in_features, out_features,bias_indicator)
        self.weight.fast = None  # Lazy hack to add fast weight link
        if bias_indicator==True:
            self.bias.fast = None

    def forward(self, x):
        if self.bias_indicator==True:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.linear(x, self.weight.fast,self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
            else:
                out = super(Linear_fw, self).forward(x)
            return out
        else:
            if self.weight.fast is not None:

                out = F.linear(x, self.weight.fast,bias=None)  # weight.fast (fast weight) is the temporaily adapted weight
            else:

                out = super(Linear_fw, self).forward(x)
            return out


class _LoRA_linear_clip(nn.Module): #implementation of lora
    lora_gate = True
    main_gate=True
    rate=1.0
    def __init__(
        self,
        linear: nn.Module,
        linear_a: nn.Module,
        linear_b: nn.Module,
    ):
        super().__init__()
        self.linear = linear
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = linear.in_features
        self.w_identity = torch.eye(linear.in_features)

    def forward(self, x):
        if _LoRA_linear_clip.main_gate:
            old = self.linear(x)  # B,N,3*org_C
        else:
            old=torch.zeros((x.shape[0],x.shape[1],self.linear.out_features)).to(x.device)
        if _LoRA_linear_clip.lora_gate:
            new = self.linear_b(self.linear_a(x))
            old[:, :, :] += _LoRA_linear_clip.rate*new
        else:
            pass
        return old


class LoRA_clipModel(nn.Module):

    def __init__(self, clip_model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_clipModel, self).__init__()
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(clip_model.vision_model.encoder.layers)))

        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in clip_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(clip_model.vision_model.encoder.layers):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.self_attn.q_proj #nn.linear
            w_v_linear = blk.self_attn.v_proj
            self.dim = w_q_linear.in_features
            #w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_a_linear_q=Linear_fw(self.dim,r,bias_indicator=False)
            #w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_b_linear_q=Linear_fw(r,self.dim,bias_indicator=False)
            #w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_a_linear_v=Linear_fw(self.dim,r,bias_indicator=False)
            #w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            w_b_linear_v=Linear_fw(r,self.dim,bias_indicator=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.self_attn.q_proj = _LoRA_linear_clip(
                w_q_linear,
                w_a_linear_q,
                w_b_linear_q,
            )
            blk.self_attn.v_proj = _LoRA_linear_clip(
                w_v_linear,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.clip_model = clip_model
        # if num_classes > 0:
        #     self.lora_vit.reset_classifier(num_classes=num_classes)
        self.visual_head = Linear_fw(self.clip_model.projection_dim, num_classes,bias_indicator=True) if num_classes > 0 else nn.Identity()

    def get_image_features(self,image):
        return self.clip_model.get_image_features(image)
    def get_text_features(self,text):
        text_features=self.clip_model.get_text_features(text)

    def get_image_outputs(self,image,output_attentions,output_hidden_states,return_dict):
        # CLIPVisionTransformer doesn't accept return_dict parameter in some versions
        # When return_dict=False is needed (for getting index_matrix from patched encoder),
        # we call the encoder directly
        if return_dict == False:
            # Get embeddings first
            hidden_states = self.clip_model.vision_model.embeddings(image)
            # Call encoder with return_dict=False to get tuple output
            encoder_outputs = self.clip_model.vision_model.encoder(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            return encoder_outputs
        else:
            # For return_dict=True, use the standard vision_model call
            return self.clip_model.vision_model(image,output_attentions=output_attentions,output_hidden_states=output_hidden_states)

    def get_image_logits(self,image):
        image_features=self.clip_model.get_image_features(image)
        image_logits=self.visual_head(image_features)
        return image_logits

    def forward(self,image):
        return self.get_image_logits(image)

    def clipModel_forward(self,input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):
        return self.clip_model(input_ids,pixel_values,attention_mask,position_ids,return_loss,output_attentions,output_hidden_states,return_dict)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.visual_head.in_features
        _out = self.visual_head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.clip_model.visual_head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.visual_head.in_features
        _out = self.visual_head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.visual_head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.visual_head.in_features
        _out = self.visual_head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.visual_head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)
    def return_lora_parameters(self):

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        _in = self.visual_head.in_features
        _out = self.visual_head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.visual_head.weight}

        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        return merged_dict

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        # Get the device of the current model
        model_device = next(self.clip_model.parameters()).device

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key).to(model_device)
                w_A_linear.weight = Parameter(saved_tensor)
                w_A_linear.weight.fast=None

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key).to(model_device)
                w_B_linear.weight = Parameter(saved_tensor)
                w_B_linear.weight.fast=None

            _in = self.visual_head.in_features
            _out = self.visual_head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key).to(model_device)
                self.visual_head.weight = Parameter(saved_tensor)
                self.visual_head.weight.fast=None
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            if w_A.weight.fast!=None:
                w_A.weight.fast=None
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            if w_B.weight.fast!=None:
                w_B.weight.fast=None