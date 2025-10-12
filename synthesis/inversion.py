from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
import os
import torchvision
from double_efficient_vit import apply_patch, reverse_patch
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from tool import generate_masked_image_from_index
from .base import BaseSynthesis
from .hooks import DeepInversionHook
from .criterions import  get_image_prior_losses
from ._utils import ImagePool2, clip_images
@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass

def jitter_and_flip(inputs_jit, lim=1./8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)
    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit

class InversionSyntheiszer(BaseSynthesis):
    def __init__(self,args, teacher, img_size,
                 iterations=1000, lr_g=0.1,
                 synthesis_batch_size=128,
                 adv=0.0, bn=1., oh=1., tv=1e-5, l2=0.0,patch_size=16,
                 save_dir='', use_mask=False,transform=None,
                 normalizer=None, device='cpu',num_classes=64,c_abs_list=None,max_batch_per_class=20,use_fp16=False,add=True):
        super(InversionSyntheiszer, self).__init__(teacher, None)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        if args!=None:
            self.args=args
        else:
            self.args=None
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.c_abs_list=c_abs_list
        self.normalizer = normalizer
        self.num_classes=num_classes
        self.transform=transform
        self.use_fp16=use_fp16
        if args!=None and add==True:
            self.data_pool = ImagePool2(args=self.args,root=self.save_dir, num_classes=self.num_classes, transform=self.transform,max_batch_per_class=max_batch_per_class,use_mask=use_mask)
        self.synthesis_batch_size = synthesis_batch_size #num of samples per class

        # Scaling factors
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.tv = tv
        self.l2 = l2
        self.patch_size=patch_size
        self.device = device
        self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda(device)
        self.prior.eval()
        self.hooks = []
        for m in self.prior.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))

        self.autocast = dummy_ctx


    def synthesize(self, targets=None, student=None,c_num=5, add=True):
        self.synthesis_batch_size = len(targets) // c_num
        self.teacher=self.teacher
        self.teacher.eval()
        reverse_patch(self.teacher.clip_model)
        best_cost = 1e6
        best_inputs = None
        targets = torch.LongTensor(targets).to(self.device)
        data_type = torch.half if self.use_fp16 else torch.float
        inputs = torch.randn( size=[len(targets), *self.img_size], device=self.device,dtype=data_type).requires_grad_()

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])

        token_length=self.teacher.clip_model.vision_model.embeddings.num_positions
        index_matrix = torch.arange(token_length).repeat(inputs.shape[0], 1).to(inputs.device)#[[0,...,5,...],]
        best_inputs = inputs.data
        for it in range(self.iterations):
            if it==int(self.iterations*0.25):
                apply_patch(self.teacher.clip_model,prune_layer_list=self.args.prune_layer,prune_ratio_list=self.args.prune_ratio,index_matrix=index_matrix)
            inputs_aug = jitter_and_flip(inputs)
            if self.oh!=0:
                t_out = self.teacher(inputs_aug)
                loss_oh = F.cross_entropy(t_out, targets)
            else:
                loss_oh=0

            if len(self.hooks)!=0 and self.bn!=0:
                _ = self.prior(inputs_aug)
                loss_bn = sum([h.r_feature for h in self.hooks])
            else:
                loss_bn=0

            if self.tv!=0:
                loss_tv = get_image_prior_losses(inputs)
            else:
                loss_tv=0

            if self.l2 != 0:
                loss_l2 = torch.norm(inputs, 2)
            else:
                loss_l2 = 0

            loss = self.bn * loss_bn + self.oh * loss_oh  + self.tv * loss_tv + self.l2 * loss_l2

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std,use_fp16=self.use_fp16)

        # Get outputs from vision model with attention scores and index_matrix
        # Use get_image_outputs to properly handle return_dict parameter
        outputs = self.teacher.get_image_outputs(best_inputs, output_attentions=True, output_hidden_states=False, return_dict=False)
        # When return_dict=False and output_attentions=True, outputs = (hidden_states, all_attentions, index_matrix)
        attention_scores = torch.mean(outputs[-2][-1], dim=1)[:, 0, :][:, 1:]
        index_matrix = outputs[-1]
        reverse_patch(self.teacher.clip_model)


        # save best inputs
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
            if self.args.mask_ratio==-1:
                mask_best_inputs=generate_masked_image_from_index(x=best_inputs,index=index_matrix,attention_scores=attention_scores,patch_size=self.patch_size,mask_ratio=0.0)# automatically mask the inverted data based on the positions of remaining tokens
            else:
                mask_best_inputs = generate_masked_image_from_index(x=best_inputs, index=index_matrix,attention_scores=attention_scores,patch_size=self.patch_size,mask_ratio=self.args.mask_ratio) #additionally mask some remaining tokens at the last layer
        if add == True:
            self.data_pool.add(imgs=best_inputs,mask=False, c_abs_list=self.c_abs_list,synthesis_batch_size_per_class=self.synthesis_batch_size)
            self.data_pool.add(imgs=mask_best_inputs,mask=True, c_abs_list=self.c_abs_list,synthesis_batch_size_per_class=self.synthesis_batch_size)
        return best_inputs,mask_best_inputs

    def get_random_task(self):
        return self.data_pool.get_random_task()