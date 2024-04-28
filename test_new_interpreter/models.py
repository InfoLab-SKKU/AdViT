import torch
import torch.nn as nn
from functools import partial
import math
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, _cfg
from timm.models.registry import register_model
from torch.autograd import Variable
from torch.distributions import Bernoulli
from timm.models.vision_transformer import VisionTransformer, _cfg

__all__ = [
    'interp_deit_small_patch16_224',
    'interp_deit_base_patch16_224'
]

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    m.total_ops += torch.DoubleTensor([int(num_mul)])

class Attention(nn.Module):
    def __init__(self, orig_attn):
        super().__init__()
        self.num_heads = orig_attn.num_heads
        self.scale = orig_attn.scale

        self.qkv = orig_attn.qkv
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop
        self.mat = matmul()

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) BxheadxNxdim

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        mask = mask.unsqueeze(1).expand_as(attn)
        # fill the attn with -inf
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, -1, _stacklevel=10)
        attn = attn.masked_fill(mask == 0, 0)
            
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class Block(nn.Module):
    def __init__(self, orig_blk):
        super().__init__()
        self.norm1 = orig_blk.norm1
        self.attn = Attention(orig_blk.attn)
        self.norm2 = orig_blk.norm2
        self.mlp = orig_blk.mlp

    def forward(self, x, mask):
        attn_x = self.attn(self.norm1(x), mask)
        x = x + attn_x
        x = x + self.mlp(self.norm2(x))
        return x

class PerBlockAgent(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.qk = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.mat = matmul()
        self.act = nn.Sigmoid()

    def forward(self, x, prev_policy=None):
        # Prev_mask: BxN
        B, N, C = x.shape
        if prev_policy == None:
            prev_policy = torch.ones(B, N)
        x = x * prev_policy.unsqueeze(2)
        x = self.norm(x)

        qk = self.qk(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k = qk[0], qk[1] # B x num_head x N x C

        attn = self.mat(q[:,:,:1], k.transpose(-2, -1)) * self.scale
        attn = self.act(attn)
        attn = attn.squeeze(2).mean(1)
        attn = attn * prev_policy
        return attn

class AttnGroup(nn.Module):
    def __init__(self, dim, block_list=None):
        super().__init__()
        self.block_list = block_list
        self.agent = PerBlockAgent(dim=dim, num_heads=6)

    def sample_pos(self, x, policy):
        B, N, C = x.shape
        x = policy.view(policy.size(0), policy.size(1), 1) * x
        mask_matrix = policy.unsqueeze(2) @ policy.unsqueeze(1)
        return x, mask_matrix.int()

    def forward_agent(self, x, prev_policy=None, sample=False, mask=False):
        probs = self.agent(x, prev_policy)
        policy_map = probs.data.clone()
        policy_map[policy_map<=0.5] = 0.0
        policy_map[policy_map>0.5] = 1.0
        policy_map = Variable(policy_map)
        policy_map[:, :1] = 1
        # probs = probs*0.8 + (1-probs)*(1-0.8)
        
        if sample:
            if prev_policy is None:
                prev_policy = 1
            self.distr = Bernoulli(probs)
            policy = self.distr.sample()
            policy = policy * prev_policy
            policy[:, :1] = 1
            if mask:
                policy = prev_policy
            return policy, (probs, self.distr)
        if mask:
            policy_map = prev_policy
        return policy_map, probs
    
    def forward_features(self, x, policy):
        x, mask = self.sample_pos(x, policy)
        for blk in self.block_list:
            x = blk(x, mask)
            
        return x

    def forward(self, x, prev_policy=None, sample=False, mask=False):
        policy, probs = self.forward_agent(x, prev_policy=prev_policy, sample=sample, mask=mask)
        x = self.forward_features(x, policy)
        return x, policy, probs


class PerBlock_VisionTransformer(nn.Module):
    def __init__(self, orig_vit, group_size, error_penalty, penalty_list):
        super().__init__()
        self.num_classes = orig_vit.num_classes
        self.embed_dim = orig_vit.embed_dim
        self.patch_embed = orig_vit.patch_embed
        self.cls_token = orig_vit.cls_token
        self.pos_embed = orig_vit.pos_embed
        self.blocks = orig_vit.blocks
        self.norm = orig_vit.norm
        self.head = orig_vit.head
        self.group_size = group_size
        self.error_penalty = error_penalty
        self.penalty_list = penalty_list
        self.get_benefit(group_size)
        self.get_groups(group_size)

    def get_benefit(self, group_size):
        self.benefit = []
        acc_s = 0
        for s in group_size[::-1]:
            acc_s += s
            self.benefit.append(acc_s / sum(group_size))
        self.benefit = self.benefit[::-1]
    
    def get_groups(self, group_size):
        assert sum(group_size) == len(self.blocks)
        start_blk = 0
        self.groups = nn.ModuleList()
        for size in group_size:
            block_list = nn.ModuleList([Block(self.blocks[i]) for i in range(start_blk, start_blk+size)])
            group = AttnGroup(self.embed_dim, block_list)
            self.groups.append(group)
            start_blk = start_blk+size

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()     

    def forward_embedding(self, images):
        B = images.shape[0]
        x = self.patch_embed(images)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        return x
    
    def forward_inference_features(self, x, cl_step=-1):
        policys = []
        probs = []
        prev_policy = None
        mask = False
        for step, grp in enumerate(self.groups):
            if step > cl_step and cl_step != -1:
                mask = True
            else:
                mask = False
            x, policy, prob = grp(x, sample=False, prev_policy=prev_policy, mask=mask)
            policys.append(policy)
            probs.append(prob)
            prev_policy = policy
        return policys, probs, x

    def get_flops(self, policys, group_size):
        flops = 1
        # group_size = [1,0,0]
        for (policy, size) in zip(policys, group_size):
            policy_sum = policy.sum(1) / policy.size(1)
            flops = flops - (1 - policy_sum) * size / sum(group_size)
        return flops

    def forward(self, x, labels=False, cl_step=-1):
        x = self.forward_embedding(x)
        policys, probs, x = self.forward_inference_features(x, cl_step)
        x = self.norm(x)[:, 0]
        preds = self.head(x)
        flops = self.get_flops(policys, self.group_size)
        return preds, flops, probs            



@register_model
def interp_deit_base_patch16_224(device, pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device)
    perblock_dyt = PerBlock_VisionTransformer(model, [4,4,4], -1, [3, 5, 8]).to(device)
    return perblock_dyt
    
    
@register_model
def interp_deit_small_patch16_224(device, pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device)
    perblock_dyt = PerBlock_VisionTransformer(model, [4,4,4], -1, [3, 5, 8]).to(device)
    return perblock_dyt