# Edge-Guided Attention Module
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, x3, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        seq_len3 = x3.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x3).view(batch_size, seq_len3, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output

class EGA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EGA, self).__init__()
        self.cross_att = CrossAttention(in_dim1=256, in_dim2=256, k_dim=96, v_dim=96, num_heads=1)
        self.raw_weights\
            = nn.Parameter(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32), requires_grad=True)

    # 试一下这个
    def forward(self, modality1_info, modality2_info, modality3_info, edge_info):
        modality1_info = modality1_info * edge_info
        modality2_info = modality2_info * edge_info
        modality3_info = modality3_info * edge_info

        weights = F.softmax(self.raw_weights, dim=0)

        modality1_info = modality1_info * weights[0]
        modality2_info = modality2_info * weights[1]
        modality3_info = modality3_info * weights[2]

        b, c, h, w = modality3_info.size()

        modality1_patch = rearrange(modality1_info, 'b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=16, p2=16)
        modality2_patch = rearrange(modality2_info, 'b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=16, p2=16)
        modality3_patch = rearrange(modality3_info, 'b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=16, p2=16)

        fusion_feature = self.cross_att(modality1_patch, modality2_patch, modality3_patch)

        restored_patches = rearrange(fusion_feature, 'b (c h w) (p1 p2) -> b (c h w) p1 p2', h=h // 16,w=w // 16, p1=16, p2=16)
        fusion_feature = rearrange(restored_patches, 'b (c ph pw) p1 p2 -> b c (ph p1) (pw p2)', c=c, ph=h // 16,pw=w // 16)

        return fusion_feature