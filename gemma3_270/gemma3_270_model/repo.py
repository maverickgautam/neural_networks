import math

import torch
from torch._C.cpp import nn


class Rope():

    def __init__(self):
        super().__init__()


    def compute_rope_param(self,head_dim, theta_base=10_000,context_length=4096,dtype=torch.float32):
        assert head_dim % 2 == 0

        inv_freq = 1 / (theta_base ** (torch.arange(0,head_dim,2,dtype=dtype)[:, (head_dim // 2)].float() / head_dim))

        position = torch.arange(context_length,dtype=dtype)

        angles = position[:,None] * inv_freq[None,:]

        angles = torch.cat([angles,angles],dim=1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin


    def apply_rope(self,x,cos,sin):

        batch_size, num_heads, seq_len, head_dim = x.shape
        assert  head_dim % 2 == 0


        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]

        cos = cos[:seq_len,:].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len,:].unsqueeze(0).unsqueeze(0)

        rotate = torch.cat([-x2, x1], dim=-1)
        x_rotated = (x*cos) + (rotate * sin)

        return x_rotated.to(dtype=x.dtype)
