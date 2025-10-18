import torch.nn as nn
import torch
from gemma3_270.config.gemma_config import GEMMA270_CONFIG



class RMSNorm(nn.Module):

    def __init__(self,embedding_dimension,eps=1e-6,bias=False):
        super().__init__()
        self.eps = eps
        # nn.Parameter, it tells PyTorch that this tensor should be treated as a parameter that needs to be optimized during training
        # (i.e., its values should be updated via backpropagation).
        self.scale = nn.Parameter(torch.zeros(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension)) if bias else None

    def forward(self,input_tensor:torch.Tensor):

        input_dtype = input_tensor.dtype
        input_f = input_tensor.float()

        # square of the elements in the row  and then take a mean  1/n (x1^2 + x2^2 + x3^2)
        var = input_f.pow(2).mean(dim=-1, keepdim=True)

        # normalize  x1 /  Mean Square root
        x_norm = input_f * torch.rsqrt(var + self.eps)

        # Multiple each element by (1 + shift)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            # add shift to each element
            out = out + self.shift.float()

        return out.to(input_dtype)





