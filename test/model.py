import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return x * self._norm(x.float()) * self.weight.type_as(x)
    

