import torch
import math
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).typed_as(x)


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if rope_scaling is not None:
        original_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )

        if end / original_max > 1.0:
            corr_dim = next(
                (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > original_max),
                dim // 2,
            )

            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(
                dim // 2 - 1, 1
            )

            beta = beta_slow + (beta_fast - beta_slow) * power

            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor,
            )

            freqs = freqs * scale

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        return freqs_cos, freqs_sin
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed