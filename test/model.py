import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F
import math

class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
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
    

def precompute_freqs_cis(dim:int, end:int=int(32*1024), repe_base:float=1e6, rope_scaling:Optional[dict]=None):
    freqs = 1.0 / (rope_scaling['base'] ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        if end / orig_max > 1.0:
                corr_dim = next(
                    (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max),
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
    
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    
def repeat_kv(x:torch.Tensor, n_rep:int):
    bs, slen, num_key_value_heads, head_dim = x.size()
    if n_rep == 1:
        return x    
    return (x.unsqueeze(2).expand(bs, slen, n_rep, num_key_value_heads, head_dim)
        .reshape(bs, slen, n_rep * num_key_value_heads, head_dim))


class Attention(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()
        self.key_value_heads = config.num_key_value_heads if config.num_key_value_heads is None else config.num_attention_heads

        assert config.num_attention_heads % config.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


        self.atten_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.dropout = config.dropout

        self.flash_attention = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attention
        #电脑磁盘IO处理方式，能够加速大规模注意力计算

    def forward(self, x:torch.Tensor, 
                position_embedding: Tuple[torch.Tensor, torch.Tensor], 
                past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                use_cache=False, 
                attention_mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        
        bs, seq_len, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_key_value = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if self.flash_attention and seq_len > 1 and(attention_mask is None or torch.all(attention_mask==0)):
            attn_mask = (None if attention_mask is None else attention_mask.view(bs, 1, 1, -1).expand(bs, self.n_local_heads,
                            seq_len, -1).bool())
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)


        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(scores.device).unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e-9
                scores = scores + extended_attention_mask


            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.atten_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.n_local_heads * self.head_dim)
        output = self.resid_dropout(self.o_proj(output))        
        return output, past_key_value


         
