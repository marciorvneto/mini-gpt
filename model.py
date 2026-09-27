from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int     = 16_384
    context_length: int = 512

    n_layers: int       = 12
    n_heads: int        = 8
    d_model: int        = 512
    d_ff: int           = 2048


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(
                config.d_model,
                3 * config.d_model,
                )

        self.out_proj = nn.Linear(
                config.d_model,
                config.d_model,
                )


    def forward(self, x):
        B, T, C = x.shape

        # qkv = X * [Wq1, Wq1,... | Wk1, Wk2,... | Wv1, Wv2,...]
        # (B, T, C = d_model) * (d_model, 3 * d_model) -> (B, T, 3*d_model)
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, 2)
        # 3 x (B, T, d_model)

        # (B, T, d_model) -> (B, T, n_heads, head_dim)
        # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(
                B, T,
                self.n_heads,
                self.head_dim
                ).transpose(1,2)
        k = k.view(
                B, T,
                self.n_heads,
                self.head_dim
                ).transpose(1,2)
        v = v.view(
                B, T,
                self.n_heads,
                self.head_dim
                ).transpose(1,2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        # (B, n_heads, T, head_dim) -> (B, T, C = d_model)
        y = y.transpose(1,2)
        y = y.view(B,T,C)

        return self.out_proj(y)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.in1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)

        self.in2  = nn.LayerNorm(config.d_model)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.in1(x))
        x = x + self.mlp(self.in2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.d_model
                )
        self.position_embedding = nn.Embedding(
                config.context_length,
                config.d_model
                )
        self.blocks = nn.ModuleList([
            Block(config)
            for _ in range(config.n_layers)
            ])

        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
                config.d_model,
                config.vocab_size,
                bias=False
                )
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.config.context_length}"
            )

        positions = torch.arange(T, device=idx.device)

        tok = self.token_embedding(idx)
        pos = self.position_embedding(positions)

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                    )
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01
                    )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                    )



