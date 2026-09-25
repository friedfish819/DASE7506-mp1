"""Student model: GPT with switchable RMSNorm, RoPE and SwiGLU.

Config flags:
- use_rmsnorm (default True):  RMSNorm instead of LayerNorm.
- use_rope    (default True):  RoPE instead of learned absolute positions.
- use_swiglu  (default False): SwiGLU instead of GELU MLP.

All flags can be turned off independently for ablation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learnable per-channel scale."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return x * torch.rsqrt(rms) * self.weight


def build_rope_cache(head_dim, max_context, base=10000.0):
    """Precompute cos/sin tables for RoPE. head_dim must be even."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
    positions = torch.arange(max_context, dtype=torch.float32)
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    """Rotate query/key tensors with RoPE. x: [b, h, t, d]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_half = cos[..., 0::2]
    sin_half = sin[..., 0::2]
    rot_even = x_even * cos_half - x_odd * sin_half
    rot_odd = x_even * sin_half + x_odd * cos_half
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    """Gated MLP. Hidden size chosen to roughly match a 4x GELU MLP."""

    def __init__(self, width, hidden):
        super().__init__()
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, width, heads, norm_cls, use_rope, use_swiglu):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.use_rope = use_rope
        self.norm1 = norm_cls(width)
        self.norm2 = norm_cls(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        if use_swiglu:
            hidden = (8 * width) // 3          # ~2.67x, matches GELU-4x param count
            self.mlp = SwiGLU(width, hidden)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
            )

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        self.heads = heads
        self.head_dim = width // heads
        if self.head_dim % 2 != 0:
            raise ValueError('head_dim must be even for RoPE.')

        self.use_rmsnorm = config.get('use_rmsnorm', True)
        self.use_rope = config.get('use_rope', True)
        self.use_swiglu = config.get('use_swiglu', False)
        norm_cls = RMSNorm if self.use_rmsnorm else nn.LayerNorm

        self.token = nn.Embedding(config['vocab'], width)
        if not self.use_rope:
            self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            [Block(width, heads, norm_cls, self.use_rope, self.use_swiglu)
             for _ in range(config['depth'])]
        )
        self.norm = norm_cls(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

        if self.use_rope:
            cos, sin = build_rope_cache(self.head_dim, self.context)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if not self.use_rope:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        length = ids.shape[1]
        if self.use_rope:
            cos = self.rope_cos[:length]
            sin = self.rope_sin[:length]
        else:
            cos = sin = None
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, no future access."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """Factory used by common.make_model; must expose context=256."""
    return StudentGPT(config)