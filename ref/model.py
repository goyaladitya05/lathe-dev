# Minimal SmolLM2-135M (Llama) reference. HF parameter names, HF init, lathe weight export.
#   python ref/model.py [--model smol135] --init data/smol135_init.bin   random init (seed 0)
#   python ref/model.py [--model smol135] --pretrained data/smol135_pretrained.bin   real SmolLM2 weights
import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAGIC, VERSION = 0x4854414C, 1  # "LATH"


@dataclass
class Config:
    name: str
    vocab: int
    dim: int
    layers: int
    heads: int
    kv_heads: int
    ffn: int
    rope_theta: float
    rms_eps: float
    init_std: float

    @property
    def head_dim(self):
        return self.dim // self.heads

    @staticmethod
    def load(model="smol135"):
        c = json.loads((DATA / model / "config.json").read_text())
        assert c["tie_word_embeddings"] and not c["attention_bias"] and c["hidden_act"] == "silu"
        assert c["rope_scaling"] is None and not c["rope_interleaved"]
        return Config(model, c["vocab_size"], c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                      c["num_key_value_heads"], c["intermediate_size"], float(c["rope_theta"]),
                      c["rms_norm_eps"], c["initializer_range"])


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).type_as(x)


def rope_table(cfg, T, device):
    inv = 1.0 / cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim)
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def rope(x, cos, sin):  # x [B, H, T, D], rotates pairs (i, i + D/2)
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.heads * hd, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c, hd = self.cfg, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, c.heads, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.kv_heads, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.kv_heads, hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)  # q head h -> kv head h // 3
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.down_proj = nn.Linear(cfg.ffn, cfg.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim, cfg.rms_eps)
        self.ckpt = False  # recompute each block in backward (fits fp32 runs in memory)

    def forward(self, idx, targets=None):
        cos, sin = rope_table(self.cfg, idx.shape[1], idx.device)
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = checkpoint(layer, x, cos, sin, use_reentrant=False) if self.ckpt else layer(x, cos, sin)
        logits = F.linear(self.norm(x), self.embed_tokens.weight)  # tied
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


def init_weights(model):
    # HF Llama init: N(0, initializer_range) for every matrix, ones for norms. CPU RNG with seed 0 for determinism.
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                p.normal_(0.0, model.cfg.init_std, generator=g)
            else:
                p.fill_(1.0)


def load_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    out = {}
    for k, v in header.items():
        if k != "__metadata__":
            s, e = v["data_offsets"]
            out[k] = torch.frombuffer(bytearray(raw[8 + n + s:8 + n + e]), dtype=dtypes[v["dtype"]]).view(v["shape"])
    return out


def load_pretrained(model):
    sd = {k.removeprefix("model."): v for k, v in load_safetensors(DATA / model.cfg.name / "model.safetensors").items()}
    model.load_state_dict(sd, strict=True)


def lathe_layout(sd, cfg):
    # Order lathe stores parameters in (src/train.cu): embed, per layer [qkv, o, gate|up, down, ln1, ln2], final norm.
    yield sd["embed_tokens.weight"]
    for i in range(cfg.layers):
        p = f"layers.{i}."
        yield torch.cat([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"],
                         sd[p + "self_attn.v_proj.weight"]])
        yield sd[p + "self_attn.o_proj.weight"]
        yield torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]])
        yield sd[p + "mlp.down_proj.weight"]
        yield sd[p + "input_layernorm.weight"]
        yield sd[p + "post_attention_layernorm.weight"]
    yield sd["norm.weight"]


def export(model, path):
    c = model.cfg
    header = struct.pack("<8i2f", MAGIC, VERSION, c.vocab, c.dim, c.layers, c.heads, c.kv_heads, c.ffn,
                         c.rope_theta, c.rms_eps)
    with open(path, "wb") as f:
        f.write(header.ljust(256, b"\0"))
        for t in lathe_layout(model.state_dict(), c):
            f.write(t.detach().float().cpu().contiguous().numpy().tobytes())


def from_lathe(flat, cfg):
    # Inverse of lathe_layout: flat fp32 tensor in lathe order -> HF-named views.
    hd, out, off = cfg.head_dim, {}, 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        off += n
        return flat[off - n:off].view(shape)

    out["embed_tokens.weight"] = take(cfg.vocab, cfg.dim)
    for i in range(cfg.layers):
        p = f"layers.{i}."
        q, k, v = take((cfg.heads + 2 * cfg.kv_heads) * hd, cfg.dim).split([cfg.heads * hd, cfg.kv_heads * hd,
                                                                            cfg.kv_heads * hd])
        out[p + "self_attn.q_proj.weight"], out[p + "self_attn.k_proj.weight"] = q, k
        out[p + "self_attn.v_proj.weight"] = v
        out[p + "self_attn.o_proj.weight"] = take(cfg.dim, cfg.heads * hd)
        out[p + "mlp.gate_proj.weight"], out[p + "mlp.up_proj.weight"] = take(2 * cfg.ffn, cfg.dim).split(cfg.ffn)
        out[p + "mlp.down_proj.weight"] = take(cfg.dim, cfg.ffn)
        out[p + "input_layernorm.weight"] = take(cfg.dim)
        out[p + "post_attention_layernorm.weight"] = take(cfg.dim)
    out["norm.weight"] = take(cfg.dim)
    assert off == flat.numel()
    return out


def load_lathe(model, path):
    model.load_state_dict(from_lathe(torch.from_numpy(np.fromfile(path, dtype=np.float32, offset=256)), model.cfg))


# Training hyperparameters shared by baseline, tests, and lathe (src/train.cu must match).
LR, BETAS, EPS, WD = 3e-4, (0.9, 0.95), 1e-8, 0.1


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fused=True)


def get_batch(tokens, i, B, T):
    # Batch i = contiguous span of B*T+1 tokens, wrapping over whole batches. lathe uses the same order.
    nb = (len(tokens) - 1) // (B * T)
    s = (i % nb) * B * T
    buf = torch.from_numpy(tokens[s:s + B * T + 1].astype(np.int64))
    return buf[:-1].view(B, T), buf[1:].view(B, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="export seeded random init to this path")
    ap.add_argument("--pretrained", help="export real SmolLM2 weights to this path")
    ap.add_argument("--model", default="smol135")
    args = ap.parse_args()
    cfg = Config.load(args.model)
    if args.init:
        model = Model(cfg)
        init_weights(model)
        export(model, args.init)
    if args.pretrained:
        model = Model(cfg)
        load_pretrained(model)
        export(model, args.pretrained)


if __name__ == "__main__":
    main()
