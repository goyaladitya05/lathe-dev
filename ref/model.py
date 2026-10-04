# Minimal SmolLM2-135M (Llama) reference. HF parameter names, HF init, lathe weight export.
#   python ref/model.py --init data/smol135_init.bin          random init (seed 0)
#   python ref/model.py --pretrained data/smol135_pretrained.bin   real SmolLM2-135M weights
import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAGIC, VERSION = 0x4854414C, 1  # "LATH"


@dataclass
class Config:
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
    def load(path=DATA / "smol135" / "config.json"):
        c = json.loads(Path(path).read_text())
        assert c["tie_word_embeddings"] and not c["attention_bias"] and c["hidden_act"] == "silu"
        assert c["rope_scaling"] is None and not c["rope_interleaved"]
        return Config(c["vocab_size"], c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
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

    def forward(self, idx, targets=None):
        cos, sin = rope_table(self.cfg, idx.shape[1], idx.device)
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = F.linear(self.norm(x), self.embed_tokens.weight)  # tied
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


def init_weights(model, seed=0):
    # HF Llama init: N(0, initializer_range) for every matrix, ones for norms. CPU RNG for determinism.
    g = torch.Generator().manual_seed(seed)
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
    sd = {k.removeprefix("model."): v for k, v in load_safetensors(DATA / "smol135" / "model.safetensors").items()}
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


def load_lathe(model, path):
    params = np.fromfile(path, dtype=np.float32, offset=256)
    sd, off = model.state_dict(), 0
    flat = []
    for t in lathe_layout(sd, model.cfg):
        flat.append(torch.from_numpy(params[off:off + t.numel()].copy()).view(t.shape))
        off += t.numel()
    assert off == params.size
    unsplit(model, flat)


def unsplit(model, flat):
    # Inverse of lathe_layout: list of lathe-order tensors -> HF state dict tensors (in place).
    c, hd = model.cfg, model.cfg.head_dim
    sd = model.state_dict()
    it = iter(flat)
    with torch.no_grad():
        sd["embed_tokens.weight"].copy_(next(it))
        for i in range(c.layers):
            p = f"layers.{i}."
            q, k, v = next(it).split([c.heads * hd, c.kv_heads * hd, c.kv_heads * hd])
            sd[p + "self_attn.q_proj.weight"].copy_(q)
            sd[p + "self_attn.k_proj.weight"].copy_(k)
            sd[p + "self_attn.v_proj.weight"].copy_(v)
            sd[p + "self_attn.o_proj.weight"].copy_(next(it))
            g, u = next(it).split([c.ffn, c.ffn])
            sd[p + "mlp.gate_proj.weight"].copy_(g)
            sd[p + "mlp.up_proj.weight"].copy_(u)
            sd[p + "mlp.down_proj.weight"].copy_(next(it))
            sd[p + "input_layernorm.weight"].copy_(next(it))
            sd[p + "post_attention_layernorm.weight"].copy_(next(it))
        sd["norm.weight"].copy_(next(it))


def get_batch(tokens, i, B, T):
    # Batch i = contiguous span of B*T+1 tokens, wrapping over whole batches. lathe uses the same order.
    nb = (len(tokens) - 1) // (B * T)
    s = (i % nb) * B * T
    buf = torch.from_numpy(tokens[s:s + B * T + 1].astype(np.int64))
    return buf[:-1].view(B, T), buf[1:].view(B, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", help="export seeded random init to this path")
    ap.add_argument("--pretrained", help="export real SmolLM2-135M weights to this path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = Config.load()
    if args.init:
        model = Model(cfg)
        init_weights(model, args.seed)
        export(model, args.init)
    if args.pretrained:
        model = Model(cfg)
        load_pretrained(model)
        export(model, args.pretrained)


if __name__ == "__main__":
    main()
