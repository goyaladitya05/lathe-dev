# Debug aid: next to each attention-backward kernel check, print PyTorch's own bf16 SDPA backward error against the
# same fp32 reference on the same inputs. Run in a scratch copy of the tree, then run ref/test.py.
from pathlib import Path
p = Path("ref/test.py"); s = p.read_text()
a = '''            check(f"attention + rope bwd {name}", rel(got, w), KERNEL_TOL)'''
assert a in s
s = s.replace(a, a + '''
        qkv16 = f("qkv").to(torch.bfloat16).view(B, T, -1, cfg.head_dim).transpose(1, 2)
        q16, k16, v16 = (t.detach().clone().requires_grad_() for t in qkv16.split([cfg.heads, cfg.kv_heads, cfg.kv_heads], 1))
        o16 = F.scaled_dot_product_attention(q16, k16, v16, is_causal=True, enable_gqa=True)
        o16.backward(f("datt").to(torch.bfloat16).view(B, T, cfg.heads, cfg.head_dim).transpose(1, 2))
        flat = lambda t: t.float().transpose(1, 2).reshape(B * T, -1)
        for name, t, w in zip(["dq", "dk", "dv"], [unrope(flat(q16.grad)), unrope(flat(k16.grad)), flat(v16.grad)], [unrope(dq), unrope(dk), dv]):
            print(f"         torch bf16 SDPA bwd {name}: {rel(t, w):.2e}")''')
p.write_text(s)
print("patched")
