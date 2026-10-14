# Debug aid: patch ref/test.py's 8-bit state check to print the mismatched scales (lathe vs reference, in ulps) and
# the same reference with the sqrt on the GPU. Run in a scratch copy of the tree, then run ref/test.py --adam8.
from pathlib import Path
p = Path("ref/test.py"); s = p.read_text()
a = '            check(f"{name} scales, layer {l} (mismatches)", float((sc != d[f"s{name}.{l}"].view(-1)).sum()), 0)'
assert a in s
s = s.replace(a, a + '''
            got = d[f"s{name}.{l}"].view(-1)
            bad = (sc != got).nonzero().view(-1)
            if len(bad):
                ulp = (sc[bad].view(torch.int32) - got[bad].view(torch.int32)).abs()
                print(f"    {name} scales: {len(bad)} of {len(sc)} differ, ulps max {int(ulp.max())} median {float(ulp.float().median())}")
                for i in bad[:4].tolist():
                    print(f"      block {i}: ref {sc[i].item()!r} lathe {got[i].item()!r} amax ref {a[i].item()!r} lathe*448 {(got[i] * 448).item()!r}")
                if name == "u":
                    ag = v.cuda().sqrt().view(-1, 64).abs().amax(1).cpu()
                    print(f"    u with the sqrt on the GPU: {int((ag / 448 != got).sum())} mismatches; CPU vs GPU sqrt differ in {int((v.sqrt() != v.cuda().sqrt().cpu()).sum())} of {v.numel()} values")
                    sub = (v > 0) & (v < 1.1754944e-38)
                    print(f"    subnormal v: {int(sub.sum())}; torch flush-denormal state: {torch.get_flush_denormal() if hasattr(torch, 'get_flush_denormal') else '?'}")''')
p.write_text(s)
print("patched")
