# Fill the README placeholders ({NAME}) from the release sitting log and BENCH.md; no number is typed by hand.
#   python3 release/fill_numbers.py LOG BENCH TEMPLATE OUT [TEMPLATE OUT ...]
#   templates: release/templates/README.md.in -> README.md, release/templates/release_README.md.in -> release/README.md
#   LOG    build/night/release_nov04.log (ROW / STRONG / front lines of the release sitting)
#   BENCH  BENCH.md (memory fronts, PyTorch's 360M front, the 2026-10-13 tuned rows of record)
# Prints every value with its source, fails on an unknown or unfilled placeholder, then lists any number in the output
# that does not occur verbatim in LOG or BENCH (the provenance check; derived ratios are listed as derived).
import re
import sys
from pathlib import Path

LOG, BENCH = Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text()
PAIRS = list(zip(sys.argv[3::2], sys.argv[4::2]))
ROOT = Path(__file__).resolve().parent.parent
V, SRC, DERIVED = {}, {}, set()


def put(name, value, source, derived=False):
    V[name], SRC[name] = value, source
    if derived:
        DERIVED.add(name)


def num(s):
    return float(s.replace(",", "").replace("%", ""))


def fmt(x):
    return f"{x:,.0f}"


def shape(s):  # "1x2048" -> "1 x 2048"
    return s.replace("x", " x ")


# ---- release sitting log
rows = {}  # (stack, shape) -> fields
for line in LOG.splitlines():
    if line.startswith("ROW |"):
        f = [x.strip() for x in line.split("|")]
        rows[(f[3], f[4])] = {"toks": f[5], "mfu": f[7], "proc": f[10], "date": f[1], "commit": f[2], "line": line}
strong = {}  # (shape "1 x 1024", switches) -> tok/s
for line in LOG.splitlines():
    m = re.match(r"STRONG \| (\d+ x \d+) \| ([^|]+?) \| .*tok/s (\d+) ", line)
    if m:
        strong[(m.group(1), m.group(2).strip())] = int(m.group(3))
front = {}
for line in LOG.splitlines():
    m = re.match(r"front \| smol360 \| ([^|]+?) \| ([^|]+?) \|", line)
    if m:
        front[m.group(1).strip()] = m.group(2).strip()

E, C, L, LT = "torch eager", "torch compile", "lathe", "lathe (tuned gemms, cuBLASLt dW)"
L8 = "lathe (8-bit state, tuned gemms, cuBLASLt dW) (smol360)"
S135 = ["1x2048", "2x1024", "4x512"]
for i, s in enumerate(S135, 1):
    for key, stack in (("E", E), ("C", C), ("L", L), ("LT", LT)):
        r = rows[(stack, s)]
        put(f"{key}{i}", f"{r['toks']} ({r['mfu']})", f"log ROW {stack} {s}")
lt = [num(rows[(LT, s)]["toks"]) for s in S135]
vs_c = [100 * (a / num(rows[(C, s)]["toks"]) - 1) for a, s in zip(lt, S135)]
vs_e = [100 * (a / num(rows[(E, s)]["toks"]) - 1) for a, s in zip(lt, S135)]
mfu = [num(rows[(LT, s)]["mfu"]) for s in S135]
put("CMP_COMPILE", f"{min(vs_c):.1f}-{max(vs_c):.1f}%", "log: lathe tuned / torch compile - 1, 135M rows", True)
put("CMP_EAGER", f"{min(vs_e):.1f}-{max(vs_e):.1f}%", "log: lathe tuned / torch eager - 1, 135M rows", True)
put("CMP_MFU", f"{min(mfu):.1f}-{max(mfu):.1f}%", "log: lathe tuned MFU, 135M rows")
put("MEM_L", f"{rows[(L, '1x2048')]['proc']} MiB", "log ROW lathe 1x2048 proc MiB")
put("MEM_C", f"{rows[(C, '1x2048')]['proc']}-{rows[(C, '4x512')]['proc']} MiB", "log ROW torch compile proc MiB")
put("MEM_E", f"{rows[(E, '1x2048')]['proc']} MiB", "log ROW torch eager proc MiB")
put("SITTING_DATE", rows[(L, "1x2048")]["date"], "log ROW date")
put("SITTING_COMMIT", rows[(L, "1x2048")]["commit"].split("-g")[-1], "log ROW commit")

S360 = ["1x1024", "1x1536", "2x1152", "1x2048", "9x256"]
CA, CKA = "--compile --adam8", "--compile --ckpt --adam8"
table, ratios, best = [], [], (0, None, None)
for s in S360:
    r = rows[(L8, s)]
    ca, cka = strong.get((shape(s), CA)), strong.get((shape(s), CKA))
    torch_best = max(v for (sh, _), v in strong.items() if sh == shape(s))
    for (sh, sw), v in strong.items():
        if sh == shape(s) and v > best[0]:
            best = (v, sh, sw)
    ratios.append(num(r["toks"]) / torch_best)
    table.append(f"| {shape(s)} | {fmt(ca) if ca else '-'} | {fmt(cka) if cka else '-'} | **{r['toks']} ({r['mfu']})** |")
put("T360_ROWS", "\n".join(table), "log STRONG lines and ROW lathe 8-bit tuned, 360M")
put("T360_RATIO", f"{min(ratios):.2f}-{max(ratios):.2f}x", "log: lathe 8-bit tuned / best STRONG at each 360M shape", True)
put("T360_BEST", fmt(best[0]), f"log STRONG max ({best[1]}, {best[2]})")
names = {"--compile --ckpt --adam8": "torch.compile + checkpointing + AdamW8bit", "--compile --adam8": "torch.compile + AdamW8bit",
         "--adam8": "eager + AdamW8bit"}
put("T360_BEST_AT", f"{names.get(best[2], best[2])} at {best[1]}", "log STRONG max")
r = rows[(L8, "2x1152")]
put("H360_TOKS", r["toks"], "log ROW lathe 8-bit tuned 2x1152")
put("H360_MFU", r["mfu"], "log ROW lathe 8-bit tuned 2x1152")
put("F360_FUSED", f"{int(front['fused']):,}", "log front smol360 fused (eager, fp32 AdamW in the backward)")

# ---- BENCH.md


def bench_cell(pattern, what):
    m = re.search(pattern, BENCH, re.M)
    if not m:
        sys.exit(f"fill_numbers.py: BENCH.md has no {what} ({pattern})")
    return m.group(1)


sec = BENCH[BENCH.index("## Strongest PyTorch at 360M"):]
put("F360_EAGER", re.search(r"^\| adam8 \(torchao AdamW8bit\) \| ([\d,]+) \|", sec, re.M).group(1), "BENCH Strongest PyTorch at 360M: adam8")
put("F360_COMPILE", re.search(r"^\| compile\+adam8 \| ([\d,]+) \|", sec, re.M).group(1), "BENCH Strongest PyTorch at 360M: compile+adam8")
put("M135_E", bench_cell(r"^\| torch eager \| ([\d,]+) \| [\d,]+ \(4x512\)", "135M eager front"), "BENCH 135M memory front: torch eager")
put("M135_C", bench_cell(r"^\| torch compile \| ([\d,]+) \| [\d,]+ \(7x512\)", "135M compile front"), "BENCH 135M memory front: torch compile")
put("M135_L", bench_cell(r"^\| lathe \| ([\d,]+) \| [\d,]+ \(11x512", "135M lathe front"), "BENCH 135M memory front: lathe")
put("M135_L8", bench_cell(r"^\| 8-bit \| ([\d,]+) \(", "135M 8-bit front"), "BENCH 135M memory front: 8-bit state")
put("M360_L", bench_cell(r"^\| fp32 state \| ([\d,]+) \(1x", "360M fp32 front"), "BENCH 360M: fp32 state largest step")
put("M360_L8", bench_cell(r"^\| 8-bit state \| ([\d,]+) \(", "360M 8-bit front"), "BENCH 360M: 8-bit state largest step")

# Fastest configurations: the 2026-10-13 tuned rows of record (BENCH "rows of record on the winner").
rec = []
for line in BENCH.splitlines():
    f = [x.strip() for x in line.split("|")]
    if len(f) > 8 and f[1] == "2026-10-13" and "tuned gemms, cuBLASLt dW" in f[3]:
        rec.append({"stack": f[3], "shape": f[4], "toks": f[5], "mfu": f[7], "date": f[1]})


def fastest(model360, eightbit, min_t):
    c = [r for r in rec if ("smol360" in r["stack"]) == model360 and ("8-bit" in r["stack"]) == eightbit
         and int(r["shape"].split("x")[1]) >= min_t]
    return max(c, key=lambda r: num(r["toks"]))


fr = []
for model, label, st, m360, e8, t in (("SmolLM2-135M", "context >= 1024", "fp32", False, False, 1024),
                                      ("SmolLM2-135M", "any context", "fp32", False, False, 0),
                                      ("SmolLM2-360M", "context >= 1024", "8-bit", True, True, 1024),
                                      ("SmolLM2-360M", "any context", "8-bit", True, True, 0)):
    r = fastest(m360, e8, t)
    fr.append(f"| {model} | {label} | {st} | {shape(r['shape'])} | {r['toks']} | {r['mfu']} |")
put("FAST_ROWS", "\n".join(fr), "BENCH 2026-10-13 rows of record, lathe tuned + cuBLASLt dW, fastest per group")
put("FAST_DATE", rec[0]["date"], "BENCH 2026-10-13 rows of record")

# Other GPUs: the final sitting table of each Colab card, cells copied as BENCH records them.
def final_table(marker):
    lines = BENCH.splitlines()
    k = next(i for i, l in enumerate(lines) if marker in l)
    date = None
    for l in reversed(lines[:k + 1]):  # the nearest date at or above the marker, within its section
        m = re.search(r"\d{4}-\d{2}-\d{2}", l)
        if m and date is None:
            date = m.group(0)
        if l.startswith("## "):
            break
    tbl = []
    for l in lines[k + 1:]:
        if l.startswith("|"):
            tbl.append([c.strip().replace("**", "") for c in l.strip("|").split("|")])
        elif tbl:
            break
    head, rows = tbl[0], tbl[2:]
    col = lambda pred: next((i for i, h in enumerate(head) if pred(h)), None)
    cols = [col(lambda h: h == "eager"), col(lambda h: h == "compile"), col(lambda h: h == "lathe"),
            col(lambda h: "tuned" in h and "8-bit" not in h), col(lambda h: "8-bit" in h and "tuned" in h),
            col(lambda h: "vs compile" in h)]
    out = ["| model | batch x context | PyTorch eager | torch.compile | lathe | lathe, tuned | lathe, 8-bit state, tuned | faster lathe vs torch.compile |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| SmolLM2-" + r[0] + " | " + shape(r[1]) + " | " + " | ".join(r[c] if c is not None else "" for c in cols) + " |")
    return date, "\n".join(out), k + 1


for key, marker in (("L4", "**L4 final sitting**"), ("A100", "**A100 final sitting**"), ("G4", "**G4 final sitting**")):
    d, t, line = final_table(marker)
    put(f"{key}_DATE", d, f"BENCH {marker} (line {line}): nearest date above it in its section")
    put(f"{key}_ROWS", t, f"BENCH {marker} table, cells verbatim")

# Memory fronts: the dates of the BENCH sections the front values come from.
front_dates = []
for anchor in ("SmolLM2-135M, largest tokens per step", "135M memory front, 8-bit state", "| 8-bit state | 2,304 (", "## Strongest PyTorch at 360M"):
    k = BENCH.index(anchor)
    m = list(re.finditer(r"\d{4}-\d{2}-\d{2}", BENCH[:k]))
    front_dates.append(m[-1].group(0))
put("FRONT_FIRST", min(front_dates), "BENCH: nearest date before each memory-front table (" + ", ".join(front_dates) + ")")
put("FRONT_LAST", max(front_dates), "BENCH: nearest date before each memory-front table")

# What was tried and dropped: BENCH rows of the reverted changes.
def pcts(anchor):
    line = next(l for l in BENCH.splitlines() if anchor in l)
    cell = line.split("|")[2]
    return [float(x) for x in re.findall(r"[+-]\d+\.\d+(?=%)", cell)]
def span(v, digits=2):
    lo, hi = sorted(v, key=abs)[0], sorted(v, key=abs)[-1]
    return f"{lo:+.{digits}f} to {hi:+.{digits}f}%"
put("DROP_HEADS", span(pcts("(e055) |")), "BENCH chisel follow-ups table, e055 row")
put("DROP_DQ", span(pcts("(e061) |")), "BENCH chisel follow-ups table, e061 row")
put("DROP_DSBUF", span(pcts("(e062) |")), "BENCH chisel follow-ups table, e062 row")
atomic = [float(m) for m in re.findall(r"^\| \d+x\d+ \| [\d,]+ / [\d,]+ \| [\d,]+ / [\d,]+ \| ([+-]\d+\.\d+)%", BENCH, re.M)]
put("DROP_ATOMIC", span(atomic, 1), "BENCH atomic-dq table, gain column")

# ---- repository facts (not measurements)
peak = re.search(r"SOL_BF16_TFLOPS = ([\d.]+)", (ROOT / "src/targets/rtx4050.h").read_text()).group(1)
put("PEAK", peak, "src/targets/rtx4050.h SOL_BF16_TFLOPS")
loc = sum(len((ROOT / p).read_text().splitlines()) for p in ("src/train.cu", "src/kernels.cuh"))
put("LOC", f"{round(loc, -2):,}", f"wc -l src/train.cu src/kernels.cuh = {loc}")

for k in sorted(V):
    print(f"{k:14s} {V[k] if chr(10) not in V[k] else '(table)'}    <- {SRC[k]}")

# ---- fill
for tpl, out in PAIRS:
    text = Path(tpl).read_text()
    unknown = sorted(set(re.findall(r"\{([A-Z0-9_]+)\}", text)) - set(V))
    if unknown:
        sys.exit(f"fill_numbers.py: {tpl}: no value for {unknown}")
    text = re.sub(r"\{([A-Z0-9_]+)\}", lambda m: V[m.group(1)], text)
    Path(out).write_text(text)
    # provenance: every comma number and every percentage must occur in LOG or BENCH, unless derived here
    derived = {x for k in DERIVED for x in re.findall(r"[\d,.]+", V[k])}
    loose = []
    for tok in sorted(set(re.findall(r"\d{1,3}(?:,\d{3})+|\d+\.\d+%?|\d+%", text))):
        if tok not in LOG and tok not in BENCH and tok.rstrip("%") not in derived and tok not in peak:
            loose.append(tok)
    print(f"\nwrote {out}; numbers not found verbatim in LOG/BENCH (check by hand): {loose or 'none'}")
