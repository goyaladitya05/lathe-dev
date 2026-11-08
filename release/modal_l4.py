# lathe on a Modal L4 (sm_89, 24 GB): every correctness test (the 360M ones need ~10 GB of host RAM) and the v0
# result set measured on the L4. Code comes from build/modal/{main,fp8} (release/modal_l4.sh stages it).
#   release/modal_l4.sh [bench|fp8|all]   (MODAL_PROFILE picks the account)
# Logs: printed, and kept in the lathe-data volume under logs/ (modal volume get lathe-data logs/ build/modal/).
import re
import subprocess
import time
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent.parent / "build/modal"
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .apt_install("git", "make", "util-linux")
         .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cu130")
         .pip_install("torchao==0.18.0", "tokenizers", "pyarrow", "numpy")
         .env({"LIBRARY_PATH": "/usr/local/cuda/lib64/stubs"})  # link-time libcuda; the driver's at run time
         .add_local_dir(STAGE, "/src"))
vol = modal.Volume.from_name("lathe-data", create_if_missing=True)
app = modal.App("lathe-l4", image=image)
GPU = dict(gpu="L4", volumes={"/vol": vol}, cpu=4, memory=16384)  # CPU and memory are billed too: keep them small
CPU = dict(volumes={"/vol": vol}, cpu=4, memory=16384)
# The L4 is power-capped (72 W): clocks fall from ~2 GHz to ~0.9-1.5 GHz as it heats, so every row warms up for
# 400 steps (about 1-2 minutes) and is timed at the settled clock.
TIMING = "--warmup 400 --steps 40 --runs 5"


def setup(tree):
    # /work/<tree> with data/ -> /vol/data (shared by both trees and kept between runs)
    w = Path("/work") / tree
    subprocess.run(f"rm -rf {w} && mkdir -p /work /vol/data /vol/logs && cp -r /src/{tree} {w} && "
                   f"cp {w}/data/prep.py /vol/data/ && rm -rf {w}/data && ln -s /vol/data {w}/data", shell=True, check=True)
    return w


class Log:
    def __init__(self, name):
        self.path = Path("/vol/logs") / name
        self.f = open(self.path, "a")
        self.say(f"== {name} {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def say(self, line):
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def run(self, cmd, cwd):  # returns the output; a failing command is logged, not fatal
        self.say(f"$ {cmd}")
        p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout.splitlines():
            self.say(line)
        if p.returncode:
            self.say(f"# exit {p.returncode}")
        vol.commit()
        return p.stdout


def tune(log, w, m, b, t, *flags):
    suf = "".join({"ADAM8=1": "_adam8", "DW_LT=1": "_dwlt"}[f] for f in flags)
    log.run(f"make -s MODEL={m} BATCH={b} SEQ={t} {' '.join(flags)} && "
            f"build/train_{m}_b{b}_t{t}{suf} --tune 1 --steps 1 --data data/train.bin | grep -E 'gemm_tune|rror'", w)


@app.function(timeout=3600, **CPU)
def prep():  # downloads and weight export: no GPU
    w = setup("main")
    log = Log("prep.log")
    log.run("python data/prep.py --models smol135,smol360", w)
    for m in ("smol135", "smol360"):
        for k in ("init", "pretrained"):
            if not (w / f"data/{m}_{k}.bin").exists():
                log.run(f"python ref/model.py --model {m} --{k} data/{m}_{k}.bin", w)
    log.run("ls -la data/", w)


@app.function(timeout=4 * 3600, **GPU)
def tests():
    w, f8 = setup("main"), setup("fp8")
    log = Log("tests.log")
    log.run("python ref/test.py", w)
    tune(log, w, "smol135", 2, 512, "DW_LT=1")
    log.run("python ref/test.py --tuned --dw-lt", w)
    log.run("python ref/test.py --adam8", w)
    log.run("python ref/test.py --model smol360 --B 1 --T 256", w)
    log.run("python ref/test.py --adam8 --model smol360 --B 1 --T 256", w)
    tune(log, w, "smol360", 1, 256, "ADAM8=1", "DW_LT=1")
    log.run("python ref/test.py --adam8 --tuned --dw-lt --model smol360 --B 1 --T 256", w)
    log.run("python ref/test.py --fp8", f8)
    log.run("python ref/test.py --fp8 --adam8 --model smol360 --B 1 --T 256", f8)
    log.say("== tests done")


@app.function(timeout=3600, **GPU)
def fp8_check():
    log = Log("fp8_check.log")
    log.run("python ref/test.py --fp8 --adam8 --model smol360 --B 1 --T 256", setup("fp8"))


SHAPES = [(m, b, t) for m in ("smol135", "smol360") for b, t in ((1, 2048), (2, 1024), (4, 512))]


@app.function(timeout=3 * 3600, **GPU)
def bench(m: str, b: int, t: int):
    # One shape per container, every stack on the same card (Modal L4s differ by ~10% in measured peak), two sittings.
    # MFU against this card's measured bf16 GEMM peak (written into this copy's targets/rtx4050.h).
    w = setup("main")
    log = Log(f"bench_{m}_{b}x{t}.log")
    out = log.run("python ref/baseline.py --mode sol", w)
    peak = max(float(x) for x in re.findall(r"\d+\^3 \| ([\d.]+)", out))  # best size, as on the laptop
    h = w / "src/targets/rtx4050.h"
    h.write_text(re.sub(r"SOL_BF16_TFLOPS = [\d.]+", f"SOL_BF16_TFLOPS = {peak}", h.read_text()))
    log.say(f"# MFU below against {peak} TFLOPS (this L4, measured)")
    tune(log, w, m, b, t, "DW_LT=1")
    if m == "smol360":
        tune(log, w, m, b, t, "ADAM8=1", "DW_LT=1")
    rows = f"--model {m} --B {b} --T {t} {TIMING}"
    for sitting in (1, 2):
        log.say(f"== sitting {sitting}")
        for mode in ("eager", "compile"):
            if "failed" in log.run(f"python ref/baseline.py --mode {mode} {rows}", w):
                log.run(f"python ref/baseline.py --mode {mode} {rows}", w)  # first compile can die in capture
        log.run(f"python ref/baseline.py --mode lathe {rows}", w)
        log.run(f"python ref/baseline.py --mode lathe {rows} --tuned --dw-lt", w)
        if m == "smol360":
            log.run(f"python ref/baseline.py --mode lathe --adam8 {rows} --tuned --dw-lt", w)
    log.say("== bench done")


@app.local_entrypoint()
def main(only: str = "all"):
    prep.remote()
    calls = [bench.spawn(*sh_) for sh_ in SHAPES] if only in ("all", "bench") else []
    calls += [fp8_check.spawn()] if only in ("all", "fp8") else []
    for c in calls:
        c.get()


@app.function(timeout=3600, **GPU)
def sh(cmd: str, tree: str = "main"):
    # One-off command in a fresh copy of a tree (debugging): modal run release/modal_l4.py::sh --cmd "..."
    Log("sh.log").run(cmd, setup(tree))
