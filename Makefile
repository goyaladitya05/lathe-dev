# make MODEL=smol135 BATCH=1 SEQ=2048  ->  build/train_smol135_b1_t2048 (model and shapes are compile-time)
# STEP_TIMES=1 -> build/..._timed: per-op GPU time per step (eager steps, an event after every launch)
MODEL ?= smol135
BATCH ?= 1
SEQ ?= 2048
STEP_TIMES ?=
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
OUT := build/train_$(MODEL)_b$(BATCH)_t$(SEQ)$(if $(STEP_TIMES),_timed)
# chisel's SASS dW cores for this model and token count, when built (src/chisel/, from chisel's gen/lathe_export.py)
CHISEL_DW_H := chisel/dw_$(MODEL)_n$(shell echo $$(($(BATCH) * $(SEQ)))).h
CHISEL_ATT_H := chisel/att_$(MODEL)_b$(BATCH)_t$(SEQ).h
NVFLAGS := -O3 -std=c++17 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra -Xptxas -warn-spills
LDFLAGS := -lcublasLt -lcuda -Xlinker -rpath,$(CUDA)/lib64

$(OUT): src/train.cu src/kernels.cuh src/models/$(MODEL).h src/targets/sm89.h $(wildcard src/$(CHISEL_DW_H)) $(wildcard src/$(CHISEL_ATT_H))
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(if $(STEP_TIMES),-DSTEP_TIMES) -DMODEL='"$(MODEL)"' -DMODEL_H='"models/$(MODEL).h"' \
		-DCHISEL_DW_H='"$(CHISEL_DW_H)"' -DCHISEL_ATT_H='"$(CHISEL_ATT_H)"' -DBATCH=$(BATCH) -DSEQ=$(SEQ) -Isrc $< -o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: clean
