# make MODEL=smol135 BATCH=1 SEQ=2048  ->  build/train_smol135_b1_t2048 (model and shapes are compile-time)
# STEP_TIMES=1 -> build/..._timed: per-op GPU time per step (eager steps, an event after every launch)
# ADAM8=1 -> build/..._adam8: 8-bit AdamW moments for the layer matrices (BENCH.md)
# DW_LT=1 -> build/..._dwlt: layer weight grads by cuBLASLt instead of own / chisel kernels
# TARGET=a100 -> build/..._a100: the GPU (src/targets/<TARGET>.h); rtx4050 by default, no suffix
MODEL ?= smol135
BATCH ?= 1
SEQ ?= 2048
STEP_TIMES ?=
ADAM8 ?=
DW_LT ?=
TARGET ?= rtx4050
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCH_rtx4050 := 89
ARCH_l4 := 89
ARCH_a100 := 80
ARCH_g4 := 120
ARCH := $(ARCH_$(TARGET))
$(if $(ARCH),,$(error unknown TARGET=$(TARGET), see src/targets/))
OUT := build/train_$(MODEL)_b$(BATCH)_t$(SEQ)$(if $(ADAM8),_adam8)$(if $(DW_LT),_dwlt)$(if $(filter-out rtx4050,$(TARGET)),_$(TARGET))$(if $(STEP_TIMES),_timed)
# chisel's SASS dW cores for this model and token count, when built (src/chisel/, from chisel's gen/lathe_export.py); sm_89 only
CHISEL_DW_H := $(if $(filter 89,$(ARCH)),chisel/dw_$(MODEL)_n$(shell echo $$(($(BATCH) * $(SEQ)))).h)
NVFLAGS := -O3 -std=c++17 -arch=sm_$(ARCH) -lineinfo -Xcompiler -Wall,-Wextra -Xptxas -warn-spills
LDFLAGS := -lcublasLt -lcuda -Xlinker -rpath,$(CUDA)/lib64

$(OUT): src/train.cu src/kernels.cuh src/models/$(MODEL).h src/targets/$(TARGET).h $(if $(CHISEL_DW_H),$(wildcard src/$(CHISEL_DW_H)))
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(if $(STEP_TIMES),-DSTEP_TIMES) $(if $(ADAM8),-DADAM8=1) $(if $(DW_LT),-DDW_LT=1) -DMODEL='"$(MODEL)"' -DMODEL_$(MODEL) -DMODEL_H='"models/$(MODEL).h"' \
		-DTARGET_H='"targets/$(TARGET).h"' $(if $(CHISEL_DW_H),-DCHISEL_DW_H='"$(CHISEL_DW_H)"') -DBATCH=$(BATCH) -DSEQ=$(SEQ) -Isrc $< -o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: clean
