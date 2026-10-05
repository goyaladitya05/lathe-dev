# make MODEL=smol135 BATCH=1 SEQ=2048  ->  build/train_smol135_b1_t2048 (model and shapes are compile-time)
MODEL ?= smol135
BATCH ?= 1
SEQ ?= 2048
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
STEP_TIMES ?=
CHISEL_DOWN ?=
CHISEL_DKV ?=
PROBE_DW ?=
CHISEL_DW ?=
CHISEL_DWS ?=
DWS_BLOCKS ?=
CHISEL_DSQ ?=
DWS_CS ?=
CHISEL_DXPF ?=
OUT := build/train_$(MODEL)_b$(BATCH)_t$(SEQ)$(if $(CHISEL_DOWN),_cdown)$(if $(CHISEL_DKV),_cdkv)$(if $(PROBE_DW),_pdw)$(if $(CHISEL_DW),_cdw)$(if $(CHISEL_DWS),_cdws$(CHISEL_DWS))$(if $(DWS_BLOCKS),_g$(DWS_BLOCKS))$(if $(DWS_CS),_cs)$(if $(CHISEL_DSQ),_cdsq)$(if $(CHISEL_DXPF),_cdxpf)$(if $(STEP_TIMES),_timed)
NVFLAGS := -O3 -std=c++17 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra -Xptxas -warn-spills
LDFLAGS := -lcublasLt -Xlinker -rpath,$(CUDA)/lib64

$(OUT): src/train.cu src/kernels.cuh src/models/$(MODEL).h src/targets/sm89.h $(if $(CHISEL_DOWN),src/chisel_gemm.h) $(if $(CHISEL_DKV),src/chisel_att.h) $(if $(PROBE_DW)$(CHISEL_DWS),src/chisel_dw.h) $(if $(CHISEL_DW),src/chisel_dwadam.h)
	@mkdir -p build
	$(NVCC) $(NVFLAGS) $(if $(STEP_TIMES),-DSTEP_TIMES) $(if $(CHISEL_DOWN),-DCHISEL_DOWN) $(if $(CHISEL_DKV),-DCHISEL_DKV) $(if $(PROBE_DW),-DPROBE_DW) $(if $(CHISEL_DW),-DCHISEL_DW) $(if $(CHISEL_DWS),-DCHISEL_DWS=$(CHISEL_DWS)) $(if $(DWS_BLOCKS),-DDWS_BLOCKS=$(DWS_BLOCKS)) $(if $(CHISEL_DSQ),-DCHISEL_DSQ) $(if $(DWS_CS),-DDWS_CS) $(if $(CHISEL_DXPF),-DCHISEL_DXPF) -DMODEL='"$(MODEL)"' -DMODEL_H='"models/$(MODEL).h"' -DBATCH=$(BATCH) -DSEQ=$(SEQ) -Isrc $< -o $@ $(LDFLAGS) $(if $(CHISEL_DOWN)$(CHISEL_DKV)$(PROBE_DW)$(CHISEL_DW)$(CHISEL_DWS),-lcuda)

clean:
	rm -rf build

.PHONY: clean
