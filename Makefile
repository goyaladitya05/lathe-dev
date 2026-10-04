# make MODEL=smol135 BATCH=1 SEQ=2048  ->  build/train_smol135_b1_t2048 (model and shapes are compile-time)
MODEL ?= smol135
BATCH ?= 1
SEQ ?= 2048
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
OUT := build/train_$(MODEL)_b$(BATCH)_t$(SEQ)
NVFLAGS := -O3 -std=c++17 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra -Xptxas -warn-spills
LDFLAGS := -lcublasLt -Xlinker -rpath,$(CUDA)/lib64

$(OUT): src/train.cu src/kernels.cuh src/models/$(MODEL).h src/targets/sm89.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -DMODEL='"$(MODEL)"' -DMODEL_H='"models/$(MODEL).h"' -DBATCH=$(BATCH) -DSEQ=$(SEQ) -Isrc $< -o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: clean
