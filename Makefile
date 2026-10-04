# make BATCH=1 SEQ=2048  ->  build/train_b1_t2048 (shapes are compile-time)
BATCH ?= 1
SEQ ?= 2048
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
OUT := build/train_b$(BATCH)_t$(SEQ)
NVFLAGS := -O3 -std=c++17 -arch=sm_89 -lineinfo -Xcompiler -Wall,-Wextra -Xptxas -warn-spills
LDFLAGS := -lcublasLt -Xlinker -rpath,$(CUDA)/lib64

$(OUT): src/train.cu src/kernels.cuh src/models/smol135.h src/targets/sm89.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -DBATCH=$(BATCH) -DSEQ=$(SEQ) -Isrc $< -o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: clean
