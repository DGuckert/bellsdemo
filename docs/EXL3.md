# EXL3 support

This fork accepts EXL3-packed GGUF tensors (ExLlamaV3 trellis, `exl3.pack=mma`)
on the CUDA backend. Normal Q/IQ GGUF models are unchanged.

## Hardware

The upstream EXL3 CUDA kernels target `sm_86` and `sm_89` (RTX 30-series and
RTX 40-series). Build with:

```sh
cmake -S . -B build-cuda-exl3 -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"
cmake --build build-cuda-exl3 --target llama-server
```

`sm_75` (RTX 20-series) and `sm_120` (RTX 50-series) are not covered by the
current EXL3 cubins. Those GPUs can still run ordinary Q/IQ GGUFs.

## Convert

Start from a packed EXL3 safetensors directory produced by ExLlamaV3's
`convert.py` (the Viterbi packing itself is not part of this repo). Then:

```sh
PYTHONPATH=./gguf-py python3 convert_hf_to_gguf.py <exl3-dir> --outtype auto
```

The converter writes `exl3.pack`, `exl3.codebook`, the packed weight, and the
required `suh`/`svh` F32 companions. `EXL3_PACK=xe2-16` is SYCL-only and is
rejected by the CUDA runtime.

## BELLS

EXL3 is a normal GGML tensor type to BELLS. The expert cache sees only tensor
type, block size, byte size, and `ggml_nbytes`; it does not carry per-quant
weights, learned constants, or quant-specific slot heuristics. Adding EXL3
therefore does not special-case BELLS and does not narrow its model or
quantization coverage.

## Verification

Executed here:

```sh
# CPU/static regression build
cmake -S . -B build-cpu -G "Visual Studio 17 2022" -A x64 \
  -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cpu --config Release --target test-exl3-types test-bells-cache-regression
ctest --test-dir build-cpu -C Release -R "test-exl3-types|test-bells-cache-regression"
# 100% tests passed, 0 tests failed out of 2
```

```sh
# CUDA/EXL3 build
cmake -S . -B build-cuda-exl3 -G "Visual Studio 17 2022" -A x64 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_TESTS=ON
cmake --build build-cuda-exl3 --config Release --target llama-server test-exl3-types
ctest --test-dir build-cuda-exl3 -C Release -R test-exl3-types
# 100% tests passed, 0 tests failed out of 1
```

A synthetic 1-layer EXL3 GGUF was loaded to exercise companion lookup and graph
construction:

```text
load_tensors: loaded suh/svh companions for 1 EXL3 weights
llama_graph_n_input_tensors: exl3_h128 [f32, ne = {128, 128, 1, 1}]
```

The same model stops during context reservation on the local RTX 2060 because
the compiled EXL3 kernels require `sm_86`/`sm_89`. This is a hardware limit, not
a loader failure. End-to-end EXL3 inference on supported hardware remains to be
run.
