# encodec.cpp

![encodec.cpp](./assets/banner.png)

[![Actions Status](https://github.com/PABannier/encodec.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/PABannier/encodec.cpp/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

High-performance inference of [Meta's Encodec](https://github.com/facebookresearch/encodec) deep learning based audio codec model:

- Plain C/C++ implementation without dependencies using [ggml](https://github.com/ggerganov/ggml)

## Demo

Here is a demo of running Encodec on a single M1 MacBook Pro:

https://github.com/PABannier/encodec.cpp/assets/12958149/d11561be-98e9-4504-bba7-86bcc233a499

## Roadmap

- [x] Support of 24Khz model
- [x] Mixed F16 / F32 precision
- [ ] 4-bit and 8-bit quantization
- [ ] Metal support
- [ ] CoreML support

## Implementation details

- The core tensor operations are implemented in C ([ggml.h](ggml.h) / [ggml.c](ggml.c))
- The encoder-decoder architecture and the high-level C-style API are implemented in C++ ([encodec.h](encodec.h) / [encodec.cpp](encodec.cpp))
- Basic usage is demonstrated in [main.cpp](examples/main).

## Usage

Here are the steps for the encodec model.

### Get the code

```bash
git clone --recurse-submodules https://github.com/PABannier/encodec.cpp.git
cd encodec.cpp
```

### Build

In order to build encodec.cpp you must use `CMake`:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Using Metal

Offloading to GPU is possible with the Metal backend for MacOS. Performance are not improved but
the power consumption and CPU activity is reduced.

```bash
cmake -DGGML_METAL=ON -DBUILD_SHARED_LIBS=Off ..
cmake --build . --config Release
```

### Using cuBLAS

The inference can be offloaded on a CUDA backend with cuBLAS.

```bash
cmake -DGGML_CUBLAS=ON -DBUILD_SHARED_LIBS=Off ..
cmake --build . --config Release
```
