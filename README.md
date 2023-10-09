# encodec.cpp

High-performance inference of [Meta's Encodec](https://github.com/facebookresearch/encodec) deep learning based audio codec model:

- Plain C/C++ implementation without dependencies using [ggml](https://github.com/ggerganov/ggml)

The entire implementation of the model is contained in 3 source files:

Tensor operations: ggml.h / ggml.c<br/>
Inference: encodec.h / encodec.cpp

## Roadmap

- [x] Support of 24Khz model
- [ ] Support of 48Khz model
- [ ] Encodec's language model support
- [x] Mixed F16 / F32 precision


## Implementation details

- The core tensor operations are implemented in C ([ggml.h](ggml.h) / [ggml.c](ggml.c))
- The encoder-decoder architecture and the high-level C-style API are implemented in C++ ([encodec.h](encodec.h) / [encodec.cpp](encodec.cpp))
- Sample usage is demonstrated in [main.cpp](examples/main)

## Usage

Here are the steps for the bark model.

### Get the code

```bash
git clone https://github.com/PABannier/encodec.cpp.git
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
