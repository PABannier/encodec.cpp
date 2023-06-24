# encodec.cpp

High-performance inference of [Meta's Encodec](https://github.com/facebookresearch/encodec) deep learning based audio codec model:

- Plain C/C++ implementation without dependencies using [ggml](https://github.com/ggerganov/ggml)

The entire implementation of the model is contained in 3 source files:

Tensor operations: ggml.h / ggml.c
Inference: encodec.h / encodec.cpp
Utils operations: encoder.h / encoder.cpp

## Roadmap

- [x] Support of 24Khz model
- [ ] Support of 48Khz model
- [ ] Encodec's language model support
- [ ] Mixed F16 / F32 precision
- [ ] 4-bit / 8-bit quantization support
- [ ] Add Encodec's original language model
- [ ] Support of 48khz model


## Implementation details

- The core tensor operations are implemented in C ([ggml.h](ggml.h) / [ggml.c](ggml.c))
- The encoder-decoder architecture and the high-level C-style API are implemented in C++ ([encodec.h](encodec.h) / [encodec.cpp](encodec.cpp))
- Sample usage is demonstrated in [main.cpp](examples/main)

## Quick start
