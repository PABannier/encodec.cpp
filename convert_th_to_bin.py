"""Convert Encodec checkpoint into the GGML format.

The bytes are packed in a binary file in the following order:
    - Magic (`ggml` in binary format)
    - Tensors

For each tensor, the bytes are packed as follows:
    - Number of dimensions    (int)
    - Name length             (int)
    - Dimensions              (int[n_dims])
    - Name                    (char[name_length])
    - Data                    (float[n_dims])
"""
import argparse
from pathlib import Path
import struct

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)


def parse_model(checkpoint, outfile):
    for name in checkpoint.keys():
        var_data = checkpoint[name].numpy()
        if not "weight_v" in name:
            # if conv kernel, do not squeeze because 3d tensor
            var_data = var_data.squeeze()

        print(f"Processing variable: {name} with shape: {var_data.shape}")

        # if "conv" in name and ("bias" in name or "weight_g" in name):
        #     if len(var_data.shape) == 0:
        #         # only for decoder.model.15.conv.conv.bias
        #         var_data = var_data.reshape(1, 1)
        #     else:
        #         var_data = var_data.reshape(var_data.shape[0], 1)
        #     print(f"  Reshaped variable: {name} to shape:", var_data.shape)

        if var_data.dtype != np.float32:
            print("  Converting to float32")
            var_data = var_data.astype(np.float32)

        n_dims = len(var_data.shape)
        encoded_name = name.encode("utf-8")
        ftype = 0  # float32
        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype))

        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        outfile.write(encoded_name)

        var_data.tofile(outfile)


if __name__ == "__main__":
    args = parser.parse_args()

    dir_model = Path(args.dir_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    outfile = open(out_dir / "ggml-model.bin", "wb")
    outfile.write(struct.pack("i", 0x67676d6c))  # ggml magic

    checkpoint = torch.load(dir_model / "encodec_24khz-d7cc33bc.th", map_location="cpu")
    parse_model(checkpoint, outfile)

    outfile.close()

    print("Done.")
