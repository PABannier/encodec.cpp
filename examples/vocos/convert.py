"""Convert Vocos model checkpoint into the GGML format.

The bytes are packed in a binary file in the following order:
    - Magic (`ggml` in binary format)
    - Tensors

For each tensor, the bytes are packed as follows:
    - Number of dimensions   (int)
    - Name length            (int)
    - Dimensions             (int[n_dims])
    - Name                   (char[name_length])
    - Data                   (float[n_dims])

Usage
-----

```bash
    python convert.py \
        --dir-model <PATH_TO_CHECKPOINT> \
        --out-dir ./ \
        --use-f16
```
"""
import argparse
from pathlib import Path
import struct
import re
import yaml

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--use-f16", action="store_true")


def parse_vocos_weights(checkpoint, fout, use_f16):
    """Dump Vocos model checkpoint."""
    def _clean_name(name):
        module_name = name.split(".")[0]
        if module_name not in ["feature_extractor", "backbone", "head"]:
            raise Exception("Unknown module name")

        # Backbone
        if re.match(r"backbone\.convnext\.\d+\.dwconv\.(weight|bias)", name):
            i = re.findall(r"\d+", name)[0]
            ttype = re.findall(r"(weight|bias)", name)[0][0]
            return f"{module_name}/convnext/{i}/dwconv/{ttype}"
        elif re.match(r"backbone\.convnext\.\d+\.gamma", name):
            i = re.findall(r"\d+", name)[0]
            return f"{module_name}/convnext/{i}/gamma"
        elif re.match(r"backbone\.convnext\.\d+\.norm\.(scale|shift)\.weight", name):
            i = re.findall(r"\d+", name)[0]
            scale_or_shift = re.findall(r"(scale|shift)", name)[0]
            return f"{module_name}/convnext/{i}/norm/{scale_or_shift}"
        elif re.match(r"backbone\.convnext\.\d+\.pwconv(\d+)\.(weight|bias)", name):
            matches = re.findall(r"\d+", name)
            i, j = matches[0], matches[1]
            ttype = re.findall(r"(weight|bias)", name)[0][0]
            return f"{module_name}/convnext/{i}/pwconv/{j}/{ttype}"
        elif re.match(r"backbone\.(embed|final_layer_norm)\.(weight|bias)", name):
            ltype = re.findall(r"(embed|final_layer_norm)", name)[0]
            ttype = re.findall(r"(weight|bias)", name)[0][0]
            return f"{module_name}/{ltype}/{ttype}"
        elif re.match(r"backbone\.norm\.(scale|shift)\.weight", name):
            ltype = re.findall(r"(scale|shift)", name)[0]
            return f"{module_name}/norm/{ltype}/w"
        # Feature extractor
        elif name == "feature_extractor.codebook_weights":
            return f"{module_name}/codebook_weights"
        # Head
        elif name == "head.istft.window":
            return f"{module_name}/istft/window"
        elif re.match(r"head\.out\.(weight|bias)", name):
            ttype = re.findall(r"(weight|bias)", name)[0][0]
            return f"{module_name}/out/{ttype}"
        # Unknown
        else:
            raise Exception(f"Unknown variable name: {name}")

    n_f16, n_f32 = 0, 0

    for name in checkpoint.keys():
        var_data = checkpoint[name].cpu().numpy()
        clean_name = _clean_name(name)

        print(f"{name : <40} -> {clean_name}")
        print(f"    {var_data.shape}")

        if use_f16:
            if clean_name.endswith("/w"):
                # Only weight matrices are cast to float16
                var_data = var_data.astype(np.float16)
                ftype_cur = 1
                n_f16 += 1
            else:
                var_data = var_data.astype(np.float32)
                ftype_cur = 0
                n_f32 += 1
        else:
            var_data = var_data.astype(np.float32)
            ftype_cur = 0
            n_f32 += 1

        n_dims = len(var_data.shape)
        encoded_name = clean_name.encode("utf-8")
        fout.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))

        for i in range(n_dims):
            fout.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))
        fout.write(encoded_name)

        var_data.tofile(fout)

    print("\n")
    print(f"n_f16: {n_f16} ({n_f16 / (n_f16 + n_f32) * 100:.0f}%)")
    print(f"n_f32: {n_f32} ({n_f32 / (n_f16 + n_f32) * 100:.0f}%)")


def parse_hparams(fout, config, use_f16):
    # Backbone
    bb_config = config["backbone"]["init_args"]
    fout.write(struct.pack("i", bb_config["input_channels"]))
    fout.write(struct.pack("i", bb_config["dim"]))
    fout.write(struct.pack("i", bb_config["intermediate_dim"]))
    fout.write(struct.pack("i", bb_config["num_layers"]))
    fout.write(struct.pack("i", bb_config["adanorm_num_embeddings"]))
    # Head (padding is assumed to be `same`)
    head_config = config["head"]["init_args"]
    fout.write(struct.pack("i", head_config["dim"]))
    fout.write(struct.pack("i", head_config["n_fft"]))
    fout.write(struct.pack("i", head_config["hop_length"]))
    # General
    fout.write(struct.pack("i", int(use_f16)))


if __name__ == "__main__":
    args = parser.parse_args()

    dir_model = Path(args.dir_model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    outfile = Path(out_dir / "ggml_weights.bin")
    fout = open(outfile, "wb")
    fout.write(struct.pack("i", 0x67676d6c))

    with open(dir_model / "config.yaml", "rb") as f:
        config = yaml.safe_load(f)
        parse_hparams(fout, config, args.use_f16)

    checkpoint = torch.load(dir_model / "pytorch_model.bin", map_location="cpu")
    parse_vocos_weights(checkpoint, fout, args.use_f16)

    fout.close()
    print("Done.")
