"""Fuse one prepared multispectral patch with a trained DSDF checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch


def read_inputs(paths):
    """Read three aligned, band-first rasters in model input order."""
    arrays = []
    for path in paths:
        with rasterio.open(path) as src:
            grid = (src.count, src.height, src.width, src.crs, src.transform)
            if not arrays:
                reference_grid = grid
                profile = dict(driver="GTiff", width=src.width, height=src.height,
                               count=src.count, crs=src.crs, transform=src.transform,
                               dtype="float32", compress="deflate")
                if src.height % 8 or src.width % 8:
                    raise ValueError("Patch height and width must be divisible by eight.")
            elif grid != reference_grid:
                raise ValueError(f"{path}: band count or spatial grid differs from the fine reference.")
            if any(np.dtype(dtype).kind != "f" for dtype in src.dtypes):
                raise ValueError(f"{path}: convert digital numbers to floating-point reflectance first.")
            data = src.read(out_dtype="float32", masked=True)
            if np.any(data.mask) or not np.isfinite(data.data).all():
                raise ValueError(f"{path}: prepare a finite patch without masked pixels.")
            arrays.append(torch.from_numpy(data.data).unsqueeze(0))
    return arrays, profile


def load_weights(model, path):
    """Load a plain state dict or the model weights inside a paper checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()
                 if key.startswith("model.")}
    model.load_state_dict(state, strict=True)


def write_prediction(path, pixels, profile):
    with MemoryFile() as memory:
        with memory.open(**profile) as dst:
            dst.write(pixels)
        encoded = memory.read()
    with path.open("xb") as out:
        out.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-reference", type=Path, required=True)
    parser.add_argument("--coarse-reference", type=Path, required=True)
    parser.add_argument("--coarse-target", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New GeoTIFF path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch installation and GPU are required")

    from dsdf import DSDF

    inputs, profile = read_inputs([args.fine_reference, args.coarse_reference, args.coarse_target])
    model = DSDF(in_channels=profile["count"], use_checkpoint=False)
    load_weights(model, args.weights)
    model.cuda().eval()
    with torch.no_grad():
        prediction = model(*(image.cuda() for image in inputs)).squeeze(0)
    if not torch.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite reflectance.")
    write_prediction(args.output, prediction.cpu().numpy(), profile)
    print(f"Wrote {args.output} ({profile['count']} bands, {profile['width']} x {profile['height']})")


if __name__ == "__main__":
    main()
