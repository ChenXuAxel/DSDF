"""CPU regression checks for the GeoTIFF example and checkpoint formats."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from examples.predict import load_weights, read_inputs, write_prediction


class PredictionIOTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.profile = dict(driver="GTiff", height=16, width=24, count=2,
                            dtype="float32", crs="EPSG:32650",
                            transform=from_origin(400000, 4200000, 30, 30))

    def raster(self, name, value, **overrides):
        profile = {**self.profile, **overrides}
        path = self.root / name
        shape = (profile["count"], profile["height"], profile["width"])
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.full(shape, value, dtype=profile["dtype"]))
        return path

    def test_band_order_input_order_and_output_grid(self):
        paths = [self.raster(f"{i}.tif", i / 4) for i in range(3)]
        with rasterio.open(paths[0], "r+") as dst:
            dst.write(np.full((16, 24), 0.75, dtype="float32"), 2)
        inputs, profile = read_inputs(paths)
        self.assertEqual([x[0, 0, 0, 0].item() for x in inputs], [0, 0.25, 0.5])
        self.assertEqual(inputs[0][0, 1, 0, 0].item(), 0.75)
        self.assertEqual(tuple(inputs[0].shape), (1, 2, 16, 24))
        out = self.root / "prediction.tif"
        write_prediction(out, inputs[0][0].numpy(), profile)
        with rasterio.open(out) as src:
            self.assertEqual(src.transform, self.profile["transform"])
            self.assertEqual(src.crs.to_epsg(), 32650)
            np.testing.assert_array_equal(src.read(), inputs[0][0].numpy())
        with self.assertRaises(FileExistsError):
            write_prediction(out, inputs[1][0].numpy(), profile)

    def test_mismatched_grid_and_unprepared_pixels(self):
        fine = self.raster("fine.tif", 0.1)
        for kwargs, value, error in [
            ({"transform": from_origin(400030, 4200000, 30, 30)}, 0.1, "spatial grid"),
            ({"dtype": "uint16"}, 1200, "digital numbers"),
            ({"nodata": -9999}, -9999, "masked pixels"),
            ({}, np.nan, "finite patch"),
        ]:
            with self.subTest(error=error):
                coarse = self.raster("coarse.tif", value, **kwargs)
                with self.assertRaisesRegex(ValueError, error):
                    read_inputs([fine, coarse, fine])

    def test_checkpoint_formats_and_strict_loading(self):
        source = torch.nn.Linear(3, 4)
        target = torch.nn.Linear(3, 4)
        plain = source.state_dict()
        wrapped = {"state_dict": {f"model.{k}": v for k, v in plain.items()}}
        path = self.root / "model.pt"
        for state in (plain, wrapped):
            torch.save(state, path)
            load_weights(target, path)
            for k in plain:
                torch.testing.assert_close(target.state_dict()[k], plain[k], rtol=0, atol=0)
        torch.save({"weight": plain["weight"]}, path)
        with self.assertRaises(RuntimeError):
            load_weights(target, path)


if __name__ == "__main__":
    unittest.main()
