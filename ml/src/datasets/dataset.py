"""torch Dataset wrapping ml/data/manifest.csv.

Reads a manifest row, opens the image it points to, and applies the spatial
transform (RGB, ImageNet-normalized) and frequency transform (FFT magnitude)
from transforms.py - the two tensors DualBranchDetector.forward() expects.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.datasets.build_manifest import DATA_DIR, MANIFEST_PATH
from src.datasets.transforms import build_spatial_transform, fft_magnitude_log1p


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        split: str,
        manifest_path: Path = MANIFEST_PATH,
        data_dir: Path = DATA_DIR,
    ) -> None:
        self.data_dir = data_dir
        self.spatial_transform = build_spatial_transform(train=(split == "train"))

        with open(manifest_path) as f:
            all_rows = list(csv.DictReader(f))

        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(self.data_dir / row["path"]).convert("RGB")

        rgb = self.spatial_transform(image)
        freq = fft_magnitude_log1p(image)
        label = int(row["label"])

        return rgb, freq, label, row["generator_family"]
