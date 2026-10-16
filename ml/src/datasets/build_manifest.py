"""Builds a unified train/val/test + held-out manifest across all data sources,
stratified by generator_family, per configs/data.yaml.

Downloads/extracts each sampled image once and saves it under ml/data/images/, so
training doesn't re-stream from HF Hub every epoch. Writes ml/data/manifest.csv with
columns: path, label, generator_family, source, split.

genimage_pp is excluded from the train/val/test stratified split entirely - its role
in configs/data.yaml is held_out_eval, so every row from it is tagged split=held_out_eval.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from src.datasets.frame_extraction import iter_faceforensics_frames
from src.datasets.hf_sources import (
    Sample,
    iter_community_forensics,
    iter_genimage,
    iter_genimage_pp,
    iter_synthetic_faces_hq,
)

ML_DIR = Path(__file__).resolve().parents[2]  # ml/
DATA_DIR = ML_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

FIELDNAMES = ["path", "label", "generator_family", "source", "split"]


def _save_and_record(samples: Iterable[Sample], dest_dir: Path, data_dir: Path) -> list[dict]:
    """Saves each sample to disk as it's produced, instead of materializing the
    whole batch of decoded images in memory first - with per_source_cap in the
    hundreds, holding every decoded PIL image at once is what blows up RAM on
    memory-constrained runtimes (e.g. free-tier Colab)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(samples):
        filename = f"{sample.source}_{i:06d}.jpg"
        path = dest_dir / filename
        sample.image.save(path, format="JPEG", quality=90)
        rows.append(
            {
                "path": str(path.relative_to(data_dir)),
                "label": sample.label,
                "generator_family": sample.generator_family,
                "source": sample.source,
            }
        )
    return rows


def _stratified_split(rows: list[dict], train_frac: float, val_frac: float, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["generator_family"]].append(row)

    train, val, test = [], [], []
    for family_rows in by_family.values():
        shuffled = family_rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    return train, val, test


def build_manifest(
    per_source_cap: int = 50,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    data_dir: Path | None = None,
) -> list[dict]:
    """Builds and writes the manifest; also returns the rows written (for tests).

    `data_dir` defaults to ml/data (this repo checkout) but can point anywhere -
    e.g. a Drive-mounted path in Colab, so images/manifest persist across sessions
    without needing the code itself to live there too.
    """
    data_dir = data_dir or DATA_DIR
    images_dir = data_dir / "images"
    manifest_path = data_dir / "manifest.csv"

    pooled_rows = []

    for iterator in (iter_community_forensics(), iter_genimage(), iter_synthetic_faces_hq()):
        pooled_rows.extend(_save_and_record(islice(iterator, per_source_cap), images_dir, data_dir))

    ff_iterator = iter_faceforensics_frames(
        max_videos_per_class=max(1, per_source_cap // 10), frames_per_video=5, seed=seed
    )
    pooled_rows.extend(_save_and_record(islice(ff_iterator, per_source_cap), images_dir, data_dir))

    train, val, test = _stratified_split(pooled_rows, train_frac, val_frac, seed)
    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "val"
    for row in test:
        row["split"] = "test"

    held_out_iterator = iter_genimage_pp(max_per_config=max(1, per_source_cap // 2))
    held_out_rows = _save_and_record(islice(held_out_iterator, per_source_cap), images_dir, data_dir)
    for row in held_out_rows:
        row["split"] = "held_out_eval"

    final_rows = train + val + test + held_out_rows

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(final_rows)

    return final_rows
