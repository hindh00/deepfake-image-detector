"""Builds a unified train/val/test + held-out manifest across all data sources,
balanced real/fake and stratified by generator_family, per configs/data.yaml.

Downloads/extracts each sampled image once and saves it under ml/data/images/, so
training doesn't re-stream from HF Hub every epoch. Writes ml/data/manifest.csv with
columns: path, label, generator_family, source, split.

genimage_pp is excluded from the train/val/test stratified split entirely - its role
in configs/data.yaml is held_out_eval, so every row from it is tagged split=held_out_eval.

Real/fake balance (configs/data.yaml's `balance: real_fake_50_50`) isn't a byproduct of
source ordering - it's enforced in two places: mixed-label sources are capped per label
(see `_take_balanced_by_label`), not by taking the first N records regardless of label,
and the pooled rows are rebalanced one more time before the split (see
`_balance_real_fake`) since synthetic_faces_hq is fake-only and has no real counterpart
of its own. Skipping this lets the model collapse to always predicting fake - real
accuracy tanks while every fake generator_family accuracy climbs to ~1.0, and overall
val_acc still looks fine because fake is the majority class.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

from src.datasets.frame_extraction import iter_faceforensics_frames
from src.datasets.hf_sources import (
    FAKE,
    REAL,
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


def _take_balanced_by_label(iterator: Iterable[Sample], per_label_cap: int) -> Iterator[Sample]:
    """Pulls up to `per_label_cap` REAL and `per_label_cap` FAKE samples from a
    mixed-label stream, filtering by label as it goes instead of taking the first N
    records regardless of label - otherwise real representation is just whatever the
    underlying shard ordering happens to give you. Stays lazy so it composes with
    `_save_and_record`'s save-as-you-go behavior."""
    counts = {REAL: 0, FAKE: 0}
    for sample in iterator:
        if counts[sample.label] >= per_label_cap:
            if all(c >= per_label_cap for c in counts.values()):
                break
            continue
        counts[sample.label] += 1
        yield sample


def _balance_real_fake(rows: list[dict], seed: int) -> list[dict]:
    """Downsamples the majority label to match the minority label's count, stratified
    within each generator_family so no family gets wiped out entirely. Runs after
    pooling since synthetic_faces_hq is fake-only and has no real counterpart to cap
    against - per-source label capping alone can't fully balance the pool."""
    real_rows = [r for r in rows if r["label"] == REAL]
    fake_rows = [r for r in rows if r["label"] == FAKE]

    if len(real_rows) == len(fake_rows):
        return rows

    majority, minority = (fake_rows, real_rows) if len(fake_rows) > len(real_rows) else (real_rows, fake_rows)
    keep_fraction = len(minority) / len(majority)

    rng = random.Random(seed)
    by_family = defaultdict(list)
    for row in majority:
        by_family[row["generator_family"]].append(row)

    trimmed_majority = []
    for family_rows in by_family.values():
        shuffled = family_rows[:]
        rng.shuffle(shuffled)
        trimmed_majority.extend(shuffled[: round(len(shuffled) * keep_fraction)])

    return trimmed_majority + minority


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

    # community_forensics and genimage are mixed-label - cap per label so real isn't
    # just whatever fraction the stream's native ordering happens to hand us.
    for iterator in (iter_community_forensics(), iter_genimage()):
        pooled_rows.extend(
            _save_and_record(_take_balanced_by_label(iterator, per_source_cap), images_dir, data_dir)
        )

    # synthetic_faces_hq is fake-only by construction (no label to balance within it);
    # it's accounted for by the pool-wide rebalance below instead.
    pooled_rows.extend(
        _save_and_record(islice(iter_synthetic_faces_hq(), per_source_cap), images_dir, data_dir)
    )

    ff_iterator = iter_faceforensics_frames(
        max_videos_per_class=max(1, per_source_cap // 10), frames_per_video=5, seed=seed
    )
    pooled_rows.extend(_save_and_record(islice(ff_iterator, per_source_cap), images_dir, data_dir))

    pooled_rows = _balance_real_fake(pooled_rows, seed)

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
