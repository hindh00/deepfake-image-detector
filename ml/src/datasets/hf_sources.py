"""Loader wrappers for each dataset in configs/data.yaml.

Each source is exposed as a lazy generator of Sample, normalized to one contract
regardless of how differently each dataset is actually packaged on HF Hub:
CommunityForensics-Small and Tiny-GenImage are proper `datasets`-library parquet
(simple streaming); SyntheticFacesHQ streams via `datasets`' imagefolder auto-detection
over its zip files; GenImage++ is NOT `datasets`-loadable at all (its tar.zst archives
trip up `datasets`' webdataset auto-detection), so it's read by partially streaming the
compressed tar directly and stopping once `max_per_config` images are pulled - avoiding
a multi-GB download for what's only meant to be a small held-out eval slice.

AlexWortega/celeb was dropped: despite being labeled as a Celeb-DF mirror, it's actually
plain CelebA (real photos only, no fakes) plus unrelated model checkpoints.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
import zstandard
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url
from PIL import Image

load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # picks up HF_TOKEN if present

REAL = 0
FAKE = 1

GENIMAGE_GENERATORS = {
    0: "real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}

# Only the two single-file (non multi-part) base subsets - matches the "Flux.1/SD3"
# generalization target without the multi-part concatenation that flux_multistyle,
# sdxl_style, etc. would need.
GENIMAGE_PP_CONFIGS = {
    "flux": "Flux.1",
    "stable_diffusion_v_3_0": "SD3",
}


@dataclass
class Sample:
    image: Image.Image
    label: int  # 0 = real, 1 = fake
    generator_family: str
    source: str


def _label_capped(
    rows: Iterator[dict], per_label_cap: int | None, max_scan: int | None
) -> Iterator[dict]:
    """Filters a raw (pre-decode) example stream down to at most `per_label_cap` REAL
    and `per_label_cap` FAKE rows, checking `row["label"]` before any image decoding
    happens downstream - so a skipped row never pays for a decode it won't use.

    `max_scan` bounds how many raw rows this will look at before giving up, even if a
    label's quota isn't filled yet. Needed because CommunityForensics-Small (confirmed
    via the datasets-server API) is NOT shuffled - it's label-sorted in blocks, fake
    for roughly its first 8500 rows and real only in the last ~1500 - so an unbounded
    per-label scan would stream nearly the whole 5.7GB dataset just to collect a few
    hundred real examples, which is what was spiking RAM. Without a cap this source
    simply won't surface real examples within budget; that's fine, other sources
    (genimage interleaves real/fake every row) and build_manifest's pool-wide rebalance
    cover the shortfall.
    """
    counts = {REAL: 0, FAKE: 0}
    for i, row in enumerate(rows):
        if max_scan is not None and i >= max_scan:
            return
        label = row["label"]
        if per_label_cap is not None:
            if counts[label] >= per_label_cap:
                if all(c >= per_label_cap for c in counts.values()):
                    return
                continue
            counts[label] += 1
        yield row


def iter_community_forensics(
    per_label_cap: int | None = None, max_scan: int | None = None
) -> Iterator[Sample]:
    ds = load_dataset("OwensLab/CommunityForensics-Small", split="train", streaming=True)
    for ex in _label_capped(ds, per_label_cap, max_scan):
        image = Image.open(io.BytesIO(ex["image_data"])).convert("RGB")
        generator_family = "real" if ex["label"] == REAL else ex["architecture"]
        yield Sample(image=image, label=ex["label"], generator_family=generator_family, source="community_forensics")


def iter_genimage(
    split: str = "train", per_label_cap: int | None = None, max_scan: int | None = None
) -> Iterator[Sample]:
    ds = load_dataset("TheKernel01/Tiny-GenImage", split=split, streaming=True)
    for ex in _label_capped(ds, per_label_cap, max_scan):
        image = ex["image"].convert("RGB")
        generator_family = GENIMAGE_GENERATORS[ex["generator"]]
        yield Sample(image=image, label=ex["label"], generator_family=generator_family, source="genimage")


def iter_genimage_pp(
    configs: dict[str, str] = GENIMAGE_PP_CONFIGS,
    max_per_config: int = 1000,
) -> Iterator[Sample]:
    """Streams the first `max_per_config` images out of each config's compressed tar
    without downloading the full multi-GB archive. Fake-only (this benchmark has no
    real counterpart images); pair with reals from another source during manifest building.
    """
    for config_name, generator_family in configs.items():
        url = hf_hub_url("Lunahera/genimagepp", f"archives/{config_name}.tar.zst", repo_type="dataset")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(resp.raw)
            tf = tarfile.open(fileobj=reader, mode="r|")
            try:
                count = 0
                for member in tf:
                    if not member.isfile():
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    image = Image.open(io.BytesIO(extracted.read())).convert("RGB")
                    yield Sample(image=image, label=FAKE, generator_family=generator_family, source="genimage_pp")
                    count += 1
                    if count >= max_per_config:
                        break
            finally:
                tf.close()
        finally:
            resp.close()


def iter_synthetic_faces_hq() -> Iterator[Sample]:
    """All-fake diffusion/GAN face source; pair with reals from another source
    during manifest building (see configs/data.yaml)."""
    ds = load_dataset("bitmind/SyntheticFacesHQ", split="train", streaming=True)
    for ex in ds:
        image = ex["image"].convert("RGB")
        yield Sample(image=image, label=FAKE, generator_family="diffusion_face", source="synthetic_faces_hq")
