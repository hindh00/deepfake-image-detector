"""Frame extraction for FaceForensics++ (bitmind/FaceForensicsC23).

This reads the HF-hosted zip's central directory via
HTTP range requests (confirmed the host advertises Accept-Ranges: bytes) and extracts
only the specific video entries actually sampled.
"""

from __future__ import annotations

import random
import tempfile
import zipfile
from collections.abc import Iterator

import cv2
import requests
from huggingface_hub import hf_hub_url
from PIL import Image

from src.datasets.hf_sources import FAKE, REAL, Sample

REPO_ID = "bitmind/FaceForensicsC23"
ZIP_PATH = "FaceForensics++_C23.zip"


class HTTPRangeFile:
    """Minimal seekable file-like object backed by HTTP range requests, so zipfile
    can read the central directory and individual entries without downloading
    the whole archive."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True)
        self.size = int(head.headers["content-length"])
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        end = self.size - 1 if n is None or n < 0 else min(self.pos + n, self.size) - 1
        if self.pos > end:
            return b""
        resp = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = resp.content
        self.pos += len(data)
        return data


def _open_zip() -> zipfile.ZipFile:
    url = hf_hub_url(REPO_ID, ZIP_PATH, repo_type="dataset")
    return zipfile.ZipFile(HTTPRangeFile(url))


def _sample_frames(video_bytes: bytes, frames_per_video: int) -> list[Image.Image]:
    """Evenly spaced frames across the video's full length, not just the opening seconds."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total > 0:
            indices = [int(i * total / frames_per_video) for i in range(frames_per_video)]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames


def iter_faceforensics_frames(
    max_videos_per_class: int = 50,
    frames_per_video: int = 5,
    seed: int = 42,
) -> Iterator[Sample]:
    zf = _open_zip()
    names = zf.namelist()

    real_videos = [n for n in names if n.startswith("FaceForensics++_C23/real/") and n.endswith(".mp4")]
    fake_videos = [n for n in names if n.startswith("FaceForensics++_C23/fake/") and n.endswith(".mp4")]

    rng = random.Random(seed)
    selected_real = rng.sample(real_videos, min(max_videos_per_class, len(real_videos)))
    selected_fake = rng.sample(fake_videos, min(max_videos_per_class, len(fake_videos)))

    plan = [(n, REAL, "real") for n in selected_real] + [(n, FAKE, "gan_face_swap") for n in selected_fake]

    for name, label, generator_family in plan:
        video_bytes = zf.read(name)
        for frame in _sample_frames(video_bytes, frames_per_video):
            yield Sample(image=frame, label=label, generator_family=generator_family, source="faceforensics")
