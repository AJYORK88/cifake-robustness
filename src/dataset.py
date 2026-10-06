"""CIFAKE loading and the frozen train/val split.

Labels: REAL = 0, FAKE = 1. Images are returned as float tensors in [0, 1], shape (3, 32, 32).

Split files (data/splits/train.txt, data/splits/val.txt) hold one path per line,
RELATIVE to data/raw and using forward slashes, so both partners get byte-identical
files regardless of OS or where the repo is cloned. Lists are sorted before the
seeded shuffle because directory listing order differs across filesystems.

CLI:
    python -m src.dataset --check          # show detected folder layout and counts
    python -m src.dataset --make-split     # write data/splits/{train,val}.txt (refuses to overwrite)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import load_config, repo_path

LABELS = {"REAL": 0, "FAKE": 1}
POOLS = ("train", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
EXPECTED_COUNTS = {("train", "REAL"): 50_000, ("train", "FAKE"): 50_000,
                   ("test", "REAL"): 10_000, ("test", "FAKE"): 10_000}


# ---------------------------------------------------------------- layout detection
def find_class_dirs(raw_dir: Path, max_depth: int = 3) -> dict[tuple[str, str], Path]:
    """Locate {train,test} x {REAL,FAKE} folders under raw_dir, case-insensitively.

    Handles the zip unpacking either directly into raw_dir or into a nested folder
    (e.g. raw_dir/cifake/train/REAL). Returns {(pool, CLASS): absolute_dir}.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{raw_dir} does not exist. Unzip CIFAKE there first (see README).")

    candidates = [raw_dir]
    frontier = [raw_dir]
    for _ in range(max_depth):
        frontier = [c for d in frontier for c in sorted(d.iterdir()) if c.is_dir()]
        candidates.extend(frontier)

    for root in candidates:
        found: dict[tuple[str, str], Path] = {}
        subdirs = {c.name.lower(): c for c in root.iterdir() if c.is_dir()}
        for pool in POOLS:
            if pool not in subdirs:
                break
            classes = {c.name.upper(): c for c in subdirs[pool].iterdir() if c.is_dir()}
            for cls in LABELS:
                if cls in classes:
                    found[(pool, cls)] = classes[cls]
        if len(found) == 4:
            return found
    raise FileNotFoundError(
        f"Could not find train/{{REAL,FAKE}} and test/{{REAL,FAKE}} under {raw_dir} "
        f"(searched {max_depth} levels deep). Check how the zip unpacked."
    )


def label_from_path(rel_path: str) -> int:
    """Label from the parent folder name (REAL/FAKE, case-insensitive)."""
    return LABELS[rel_path.split("/")[-2].upper()]


def pool_from_path(rel_path: str) -> str:
    """'train' or 'test', from the grandparent folder name."""
    return rel_path.split("/")[-3].lower()


def list_images(raw_dir: Path, pool: str) -> list[str]:
    """Sorted relative POSIX paths of every image in one pool ('train' or 'test')."""
    raw_dir = Path(raw_dir)
    dirs = find_class_dirs(raw_dir)
    paths = []
    for cls in LABELS:
        for p in dirs[(pool, cls)].iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p.relative_to(raw_dir).as_posix())
    return sorted(paths)


# ---------------------------------------------------------------- split
def make_split(raw_dir: Path, val_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """Stratified holdout from the official TRAIN pool only. Returns (train, val), both sorted."""
    train_pool = list_images(raw_dir, "train")
    rng = random.Random(seed)
    train, val = [], []
    for cls, label in LABELS.items():
        cls_paths = sorted(p for p in train_pool if label_from_path(p) == label)
        if len(cls_paths) <= val_per_class:
            raise ValueError(f"Only {len(cls_paths)} {cls} images; cannot hold out {val_per_class}.")
        rng.shuffle(cls_paths)
        val.extend(cls_paths[:val_per_class])
        train.extend(cls_paths[val_per_class:])
    return sorted(train), sorted(val)


def write_split(paths: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(paths) + "\n")


def read_split(path: Path) -> list[str]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.dataset --make-split")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def split_paths(cfg: dict, split: str) -> list[str]:
    """Relative paths for 'train', 'val', or 'test' (official test = whole test pool)."""
    if split == "test":
        return list_images(repo_path(cfg["data"]["raw_dir"]), "test")
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split {split!r}")
    return read_split(repo_path(cfg["data"]["splits_dir"]) / f"{split}.txt")


# ---------------------------------------------------------------- dataset / loader
class CifakeDataset(Dataset):
    """Loads images by relative path. Optional `corrupt` is applied to the PIL image
    BEFORE ToTensor (test-time only; never pass it for the training split)."""

    def __init__(self, raw_dir: Path, rel_paths: list[str], image_size: int = 32,
                 corrupt: Callable[[Image.Image], Image.Image] | None = None):
        self.raw_dir = Path(raw_dir)
        self.rel_paths = rel_paths
        self.labels = [label_from_path(p) for p in rel_paths]
        self.image_size = image_size
        self.corrupt = corrupt
        self.to_tensor = transforms.ToTensor()  # uint8 HWC -> float CHW in [0, 1]

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.raw_dir / self.rel_paths[idx]) as im:
            img = im.convert("RGB")
        if self.corrupt is not None:
            img = self.corrupt(img)
        if img.size != (self.image_size, self.image_size):
            raise ValueError(f"{self.rel_paths[idx]}: got {img.size}, expected {self.image_size}px "
                             f"(corruptions must return {self.image_size}x{self.image_size}).")
        return self.to_tensor(img), self.labels[idx]


def build_dataset(cfg: dict, split: str, corrupt=None) -> CifakeDataset:
    if split == "train" and corrupt is not None:
        raise ValueError("Corruptions are test-time only; refusing to corrupt the training split.")
    return CifakeDataset(repo_path(cfg["data"]["raw_dir"]), split_paths(cfg, split),
                         cfg["data"]["image_size"], corrupt)


def build_loader(cfg: dict, split: str, corrupt=None, shuffle: bool | None = None,
                 batch_size: int | None = None) -> DataLoader:
    ds = build_dataset(cfg, split, corrupt)
    shuffle = (split == "train") if shuffle is None else shuffle
    gen = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(ds, batch_size=batch_size or cfg["train"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["num_workers"], generator=gen)


# ---------------------------------------------------------------- CLI
def _check(cfg: dict) -> None:
    raw_dir = repo_path(cfg["data"]["raw_dir"])
    dirs = find_class_dirs(raw_dir)
    print(f"Detected layout under {raw_dir}:")
    for (pool, cls), d in sorted(dirs.items()):
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        expected = EXPECTED_COUNTS[(pool, cls)]
        flag = "" if n == expected else f"  <-- expected {expected}"
        print(f"  {pool}/{cls:4s} -> {d.relative_to(raw_dir).as_posix():30s} {n:6d} images{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--check", action="store_true", help="print detected layout and counts")
    ap.add_argument("--make-split", action="store_true", help="write train.txt / val.txt")
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.check or not args.make_split:
        _check(cfg)
    if args.make_split:
        out = repo_path(cfg["data"]["splits_dir"])
        if (out / "val.txt").exists() and not args.force:
            raise SystemExit(f"{out / 'val.txt'} already exists. The split is frozen and shared; "
                             "use --force only if you both agree to regenerate it.")
        train, val = make_split(repo_path(cfg["data"]["raw_dir"]), cfg["data"]["val_per_class"], cfg["seed"])
        write_split(train, out / "train.txt")
        write_split(val, out / "val.txt")
        n_val_fake = sum(label_from_path(p) for p in val)
        print(f"Wrote {len(train)} train / {len(val)} val paths "
              f"(val: {len(val) - n_val_fake} REAL, {n_val_fake} FAKE, seed={cfg['seed']}) to {out}")


if __name__ == "__main__":
    main()
