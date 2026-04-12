#!/usr/bin/env python3
"""Assemble the YOLO-format dataset for cassini/issna training.

Rules enforced:
- Exclude all image_ids in test_split.csv (held out for final evaluation).
- Exclude all image_ids in dropped_by_human.txt.
- Include only images that (a) are in train_split.csv AND (b) have a
  corresponding YOLO label .txt file on disk from the auto-labeler.
- 10% random val split with seed=42, drawn from the labeled training set.
- Symlink images and labels into images/{train,val} and labels/{train,val}.
"""
import csv
import os
import random
import sys
from pathlib import Path

ROOT = Path("/Users/upadhyay/dev/ICES/object_detection_opencv_cpp")
DATA_DIR = ROOT / "data" / "cassini" / "issna"
ART_DIR = ROOT / "artifacts" / "cassini_issna"
OUT_DIR = ROOT / "data" / "cassini_issna_yolo"

SEED = 42
VAL_FRAC = 0.10


def load_set_from_csv(path):
    s = set()
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            s.add(row["image_id"])
    return s


def load_dropped(path):
    s = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            s.add(line)
    return s


def index_labels():
    """Return dict image_id -> (label_path, image_path, body)."""
    out = {}
    for body_dir in sorted(DATA_DIR.iterdir()):
        if not body_dir.is_dir():
            continue
        labels_dir = body_dir / "labels"
        images_dir = body_dir / "images"
        if not labels_dir.is_dir():
            continue
        for lbl in labels_dir.glob("*.txt"):
            img_id = lbl.stem
            img_path = images_dir / f"{img_id}.png"
            if not img_path.exists():
                continue
            # first-hit wins; label_body is from directory
            if img_id not in out:
                out[img_id] = (lbl, img_path, body_dir.name)
    return out


def symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    train_ids = load_set_from_csv(ART_DIR / "train_split.csv")
    test_ids = load_set_from_csv(ART_DIR / "test_split.csv")
    dropped = load_dropped(ART_DIR / "dropped_by_human.txt")

    print(f"train_split rows: {len(train_ids)}")
    print(f"test_split rows:  {len(test_ids)}")
    print(f"dropped rows:     {len(dropped)}")

    labeled = index_labels()
    print(f"label files on disk: {len(labeled)}")

    # Safety: no test image may have a label fed into training
    leaked = [iid for iid in labeled if iid in test_ids]
    if leaked:
        print(f"REFUSE: {len(leaked)} labeled images are in test_split", file=sys.stderr)
        sys.exit(2)

    # Build selected list: must be in train_split, not dropped, and have a label
    selected = []
    skipped_not_in_train = 0
    skipped_dropped = 0
    for iid, info in labeled.items():
        if iid in dropped:
            skipped_dropped += 1
            continue
        if iid not in train_ids:
            skipped_not_in_train += 1
            continue
        selected.append((iid, *info))

    print(f"skipped (dropped):      {skipped_dropped}")
    print(f"skipped (not in train): {skipped_not_in_train}")
    print(f"selected for training:  {len(selected)}")

    # Sorted for determinism, then shuffle with fixed seed
    selected.sort(key=lambda x: x[0])
    rng = random.Random(SEED)
    rng.shuffle(selected)

    n_val = max(1, int(round(len(selected) * VAL_FRAC)))
    val_set = selected[:n_val]
    train_set = selected[n_val:]
    print(f"train: {len(train_set)}  val: {len(val_set)}")

    # Wipe prior symlinks in output dirs
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = OUT_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()

    def place(split_name, rows):
        class_counts = {}
        body_counts = {}
        for iid, lbl, img, body in rows:
            symlink(img, OUT_DIR / "images" / split_name / f"{iid}.png")
            symlink(lbl, OUT_DIR / "labels" / split_name / f"{iid}.txt")
            body_counts[body] = body_counts.get(body, 0) + 1
            with lbl.open() as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    class_counts[cid] = class_counts.get(cid, 0) + 1
        return class_counts, body_counts

    train_cls, train_body = place("train", train_set)
    val_cls, val_body = place("val", val_set)
    print("train body counts:", dict(sorted(train_body.items())))
    print("val   body counts:", dict(sorted(val_body.items())))
    print("train class-id counts:", dict(sorted(train_cls.items())))
    print("val   class-id counts:", dict(sorted(val_cls.items())))

    # Write data.yaml
    class_names_path = ART_DIR / "class_names.txt"
    names = [l.strip() for l in class_names_path.read_text().splitlines() if l.strip()]
    assert len(names) == 64, f"expected 64 classes, got {len(names)}"

    yaml_text = [
        f"path: {OUT_DIR}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for n in names:
        yaml_text.append(f"  - {n}")
    (OUT_DIR / "data.yaml").write_text("\n".join(yaml_text) + "\n")
    print(f"wrote {OUT_DIR/'data.yaml'}")


if __name__ == "__main__":
    main()
