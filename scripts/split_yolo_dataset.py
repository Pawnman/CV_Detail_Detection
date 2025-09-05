import shutil
import random
from pathlib import Path
from typing import List, Tuple

# ========= Settings =========
SRC_IMG = Path("ds_2\img")   # folder with images
SRC_LBS = Path("ds_2\lbs")   # folder with label-тxt
OUT_ROOT = Path("dataset")  # where to put

# split proportions (summ = 1.0)
R_TRAIN, R_VAL, R_TEST = 0.7, 0.2, 0.1

# Transfering: "copy" or "move"
TRANSFER_MODE = "copy"

# random fix
SEED = 42

# If you want to force indexes 0..75 to be considered as "negatives",
# set True. Otherwise, negatives are determined by empty .txt.
USE_INDEX_RANGE_FOR_NEG = False
NEGATIVE_INDEX_RANGE = range(0, 76)  # 0..75 
# =============================


def is_empty_label(p: Path) -> bool:
    """True, if file .txt is empty or consists from spaces and linejumps,
    or file doesnt exist (the image is nagative for YOLO)."""
    if not p.exists():
        return True
    try:
        s = p.read_text(encoding="utf-8").strip()
        return len(s) == 0
    except Exception:
        # take it as empty in reading mode
        return True


def pair_list(img_dir: Path, lbs_dir: Path) -> List[Tuple[Path, Path, bool]]:
    """Collect list (img_path, lbl_path, is_negative)."""
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem  # example "img_000"
        lbl_path = lbs_dir / f"{stem}.txt"

        if USE_INDEX_RANGE_FOR_NEG:
            # take index from file name "img_XXX"
            # take number from the end
            try:
                idx = int(stem.split("_")[-1])
                is_neg = idx in NEGATIVE_INDEX_RANGE
            except ValueError:
                # When it is not possible —fallback to file check
                is_neg = is_empty_label(lbl_path)
        else:
            is_neg = is_empty_label(lbl_path)

        # .txt exists: or create empty
        if not lbl_path.exists():
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.write_text("", encoding="utf-8")

        pairs.append((img_path, lbl_path, is_neg))
    return pairs


def stratified_split(pairs: List[Tuple[Path, Path, bool]],
                     r_train: float, r_val: float, r_test: float,
                     seed: int):
    """Stratified split by is_negative flag."""
    assert abs((r_train + r_val + r_test) - 1.0) < 1e-6, "The shares must sum to 1.0"

    neg = [p for p in pairs if p[2]]
    pos = [p for p in pairs if not p[2]]

    random.Random(seed).shuffle(neg)
    random.Random(seed + 1).shuffle(pos)

    def split_one(lst):
        n = len(lst)
        n_train = int(round(n * r_train))
        n_val   = int(round(n * r_val))
        n_test  = n - n_train - n_val
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    neg_tr, neg_va, neg_te = split_one(neg)
    pos_tr, pos_va, pos_te = split_one(pos)

    train = neg_tr + pos_tr
    val   = neg_va + pos_va
    test  = neg_te + pos_te

    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    random.Random(seed).shuffle(test)

    return train, val, test


def make_dirs(root: Path):
    for split in ["train", "val", "test"]:
        for sub in ["img", "lbs"]:
            d = root / split / sub
            d.mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError("TRANSFER_MODE must be 'copy' or 'move'")


def dump_split(split_name: str, items: List[Tuple[Path, Path, bool]], out_root: Path):
    dst_img = out_root / split_name / "img"
    dst_lbs = out_root / split_name / "lbs"

    for img_path, lbl_path, _ in items:
        transfer(img_path, dst_img / img_path.name, TRANSFER_MODE)
        transfer(lbl_path, dst_lbs / lbl_path.name, TRANSFER_MODE)


def main():
    assert SRC_IMG.exists(), f"No folder: {SRC_IMG}"
    assert SRC_LBS.exists(), f"no folder: {SRC_LBS}"

    pairs = pair_list(SRC_IMG, SRC_LBS)
    total = len(pairs)
    neg_count = sum(1 for _, _, n in pairs if n)
    pos_count = total - neg_count

    print(f"[INFO] Total images: {total}")
    print(f"[INFO] Negatives (empty .txt): {neg_count}")
    print(f"[INFO] Positives (with annotations): {pos_count}")

    train, val, test = stratified_split(pairs, R_TRAIN, R_VAL, R_TEST, SEED)

    print(f"[SPLIT] train: {len(train)} (neg {sum(1 for *_ ,n in train if n)}, pos {sum(1 for *_ ,n in train if not n)})")
    print(f"[SPLIT] val  : {len(val)} (neg {sum(1 for *_ ,n in val if n)}, pos {sum(1 for *_ ,n in val if not n)})")
    print(f"[SPLIT] test : {len(test)} (neg {sum(1 for *_ ,n in test if n)}, pos {sum(1 for *_ ,n in test if not n)})")

    make_dirs(OUT_ROOT)
    dump_split("train", train, OUT_ROOT)
    dump_split("val",   val,   OUT_ROOT)
    dump_split("test",  test,  OUT_ROOT)

    print(f"[DONE] All files are in '{OUT_ROOT}' according YOLO-structure.")


if __name__ == "__main__":
    main()
