#!/usr/bin/env python3
# json2yolo_seg.py
# Convert polygon-JSON (your schema) -> YOLO segmentation TXT
# - Input : <root>/annotations/{train,val}/*.json
# - Output: <root>/labels/{train,val}/*.txt
# - Line  : <class_id> x1 y1 x2 y2 ... xn yn    (x,y normalized to [0,1])
#
# Enhancements:
# - --zero-base : map class_id -> class_id-1 (for sources using 1..N)
# - --nc        : enforce class range [0, nc-1]; out-of-range labels are skipped and logged
# - Robust validity checks (min 3 points, non-zero area, even number of coords)
# - Conversion summary + CSV error log (yolo_convert_errors.csv)

import json
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

def normalize_polygon(polygon: List[List[float]], w: float, h: float) -> List[Tuple[float, float]]:
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: width={w}, height={h}")
    out: List[Tuple[float, float]] = []
    for pt in polygon:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        x, y = float(pt[0]), float(pt[1])
        nx = max(0.0, min(1.0, x / w))
        ny = max(0.0, min(1.0, y / h))
        out.append((nx, ny))
    return out

def poly_is_valid(poly: List[Tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False
    # Shoelace area
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) > 1e-10

def write_error(log_writer, json_path: Path, image_name: str, reason: str, extra: str = ""):
    log_writer.writerow({
        "json_path": str(json_path),
        "image_name": image_name,
        "reason": reason,
        "detail": extra
    })

def convert_one_json(json_path: Path,
                     out_txt_path: Path,
                     zero_base: bool = False,
                     nc: Optional[int] = None,
                     log_writer=None) -> int:
    """Convert a single JSON file. Returns number of objects written."""
    try:
        data: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        if log_writer: write_error(log_writer, json_path, "", "read_json_failed", str(e))
        return 0

    try:
        w = float(data["images"]["image_width"])
        h = float(data["images"]["image_height"])
        image_name = data["images"].get("image_filename", json_path.stem)
    except Exception as e:
        if log_writer: write_error(log_writer, json_path, "", "missing_image_meta", str(e))
        return 0

    annos = data.get("annotations", [])
    lines: List[str] = []
    for idx, obj in enumerate(annos):
        try:
            cls = int(obj["class_id"])
            if zero_base:
                cls = cls - 1
            if nc is not None and not (0 <= cls < nc):
                if log_writer: write_error(log_writer, json_path, image_name,
                                           "class_out_of_range",
                                           f"cls={cls}, nc={nc}, anno_idx={idx}")
                continue
            polygon = obj["polygon"]
        except Exception as e:
            if log_writer: write_error(log_writer, json_path, image_name, "missing_fields", str(e))
            continue

        norm = normalize_polygon(polygon, w, h)
        if not poly_is_valid(norm):
            if log_writer: write_error(log_writer, json_path, image_name,
                                       "invalid_polygon",
                                       f"points={len(norm)}")
            continue

        flat: List[str] = []
        for x, y in norm:
            flat.append(f"{x:.6f}")
            flat.append(f"{y:.6f}")
        if len(flat) % 2 != 0 or len(flat) < 6*2/2:  # safeguard; needs >=3 points
            if log_writer: write_error(log_writer, json_path, image_name, "coord_mismatch", f"len(flat)={len(flat)}")
            continue

        line = " ".join([str(cls)] + flat)
        lines.append(line)

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        out_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)

def process_split(root: Path, split: str, zero_base: bool, nc: Optional[int], log_writer) -> Tuple[int, int]:
    in_dir = root / "annotations" / split
    out_dir = root / "labels" / split
    if not in_dir.exists():
        print(f"[INFO] Skip: {in_dir} does not exist.")
        return 0, 0
    images = 0
    objects = 0
    for jf in sorted(in_dir.glob("*.json")):
        out_txt = out_dir / (jf.stem + ".txt")
        n = convert_one_json(jf, out_txt, zero_base=zero_base, nc=nc, log_writer=log_writer)
        images += 1
        objects += n
    print(f"[DONE] {split}: {images} files, {objects} objects -> {out_dir}")
    return images, objects

def main():
    ap = argparse.ArgumentParser(description="Convert polygon-JSON to YOLO segmentation labels.")
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (contains annotations/, images/)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="Splits to convert (default: train val)")
    ap.add_argument("--zero-base", action="store_true",
                    help="If set, map class_id -> class_id-1 (use when source uses 1..N)")
    ap.add_argument("--nc", type=int, default=None,
                    help="Number of classes. If set, skip labels with class not in [0, nc-1].")
    ap.add_argument("--error-log", type=Path, default=Path("yolo_convert_errors.csv"),
                    help="CSV path to write conversion issues.")
    args = ap.parse_args()

    # Prepare CSV error log
    args.error_log.parent.mkdir(parents=True, exist_ok=True)
    with args.error_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["json_path", "image_name", "reason", "detail"])
        writer.writeheader()

        total_imgs = 0
        total_objs = 0
        for sp in args.splits:
            imgs, objs = process_split(args.root, sp, args.zero_base, args.nc, writer)
            total_imgs += imgs
            total_objs += objs

    print(f"\n[SUMMARY] images={total_imgs}, objects={total_objs}")
    print(f"[NOTE] Output labels: <root>/labels/<split>/*.txt")
    print(f"[NOTE] Error log   : {args.error_log}")

if __name__ == "__main__":
    main()


# # 1) 클래스가 1~10이라면(=YOLO에선 0~9여야 함)
# python trash295_convert_yolo.py --root /gpfs/home1/danny472/Underwater/dataset/trash_295 --zero-base --nc 10

# # 2) 특정 스플릿만
# python trash295_convert_yolo.py --root /path/to/trash_295 --splits train --zero-base --nc 10

# # 3) 클래스 수 모름 → 우선 zero-base만 적용하고, corrupt 발생 시 error CSV 참고
# python trash295_convert_yolo.py --root /path/to/trash_295 --zero-base
