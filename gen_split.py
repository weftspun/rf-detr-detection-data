#!/usr/bin/env python3
"""Build a train/test Parquet split from a Roboflow Universe COCO export
(chibifire/clothing-instance-segmentation-1f7c9, version 2 -- the base,
non-augmented export: 696 train / 231 valid / 231 test images, each folder
holding its own _annotations.coco.json). Used for BOTH the detection and
segmentation modalities, since Roboflow's instance-segmentation COCO export
already carries both bbox and segmentation per annotation -- same source,
two repos, same schema convention as gen_split.py (COCO-native fields,
zstd-compressed Parquet, image bytes stored byte-identical to the export).

Roboflow's own train split becomes our train split. Its valid+test splits
are merged into our test split (more held-out data for generalization
tracking than either alone; image ids are re-namespaced by folder prefix
since Roboflow's per-split JSON restarts ids at 0).

Usage:
    uv run --with pyarrow gen_split_roboflow.py detection /path/to/clothing_coco detection
    uv run --with pyarrow gen_split_roboflow.py segmentation /path/to/clothing_coco segmentation
"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def build_rows(modality, split_dir, id_prefix):
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(ann_path) as f:
        coco = json.load(f)
    by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(a["image_id"], []).append(a)

    rows = []
    for image_id in sorted(anns_by_image.keys()):
        im_meta = by_id.get(image_id)
        if im_meta is None:
            continue
        img_path = os.path.join(split_dir, im_meta["file_name"])
        if not os.path.isfile(img_path):
            continue
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        anns = anns_by_image[image_id]
        row = {
            "image_id": id_prefix * 1_000_000 + image_id,
            "file_name": im_meta["file_name"],
            "width": im_meta["width"],
            "height": im_meta["height"],
            "image_bytes": image_bytes,
            "category_id": [a["category_id"] for a in anns],
            "bbox_xywh": [a["bbox"] for a in anns],
        }
        if modality == "segmentation":
            row["segmentation_json"] = [json.dumps(a["segmentation"]) for a in anns]
        rows.append(row)
    return rows, coco["categories"]


def write_split(rows, out_path):
    columns = {k: [r[k] for r in rows] for k in rows[0].keys()}
    table = pa.table(columns)
    pq.write_table(table, out_path, compression="zstd", compression_level=19)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {len(rows)} rows -> {out_path} ({size_mb:.1f} MB)")


def main():
    modality = sys.argv[1]
    coco_root = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    assert modality in ("detection", "segmentation")

    train_rows, categories = build_rows(modality, os.path.join(coco_root, "train"), id_prefix=0)
    valid_rows, _ = build_rows(modality, os.path.join(coco_root, "valid"), id_prefix=1)
    test_rows, _ = build_rows(modality, os.path.join(coco_root, "test"), id_prefix=2)
    held_out = valid_rows + test_rows

    print(f"{modality}: train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} "
          f"-> held_out(valid+test)={len(held_out)}")

    os.makedirs(out_dir, exist_ok=True)
    write_split(train_rows, os.path.join(out_dir, "train.parquet"))
    write_split(held_out, os.path.join(out_dir, "test.parquet"))
    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump(categories, f, indent=2)
    print(f"wrote {len(categories)} categories -> categories.json")


if __name__ == "__main__":
    main()
