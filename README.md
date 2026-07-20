# rf-detr-detection-data

Train/test split for RF-DETR object-detection finetuning
([rf-detr-cpp](https://github.com/weftspun/rf-detr-cpp)), sourced from the
Roboflow Universe project
[chibifire/clothing-instance-segmentation-1f7c9](https://universe.roboflow.com/chibifire/clothing-instance-segmentation-1f7c9)
(version 2 -- the base export, no tiling/augmentation multiplication: 1158
source images). License: CC BY 4.0 (see `LICENSE.txt`).

## Contents

- `train.parquet` -- 696 images (Roboflow's own "train" split), zstd-compressed.
- `test.parquet` -- 462 images (Roboflow's "valid" + "test" splits merged,
  for more held-out data than either alone), zstd-compressed.
- `categories.json` -- the 47 clothing category id -> name mapping.
- `gen_split.py` -- the script that produced these files from a Roboflow
  COCO export (kept for reproducibility; requires a Roboflow API key to
  re-run against a newer project version).

## Schema (per row)

| column | type | meaning |
|---|---|---|
| `image_id` | int64 | unique id (namespaced by source split) |
| `file_name` | string | original file name |
| `width`, `height` | int32 | image dimensions (px) |
| `image_bytes` | binary | the JPEG file, byte-identical to the Roboflow export -- no re-encode/resize/crop applied by this repo |
| `category_id` | list<int64> | one entry per annotation instance |
| `bbox_xywh` | list<list<float>> | COCO-native `[x, y, w, h]`, absolute pixels, one per instance |

Images are already square-letterboxed to 432x432 by Roboflow's own
preprocessing (auto-orient + "Fit (black edges) in" resize) -- that
transform happened upstream of this repo, not applied here.

## Loading

```python
import pyarrow.parquet as pq
table = pq.read_table("train.parquet")
row = table.to_pylist()[0]
```

Consumed by `rf-detr-cpp`'s dataset loader via
`gen_reference/gen_from_parquet_split.py` (converts each split into the
existing `write_arr`-based `.bin` format `src/dataset.cpp` reads).
