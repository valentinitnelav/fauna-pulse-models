---
license: mit
library_name: litert
pipeline_tag: object-detection
tags:
  - tflite
  - litert
  - android
  - on-device
  - faunapulse
  - insects
  - arthropods
  - biodiversity
---

# flat-bug n, LiteRT (fp16, 640 px) for FaunaPulse

An unofficial conversion of the **flat-bug n** arthropod detector to a LiteRT `.tflite` file,
made for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app (live
detection of flower visitors on the phone). Not provided or endorsed by the flat-bug authors.

## Files

{{FILES_TABLE}}

The `.json` file is the export manifest: source weights and their SHA-256, Ultralytics
version, and the agreement check below.

## What it is

- **Original model**: flat-bug n (YOLOv8-seg, one class `insect`, trained at 1024 px), weights
  as packaged by the [InsectAI Model Zoo](https://github.com/InsectAI-COST-Action/insect-model-zoo)
  (`flat_bug_N.pt`).
- **Changes**: the segmentation (outline) branch is removed, so the file is a plain box
  detector; the box and class layers are untouched. Weights stored as fp16, input 640 x 640.
- **Input**: one RGB image, 640 x 640, values 0..1. **Output**: the standard Ultralytics YOLO
  detection tensor `[1, 4 + classes, N]` without NMS (the app does NMS). The Ultralytics
  metadata (class names, input size) is embedded in the file.

## Checks

- **Agreement with PyTorch** (same model, not an accuracy test): on 30 pictures, all 28
  PyTorch boxes found again, mean IoU 0.9996, confidences within 0.006.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888, model time only): 19 ms per picture on the GPU,
  0.65 s on the CPU (2 threads). Choose between models and input sizes on your own footage.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads this file from a link
pinned to one commit of this repository and checks its SHA-256. It can also be imported by
hand (Import model files…).

## How it was made

`detector_export/export_detector.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/detector_export)
(CPU only, Ultralytics 8.4.170):

```bash
python export_detector.py --weights flat_bug_N.pt --name flatbug-n --imgsz 640 \
    --check-images /path/to/check_pictures --out out
```

## Original work, licence and citation

flat-bug: Svenning, Mougeot, Alison, Chevalier, Chavez Molina, Ong, Bjerge, Carrillo, Høye &
Geissmann (2026). A general method for detection and segmentation of terrestrial arthropods in
images. *Methods in Ecology and Evolution* 17(3), 727-739.
https://doi.org/10.1111/2041-210x.70249. Code: https://github.com/darsa-group/flat-bug (MIT).

This file follows the licence of the original (MIT). flat-bug was trained with
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Please cite the original
paper when you use this model.

## Notes

- Files in this repository are never replaced: the app pins each download to a commit.
  New weights get a new file name.
- Limitations are those of the original model; the check pictures show that the conversion
  agrees with the original, not how well it finds insects in your setting.
