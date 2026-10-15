---
license: gpl-3.0
library_name: litert
pipeline_tag: object-detection
tags:
  - tflite
  - litert
  - android
  - on-device
  - faunapulse
  - insects
  - camera-traps
  - biodiversity
---

# insectDCT detector v8-s, LiteRT (fp16, 640 px) for FaunaPulse

An unofficial conversion of the **insectDCT v8-s** insect detector to a LiteRT `.tflite` file,
made for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. Not
provided or endorsed by the insectDCT authors.

## Files

{{FILES_TABLE}}

The `.json` file is the export manifest: source weights and their SHA-256, Ultralytics
version, and the agreement check below.

## What it is

- **Original model**: insectDCT detector (YOLO11s, one class `Insect`, trained at 1920 px on
  whole camera-trap pictures), weights as packaged by the
  [InsectAI Model Zoo](https://github.com/InsectAI-COST-Action/insect-model-zoo) (`insects8Color11s.pt`).
- **Conversion**: weights stored as fp16, input 640 x 640. At 640 px insects look larger in
  pixels than in training; FaunaPulse feeds the square area of interest around a flower,
  which enlarges insects anyway. Compare input sizes on your own footage.
- **Input**: one RGB image, 640 x 640, values 0..1. **Output**: the standard Ultralytics YOLO
  detection tensor `[1, 4 + classes, N]` without NMS (the app does NMS). The Ultralytics
  metadata (class names, input size) is embedded in the file.

## Checks

- **Agreement with PyTorch** (same model, not an accuracy test): on 30 pictures, all 17
  PyTorch boxes found again, mean IoU 0.9999, confidences within 0.001.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888, model time only): 34 ms per picture on the GPU,
  1.6 s on the CPU (2 threads).

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads this file from a link
pinned to one commit of this repository and checks its SHA-256. It can also be imported by
hand (Import model files…).

## How it was made

`detector_export/export_detector.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/detector_export)
(CPU only, Ultralytics 8.4.170):

```bash
python export_detector.py --weights insects8Color11s.pt --name insectdct-v8-s --imgsz 640 \
    --check-images /path/to/check_pictures --out out
```

## Original work, licence and citation

insectDCT: Bjerge, Wogram, Serra-Marin, Sakhiashvili & Høye (2026). InsectDCT: A generalized
pipeline for detection, taxonomic classification, and tracking of insects in camera-trap
recordings. bioRxiv. https://doi.org/10.64898/2026.07.07.736939.
Code: https://github.com/kimbjerge/insectDCT (GPL-3.0).

This file follows the licence of the original (GPL-3.0). The model was trained with
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Please cite the original
paper when you use this model.

## Notes

- Files in this repository are never replaced: the app pins each download to a commit.
  New weights get a new file name.
- Limitations are those of the original model.
