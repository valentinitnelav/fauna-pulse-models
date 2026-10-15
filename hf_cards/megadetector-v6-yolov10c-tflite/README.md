---
license: agpl-3.0
library_name: litert
pipeline_tag: object-detection
tags:
  - tflite
  - litert
  - android
  - on-device
  - faunapulse
  - megadetector
  - camera-traps
  - wildlife
---

# MegaDetector V6 (YOLOv10-c), LiteRT (INT8, 256 px) for FaunaPulse

An unofficial conversion of **MegaDetector V6** (variant `MDV6-yolov10-c`) to a full-INT8
LiteRT `.tflite` file, made for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse)
Android app. Not provided or endorsed by Microsoft.

## Files

{{FILES_TABLE}}

## What it is

- **Original model**: MegaDetector V6, `MDV6-yolov10-c.pt` (md5 `1ecc38fbe462320ea33bf3c57e9e1561`)
  from [Pytorch-wildlife-model-weights, v27](https://zenodo.org/records/15398270) (Zenodo), by
  the Microsoft AI for Good Lab. Classes: `animal`, `person`, `vehicle`.
- **Conversion**: full INT8 (weights and activations 8-bit), input 256 x 256, exported
  without NMS (the app does NMS). The INT8 calibration used a representative set of pictures
  chosen to approximate FaunaPulse smartphone conditions.
- **Input**: one RGB image, 256 x 256. **Output**: Ultralytics YOLO detection tensor without
  NMS. The Ultralytics metadata (class names, input size) is embedded in the file.

MegaDetector finds animals in general, not insects specifically. For insects, see the other
models of the FaunaPulse collection.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads this file from a link
pinned to one commit of this repository and checks its SHA-256. It can also be imported by
hand (Import model files…).

## How it was made

Ultralytics TFLite export with INT8 calibration (`nms=False, end2end=False`), with the
export scripts of that time, kept in
[fauna-pulse-models/legacy](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/legacy).
This exact file has been offered since September 2026; its SHA-256 is in the table above.

## Original work, licence and citation

MegaDetector: https://github.com/microsoft/MegaDetector (model variants:
https://github.com/microsoft/MegaDetector#model-variants), release
[megadetector-v6.0](https://github.com/microsoft/MegaDetector/releases/tag/megadetector-v6.0).

This file follows the licence of the original MegaDetector V6 variant (AGPL-3.0). Please cite
MegaDetector as its authors ask when you use this model.

## Notes

- Files in this repository are never replaced: the app pins each download to a commit.
  New weights get a new file name.
- INT8 can lose a little accuracy compared with the original; check on your own pictures.
