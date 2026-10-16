---
license: gpl-3.0
library_name: litert
pipeline_tag: object-detection
tags:
  - tflite
  - litert
  - android
  - on-device
  - mobile-sensing
  - faunapulse
  - insects
  - pollinators
  - camera-traps
  - biodiversity
  - biodiversity-monitoring
  - yolo11
  - ultralytics
---

# insectDCT detector v8 (YOLO11s), LiteRT (fp16, 640 px) for FaunaPulse

An unofficial conversion of the **insectDCT** insect detector to a LiteRT `.tflite` file, made
for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. insectDCT is
the work of Bjerge et al. (2026), see [Original work](#original-work) below. This file is not
provided or endorsed by the insectDCT authors.

> **This detector is for demo purposes for the FaunaPulse app.**
> I did not train this model, and its authors did not make it for FaunaPulse.
> It is converted to `tflite` and published here so that FaunaPulse can be tried
> with a ready-made detector. How well it works in the app is unknown and not tested,
> especially in live use on natural backgrounds with insects flying to and visiting flowers.
> The model was trained on whole pictures from fixed time-lapse camera traps
> (resized to 1920 × 1080 pixels). FaunaPulse runs it on live phone video at 640 px.
> See also [Limitations](#limitations).

## Files

{{FILES_TABLE}}

- `.json`: the export manifest: source weights and their SHA-256, the Ultralytics version saved
  in the source checkpoint (8.4.156), and the agreement check below. Its `"file"` field gives the
  file's name before upload (`insectdct-v8-s_640_fp16.tflite`; same SHA-256), and its `script`
  field, `tool/detector_export/export_detector.py`, is where the export script lived at the time
  (in the FaunaPulse app repository); it is now `detector_export/export_detector.py` of
  fauna-pulse-models.
- `LICENSE`: insectDCT's licence (GPL-3.0), copied from the authors' repository.

**The name**: the authors call this model `insects8Color11s`: a YOLO11s detector for colour (RGB)
pictures, trained on their detection dataset version 8 (DV8). So "v8" is the dataset version,
not YOLOv8. FaunaPulse calls it `insectdct-v8s` (dataset version 8, size s).

## Original work

- **Authors**: Kim Bjerge, Simon F. A. Wogram, Pau Enric Serra-Marin, Otar Sakhiashvili,
  Andreas Bonnerup Jepsen and Toke T. Høye.
- **Paper**: *InsectDCT: A generalized pipeline for detection, taxonomic classification, and
  tracking of insects in camera-trap recordings*, bioRxiv preprint (2026),
  https://doi.org/10.64898/2026.07.07.736939.
- **Code and trained models**: https://github.com/kimbjerge/insectDCT (GPL-3.0).
- **Source weights**: the authors' file `runs/detect/insects8Color11s/weights/best.pt` in the
  [insectDCT repository](https://github.com/kimbjerge/insectDCT/tree/e459ae87ebe39732963e77e0fcdd0a9a28672a08/runs/detect/insects8Color11s/weights)
  (commit e459ae8, 27 September 2026; SHA-256
  `719ff88f1b8c9dc57ef0fed5e741adc12900a4c3e93c31f9a3c071379947e4ac`). The file converted here
  is byte-identical to it (same SHA-256, checked against a download from that link on
  16 October 2026). It was saved under the name of its run folder, `insects8Color11s.pt`, the
  name in the export command below and in the `.json`.
- This detector is newer than the one evaluated in the paper, which used dataset version 6.

## What it is

- **Network**: YOLO11s, one class, `Insect`; it finds insects and does not identify them. Trained
  by the authors on whole camera-trap pictures (they resize full pictures to
  1920 x 1080 pixels).
- **Conversion**: weights stored as fp16, input 640 x 640. Nothing was retrained. A whole
  picture shrunk to 640 px shows insects smaller than in training; FaunaPulse feeds the square
  area of interest around a flower instead, which enlarges them again. It is unknown how this
  impacts model performance when deployed on FaunaPulse app.
- **Input**: one RGB image `[1, 3, 640, 640]` (NCHW), float32, values 0..1.
- **Output**: the standard Ultralytics YOLO detection tensor `[1, 5, 8400]` (4 box values and 1
  class score per candidate) without NMS (the FaunaPulse app does NMS currently).
- **Embedded metadata**: Ultralytics writes class names and input size into the file. Its
  `description` ("Ultralytics YOLO11s model trained on data/insects8Color.yaml"), `author`
  ("Ultralytics") and `license` ("AGPL-3.0 License") fields are default values that the
  Ultralytics export tool writes into every file; they do not name the model's authors, who are
  the insectDCT authors (see [Original work](#original-work) and [Licence](#licence)).

## Checks

- **Agreement with PyTorch** (same model, not an accuracy test): on 30 frames of a video clip of
  bumblebees, all 17 PyTorch boxes found again, mean IoU 0.9999, confidences within 0.001.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888, model time only): 34 ms per picture on the GPU,
  1.6 s on the CPU (2 threads).

## Use in FaunaPulse

In FaunaPulse versions after v0.8.0-alpha.1, home screen menu, **Download & import models**
downloads this file from a link pinned to one commit of this repository and checks its
SHA-256. With any version you can also download it here and copy it via USB into the phone (Import model files…).

## How it was made

`detector_export/export_detector.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/detector_export)
(CPU only, Ultralytics 8.4.170, 1 October 2026):

```bash
python export_detector.py --weights insects8Color11s.pt --name insectdct-v8-s --imgsz 640 \
    --check-images /path/to/check_pictures --out out
```

The script wrote `insectdct-v8-s_640_fp16.tflite`, published here as
`insectdct-v8s_640_fp16.tflite` (the FaunaPulse naming rule; same bytes).

## Limitations

- Made by its authors for daylight time-lapse camera-trap pictures of insects on flowers and
  other vegetation, mostly from European sites; dataset version 8 adds sites in Georgia, the
  Arctic and Danish farmland. The authors' tests of this YOLO11s model give a mean F1 of 0.81 over
  9 camera-trap test sets (`metrics/detectTestColorSV8.csv` in the insectDCT repository).
- How it does on other kinds of pictures (phone video, insects in flight, light traps, lab or
  hand-held photos) is not known.
- FaunaPulse runs it at 640 × 640 pixels instead of 1920 × 1080 pixels.
- The agreement check shows that the conversion matches the original, not how well it finds
  insects in your setting.

## Licence

GPL-3.0, the licence of the insectDCT repository, which also holds the original weights; the
authors give the weights no separate licence. A copy of the licence is in `LICENSE`. Changes
made here: converted to LiteRT with fp16 weights and a 640 x 640 input on 1 October 2026; nothing
was retrained. The conversion code is in fauna-pulse-models (AGPL-3.0), linked above.

The insectDCT detector was fine-tuned with Ultralytics YOLO11 from Ultralytics' pretrained
YOLO11s weights, and Ultralytics offers both under AGPL-3.0 or a paid Enterprise License, so
check both licences before any commercial use.

## Citation

If you use this file, please cite the insectDCT paper:

Bjerge, K., Wogram, S. F. A., Serra-Marin, P. E., Sakhiashvili, O., Jepsen, A. B., & Høye, T. T.
(2026). InsectDCT: A generalized pipeline for detection, taxonomic classification, and tracking
of insects in camera-trap recordings. *bioRxiv* (preprint, not peer reviewed).
https://doi.org/10.64898/2026.07.07.736939

```bibtex
@article{bjerge2026insectdct,
  author       = {Bjerge, Kim and Wogram, Simon F. A. and Serra-Marin, Pau Enric and Sakhiashvili, Otar and Jepsen, Andreas Bonnerup and H{\o}ye, Toke T.},
  title        = {{InsectDCT}: A generalized pipeline for detection, taxonomic classification, and tracking of insects in camera-trap recordings},
  journal      = {bioRxiv},
  elocation-id = {2026.07.07.736939},
  year         = {2026},
  doi          = {10.64898/2026.07.07.736939},
  note         = {Preprint}
}
```

Also relevant:

- Data used in the paper (dataset version 6; the version 8 data are not published):
  Bjerge, K., Wogram, S. F. A., Serra-Marin, P. E., Sakhiashvili, O., Jepsen, A. B., & Høye,
  T. T. (2026). InsectDCT: Datasets for training and evaluating pipeline processing insect camera
  recordings [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21154490 (CC BY 4.0)
- Training software: Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11 (Version 11.0.0) [Computer
  software]. https://github.com/ultralytics/ultralytics

## Notes

- Files in this repository are never replaced: the app pins each download to a commit and
  checks its SHA-256. New weights get a new file name.
- The authors also publish a medium model (YOLO11m).
