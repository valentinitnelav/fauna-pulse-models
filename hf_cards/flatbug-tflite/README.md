---
license: mit
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
  - yolov8
  - ultralytics
---

# flat-bug N and S, LiteRT (fp16, 640 px) for FaunaPulse

Unofficial conversions of the **flat-bug** arthropod detector (sizes N and S) to LiteRT
`.tflite` files, made for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse)
Android app. flat-bug is the work of Svenning et al. (2026), see
[Original work](#original-work) below. These files are not provided or endorsed by the
flat-bug authors.

> **This detector is for demo purposes for the FaunaPulse app.**
> I did not train this model, and its authors did not make it for FaunaPulse.
> It is converted to `tflite` and published here so that FaunaPulse can be tried
> with a ready-made detector. It has not yet been tested at scale in the app, so how well
> it works there is not known yet, especially in live use on natural backgrounds with insects
> flying to and visiting flowers. The model was not trained for that kind of deployment.
> See also [Limitations](#limitations).

## Files

{{FILES_TABLE}}

- `flatbug-n_640_fp16.tflite`: flat-bug N (nano), the faster one. `flatbug-s_640_fp16.tflite`:
  flat-bug S (small), about 3.5 times larger. On a test phone it took about 1.8 times as long on
  the GPU and 2.6 times as long on the CPU (see [Checks](#checks)).
- `.json`: the export manifest of each file: source weights and their SHA-256, the Ultralytics
  version saved in the source checkpoint (8.3.15) and file sizes. Its
  `script` field, `tool/detector_export/export_detector.py`, is where the export script lived
  at the time (in the FaunaPulse app repository); it is now `detector_export/export_detector.py`
  of fauna-pulse-models.
- `LICENSE`: flat-bug's licence (MIT), copied from the authors' repository.

## Original work

- **Authors**: Asger Svenning, Guillaume Mougeot, Jamie Alison, Daphne Chevalier, Nisa Chavez
  Molina, Song-Quan Ong, Kim Bjerge, Juli Carrillo, Toke Thomas Høye and Quentin Geissmann.
- **Paper**: *A general method for detection and segmentation of terrestrial arthropods in
  images*, Methods in Ecology and Evolution 17(3), 727-739 (2026),
  https://doi.org/10.1111/2041-210x.70249.
- **Code and documentation**: https://github.com/darsa-group/flat-bug (MIT).
- **Training data**: the flatbug dataset, a compilation of published datasets from lab and
  field imaging systems (Zenodo, https://doi.org/10.5281/zenodo.14761447, CC BY 4.0).
- **Source weights**: `flat_bug_N.pt` (SHA-256
  `e54eb199fb8397f70dc36aca4727b92836fb7f56b11f8544be60f54c13d273c0`) and `flat_bug_S.pt`
  (SHA-256 `b410d2c902c8975f6c8263ee74871e618e55d670f86cf49ad49869de9a9ccc03`), which the
  authors publish in their model store at Aarhus University
  (https://anon.erda.au.dk/cgi-sid/ls.py?share_id=Bb0CR1FHG6&current_dir=models; the files:
  https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/flat_bug_N.pt and
  https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/flat_bug_S.pt); the `flat-bug` package
  downloads the same files. The files converted here are byte-identical to these (same SHA-256,
  checked against a download from these links on 16 October 2026).
- N and S are the original 2024 models of the paper (YOLOv8n-seg and YOLOv8s-seg). Since
  flat-bug 1.2 (July 2026) the authors' default is a newer model, `flat_bug_M_v2.pt` (medium
  size, YOLO26), which is not converted here.

## What it is

- **Network**: one class, `insect` (any terrestrial arthropod). flat-bug finds arthropods; it
  does not identify them.
- **Changes**: the segmentation (outline) branch is removed, so each file is a plain box
  detector; the box and class layers are untouched. Weights stored as fp16, input 640 x 640.
  Nothing was retrained.
- **Input**: one RGB image `[1, 3, 640, 640]` (NCHW), float32, values 0..1.
- **Output**: the standard Ultralytics YOLO detection tensor `[1, 5, 8400]` (4 box values and 1
  class score per candidate) without NMS (the app does NMS).
- **Embedded metadata**: Ultralytics writes class names and input size into the file. Its
  `description` ("Ultralytics YOLOv8n-seg model trained on" plus the authors' training path),
  `author` ("Ultralytics") and `license` ("AGPL-3.0 License") fields are default values that the
  Ultralytics export tool writes into every file; they do not name the model's authors and are
  not a statement by them (see [Original work](#original-work) and [Licence](#licence)).

## Checks

- **Accuracy**: no large-scale test of these files in FaunaPulse has been carried out yet. Converting a
  model to `.tflite` and storing its weights with fewer bits (here fp16) can change its
  outputs, so no claim is made here about how accurately they find insects when deployed in
  FaunaPulse (see [Limitations](#limitations)).
- **Phone speed** (Xiaomi 11T Pro, Snapdragon 888, model time only): N 19 ms per picture on the GPU,
  0.65 s on the CPU (2 threads); S 34 ms on the GPU, 1.7 s on the CPU.

## Use in FaunaPulse

In FaunaPulse versions after v0.8.0-alpha.1, home screen menu, **Download & import models**
downloads a file from a link pinned to one commit of this repository and checks its SHA-256.
With any version you can also download a file here and import it by hand (Import model files…).

## How it was made

`detector_export/export_detector.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/detector_export)
(CPU only, Ultralytics 8.4.170, 1 October 2026):

```bash
python export_detector.py --weights flat_bug_N.pt --name flatbug-n --imgsz 640 --out out
python export_detector.py --weights flat_bug_S.pt --name flatbug-s --imgsz 640 --out out
```

## Limitations

- flat-bug was trained mostly on top-down pictures of arthropods on flat backgrounds, from plain
  to somewhat busy (sticky cards, light-trap screens, pitfall samples, scans, lab trays; various imaging systems), cut
  into 1024 x 1024 px tiles. Its authors call this the model's central domain; only one of the
  training datasets shows insects on natural backgrounds (citizen-science photos, as described
  in the preprint, https://doi.org/10.1101/2025.04.08.647223). Insects on flowers and foliage
  are outside that domain, so missed insects and false boxes may be more frequent there.
- These files run the network once on a 640 x 640 picture, without flat-bug's multi-scale
  tiling, so the accuracy reported in the paper cannot be assumed for them.
- N is the smallest flat-bug model; the preprint reports that the smallest model loses the
  most on the hardest datasets.
- These files have not yet been tested at scale in FaunaPulse, so how well they work there is
  not known yet. The app can run them, but the results depend on many factors: image quality
  (focus, motion blur, light, resolution, video compression), how large the insects appear in
  the picture (distance, zoom, the area of interest), the background, the phone and its camera,
  and the app's settings (for example the confidence threshold and the input size). Choose
  between models and sizes on your own footage.

## Licence

MIT, the licence of flat-bug: Copyright (c) 2025 Asger Svenning, Quentin Geissmann et al. The
full text is in `LICENSE`, copied from https://github.com/darsa-group/flat-bug/blob/main/LICENSE;
keep it with any copy. The authors added this licence "to make it clear that the code and models
are actually open source" (https://github.com/darsa-group/flat-bug/pull/120).

flat-bug was trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
starting from Ultralytics' pretrained YOLOv8-seg models, and these files were exported with
Ultralytics. Ultralytics offers its software and models under AGPL-3.0 or a paid Enterprise
License and states that "all Ultralytics YOLO trained models fall under the AGPL-3.0 License by
default" (https://www.ultralytics.com/license). So these files are also covered by AGPL-3.0:
use them under its terms, or with an Ultralytics Enterprise License (for example for
closed-source commercial use). The training dataset is CC BY 4.0.

## Citation

If you use these files, please cite the flat-bug paper:

Svenning, A., Mougeot, G., Alison, J., Chevalier, D., Chavez Molina, N., Ong, S.-Q., Bjerge, K.,
Carrillo, J., Høye, T. T., & Geissmann, Q. (2026). A general method for detection and
segmentation of terrestrial arthropods in images. *Methods in Ecology and Evolution*, 17(3),
727-739. https://doi.org/10.1111/2041-210x.70249

```bibtex
@article{svenning2026flatbug,
  author  = {Svenning, Asger and Mougeot, Guillaume and Alison, Jamie and Chevalier, Daphne and {Chavez Molina}, Nisa and Ong, Song-Quan and Bjerge, Kim and Carrillo, Juli and H{\o}ye, Toke Thomas and Geissmann, Quentin},
  title   = {A general method for detection and segmentation of terrestrial arthropods in images},
  journal = {Methods in Ecology and Evolution},
  year    = {2026},
  volume  = {17},
  number  = {3},
  pages   = {727--739},
  doi     = {10.1111/2041-210x.70249}
}
```

Also relevant:

- Training data: Geissmann, Q., & Svenning, A. (2025). flatbug-dataset a compilation of dataset
  of terrestrial arthropodes on various surfaces [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.14761447 (each of its datasets has its own DOI).
- Code: Svenning, A., Mougeot, G., Alison, J., Chevalier, D., Chavez Molina, N. L., Ong, S.-Q.,
  Bjerge, K., Carrillo, J., Høye, T. T., & Geissmann, Q. (2026). darsa-group/flat-bug (v0.4.1)
  [Computer software]. Zenodo.
  https://doi.org/10.5281/zenodo.18164125
- Training and export software: Jocher, G., Qiu, J., & Chaurasia, A. (2023). Ultralytics YOLO
  (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

## Notes

- Model files in this repository are never replaced: the app pins each download to a
  commit and checks its SHA-256. New weights get a new file name.
- The `.json` manifests, which the app does not download, may be corrected.
