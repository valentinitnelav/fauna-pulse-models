---
license: agpl-3.0
library_name: litert
pipeline_tag: object-detection
tags:
  - tflite
  - litert
  - android
  - on-device
  - mobile-sensing  
  - faunapulse
  - megadetector
  - camera-traps
  - biodiversity
  - biodiversity-monitoring  
  - wildlife
  - yolov10
  - ultralytics
---

# MegaDetector V6, LiteRT for FaunaPulse

An unofficial conversion of **MegaDetector V6** (variant `MDV6-yolov10-c`) to a LiteRT `.tflite`
file, made for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app.
MegaDetector is made by the Microsoft AI for Good Lab, see [Original work](#original-work)
below. This file is not provided or endorsed by Microsoft or the MegaDetector authors.

> **This detector is for demo purposes for the FaunaPulse app.**
> I did not train this model, and its authors did not make it for FaunaPulse:
> it was trained on camera-trap pictures to find animals (its authors name mammals,
> birds, reptiles and insects, and it works best on large mammals), people and vehicles.
> It is converted to `tflite` and published here so that FaunaPulse can be tried
> with a ready-made detector, for example at a feeding site or a nest entrance.
> It has not yet been tested at scale in the app, so how well it works there is not known yet.
> Its authors say it works best on large mammals, so for insects, especially in live use on
> natural backgrounds with insects flying to and visiting flowers, prefer the insect detectors
> of the FaunaPulse collection.
> See also [Limitations](#limitations).

## Files

{{FILES_TABLE}}

- `mdv6-yolov10c_256_w8a32.tflite`: `MDV6-yolov10-c` at 256 x 256 px, 8-bit weights, with the
  values between layers kept as 32-bit floats (w8a32). The same bytes have shipped inside the app
  since release v0.7.0-alpha.1 (September 2026), and since 1 October 2026 also as the separate
  file `MDV6-yolov10-c_int8_256.tflite` of the app's GitHub release v0.8.0-alpha.1. The name
  changed here (the old one followed Ultralytics' "int8" naming, see below), the SHA-256 did not.
- `LICENSE`: the AGPL-3.0 text (the licence of this variant).

## Original work

- **Made by**: the Microsoft AI for Good Lab. The weights deposit lists Zhongqi Miao, Andrés
  Adolfo Hernández, Luisa Fernanda Vargas, Isai Daniel Chacón, Daniela Ruiz and Carl Chalmers.
  MegaDetector versions 1 to 5 were first developed at Microsoft AI for Earth by Dan Morris and
  colleagues.
- **Code and documentation**: https://github.com/microsoft/MegaDetector (code: MIT). All variants
  and their licences: [MegaDetector Model Zoo](https://microsoft.github.io/MegaDetector/model_zoo/).
  The weights are published on Zenodo and loaded by
  [PyTorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife).
- **Variant**: `MDV6-yolov10-c`, the authors' compact YOLOv10 model (YOLOv10-N size, 2.3 million
  parameters), three classes: `animal`, `person`, `vehicle`.
- **Source weights**: `MDV6-yolov10-c.pt` (md5 `1ecc38fbe462320ea33bf3c57e9e1561`, the checksum
  Zenodo lists for it) from the Zenodo deposit
  [Pytorch-wildlife-model-weights](https://doi.org/10.5281/zenodo.15398270) (version v22, the
  record PyTorch-Wildlife downloads it from; versions v24 to v27 hold the identical file).

## What it is

- **Conversion**: Ultralytics TFLite export with `int8=True`, 256 x 256 px, `nms=False`
  (Ultralytics 8.3.235, 7 August 2026, as recorded in the file's own metadata). In that
  Ultralytics version this gives 8-bit weights with 32-bit float activations (dynamic-range
  quantisation, "w8a32"), not full INT8: Ultralytics loads calibration pictures for its other
  INT8 files, but they do not change this one. The exact command was not logged. The early export scripts of that time are kept for reference in
  [fauna-pulse-models/legacy](https://github.com/valentinitnelav/fauna-pulse-models/tree/736ed0f89d0b9e77bb1e212047fef722e33eb780/legacy)
  (an earlier commit; no longer in the current repository), but in their kept form they pass options that Ultralytics 8.3.235 does not
  accept, so they are not the exact recipe of this file.
- **Input**: one RGB image `[1, 256, 256, 3]` (NHWC), float32, values 0..1.
- **Output**: one float32 tensor `[1, 300, 6]`: up to 300 boxes, each with 4 box coordinates, a
  score and a class index, from YOLOv10's head, which needs no NMS (the app's NMS step does no
  harm).
- **Embedded metadata**: Ultralytics writes class names and input size into the file. Its
  `description` ("Ultralytics YOLOv10n model trained on" plus the training path stored in the
  authors' own checkpoint), `author` ("Ultralytics") and `license` ("AGPL-3.0 License") fields
  are default values that the Ultralytics export tool writes into every file; they do not name
  the model's authors (see [Original work](#original-work)).

## Use in FaunaPulse

In FaunaPulse versions after v0.8.0-alpha.1, home screen menu, **Download & import models**
downloads this file from a link pinned to one commit of this repository and checks its
SHA-256. With any version you can also download it here and import it by hand (Import model
files…).

## Limitations

- MegaDetector was trained on camera-trap pictures to find animals, people and vehicles.
  It does not identify species.
- It was trained at 640 px, and PyTorch-Wildlife runs it at 1280 px. This file takes 256 px, so
  small animals become even smaller. The authors' figures for the original (76.8% animal recall,
  87.2% mAP50) do not apply to this file.
- It was not trained or tested by its authors on phone video of insects on flowers. For insects,
  see the other models of the FaunaPulse collection, and check any model on your own recordings.
- 8-bit weights can lose accuracy compared with the original.

## Licence

AGPL-3.0, the licence the MegaDetector authors give for `MDV6-yolov10-c` in the
[MegaDetector Model Zoo](https://microsoft.github.io/MegaDetector/model_zoo/) (the
Ultralytics-based V6 variants are AGPL-3.0; the MegaDetector code is MIT, and some other V6
variants are MIT or Apache-2.0). The Zenodo deposit as a whole is labelled CC BY 4.0; this card
follows the authors' licence for this variant. A copy of the AGPL-3.0 text is in `LICENSE`. This
file is a converted (modified) version, made on 7 August 2026 with Ultralytics' own TFLite export
(see [What it is](#what-it-is)); the early export scripts of that time and the current export
tools are in fauna-pulse-models (AGPL-3.0).

## Citation

If you use this file, please cite MegaDetector as its authors ask
(https://microsoft.github.io/MegaDetector/cite/):

- Hernandez, A., Miao, Z., Vargas, L., Beery, S., Dodhia, R., Arbelaez, P., & Lavista Ferres,
  J. M. (2024). Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation.
  arXiv:2405.12930. https://doi.org/10.48550/arXiv.2405.12930
- Beery, S., Morris, D., & Yang, S. (2019). Efficient Pipeline for Camera Trap Image Review.
  Data Mining and AI for Conservation Workshop at KDD 2019. arXiv:1907.06772.
  https://doi.org/10.48550/arXiv.1907.06772

```bibtex
@misc{hernandez2024pytorchwildlife,
  title         = {Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
  author        = {Hernandez, Andres and Miao, Zhongqi and Vargas, Luisa and Beery, Sara and Dodhia, Rahul and Arbelaez, Pablo and Lavista Ferres, Juan M.},
  year          = {2024},
  eprint        = {2405.12930},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2405.12930}
}
@misc{beery2019efficient,
  title         = {Efficient Pipeline for Camera Trap Image Review},
  author        = {Beery, Sara and Morris, Dan and Yang, Siyu},
  year          = {2019},
  eprint        = {1907.06772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.1907.06772}
}
```

Neither paper describes this exact variant (the Pytorch-Wildlife paper introduces V6 with a
YOLOv9 model); the variant itself is documented in the MegaDetector Model Zoo and the weights
deposit. Also relevant:

- Weights: Miao, Z., Hernández, A. A., Vargas, L. F., Chacón, I. D., Ruiz, D., & Chalmers, C.
  (2025). Pytorch-wildlife-model-weights (Version v22) [Model weights]. Zenodo.
  https://doi.org/10.5281/zenodo.15398270
- Architecture: Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024).
  YOLOv10: Real-Time End-to-End Object Detection. *Advances in Neural Information Processing
  Systems 37*. https://doi.org/10.52202/079017-3429

## Notes

- Files in this repository are never replaced: the app pins each download to a commit and
  checks its SHA-256. New weights get a new file name.
