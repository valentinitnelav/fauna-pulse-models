---
license: agpl-3.0
library_name: litert
pipeline_tag: object-detection
tags:
  - tflite
  - litert
  - android
  - on-device
  - edge-ai
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

Unofficial LiteRT (`.tflite`) conversions of **MegaDetector V6** (`MDV6-yolov10-c` and
`MDV6-yolov10-e`) for the [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android
app. MegaDetector is made by the Microsoft AI for Good Lab (see [Original work](#original-work)).
These files are not provided or endorsed by Microsoft or the MegaDetector authors.

> **FaunaPulse is in beta.** Tests are ongoing; no large-scale test of these files in the app has been
> carried out yet.
> MegaDetector was trained on camera-trap pictures to find animals, people and vehicles; its
> authors say it works best on large mammals. For insects on flowers, prefer the insect
> detectors of the FaunaPulse collection.
> See [Limitations](#limitations).

## Files

{{FILES_TABLE}}

- **c** (`MDV6-yolov10-c`, 2.3 million parameters, 4.7 MiB): `320` pixels (px) for live use.
  `640` px: try it for smaller or more distant animals.
- **e** (`MDV6-yolov10-e`, 29.5 million parameters, 57 MiB): the authors' larger model.
- `mdv6-yolov10c_256_w8a32.tflite`: the first file of this repository (August 2026), kept for
  older app versions. It does not run on phone GPUs (see [What it is](#what-it-is)).
- `.json`: export manifest of each new file (source weights and their SHA-256, sizes).
- `LICENSE`: the AGPL-3.0 text.

Speed on our three test phones (November 2026), model time only, ms per picture (GPU / CPU):

| File | Xiaomi 11T Pro (Snapdragon 888) | Redmi Note 9S (Snapdragon 720G) | Galaxy M12 (Exynos 850) |
|---|---|---|---|
| c 320 | 5.5 / 10 | 16 / 19 | 60 / 27 |
| c 640 | 14 / 38 | 57 / 70 | 190 / 101 |
| e 320 | 40 / 180 | 214 / 258 | 734 / 445 |
| e 640 | 137 / 795 | 663 / 1,088 | 2,773 / 1,950 |
| c 256 w8a32 | no GPU / 8 | no GPU / 14 | no GPU / 31 |

Measured with the app's engine benchmark (CPU: 4 threads on the Xiaomi, 8 on the others; fp16
files with 16-bit maths); the app adds its own work per frame. On these three phones, c at 640 px
was about 2.5 to 4 times slower than c at 320 px, and e about 7 to 21 times slower than c at the
same input size. Other phones, settings and app versions give other times. Live, the app checks
fewer pictures per second with a slower file; videos analysed afterwards only take longer. The
authors give approximate speeds of c and e on a desktop GPU, and of c on a computer's CPU, in
the [MegaDetector Model Zoo](https://microsoft.github.io/MegaDetector/model_zoo/) (Performance
Benchmarks).

## Original work

- **Made by**: the Microsoft AI for Good Lab. The weights deposit lists Zhongqi Miao, Andrés
  Adolfo Hernández, Luisa Fernanda Vargas, Isai Daniel Chacón, Daniela Ruiz and Carl Chalmers.
  MegaDetector versions 1 to 5 were first developed at Microsoft AI for Earth by Dan Morris and
  colleagues.
- **Code and documentation**: https://github.com/microsoft/MegaDetector (code: MIT). Variants and
  licences: [MegaDetector Model Zoo](https://microsoft.github.io/MegaDetector/model_zoo/). The
  weights are loaded by [PyTorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife).
- **Variants**: `MDV6-yolov10-c` (compact, YOLOv10-N size) and `MDV6-yolov10-e` (extra-size,
  YOLOv10-X size). Three classes: `animal`, `person`, `vehicle`. Both trained at 640 px.
- **Source weights**, from the Zenodo deposit
  [Pytorch-wildlife-model-weights](https://doi.org/10.5281/zenodo.15398270) (version v22; md5 as
  Zenodo lists it):
  - `MDV6-yolov10-c.pt`, md5 `1ecc38fbe462320ea33bf3c57e9e1561` (identical in versions v24 to v27)
  - `MDV6-yolov10-e.pt`, md5 `37ec2a0c959c2f8ef01e19110690eb69`

## What it is

**c and e at 320 and 640 px** (November 2026):

- **Conversion**: Ultralytics 8.4.170 TFLite export, weights stored as fp16. Nothing retrained.
- **Input**: one RGB image `[1, 3, 320, 320]` or `[1, 3, 640, 640]` (NCHW), float32, values 0..1.
- **Output**: `[1, 7, 2100]` (320 px) or `[1, 7, 8400]` (640 px): 4 box values and 3 class
  scores per candidate, without NMS (the app does NMS). This is YOLOv10's one-to-many head,
  trained together with its NMS-free head. The NMS-free head's top-k step does not run on phone
  GPUs; this head does.

**c at 256 px, w8a32** (August 2026):

- **Conversion**: Ultralytics 8.3.235 TFLite export with `int8=True`, `nms=False`, 7 August 2026
  (from the file's metadata). In that version this gives 8-bit weights with 32-bit float
  activations ("w8a32", dynamic-range quantisation), not full INT8. The exact command was not
  logged (early export scripts, not the exact recipe:
  [legacy](https://github.com/valentinitnelav/fauna-pulse-models/tree/736ed0f89d0b9e77bb1e212047fef722e33eb780/legacy)).
- **Input**: `[1, 256, 256, 3]` (NHWC), float32, values 0..1.
- **Output**: `[1, 300, 6]`: up to 300 boxes (4 coordinates, score, class index) from the
  NMS-free head.
- The same bytes shipped inside the app from v0.7.0-alpha.1 and as
  `MDV6-yolov10-c_int8_256.tflite` in app release v0.8.0-alpha.1 (same SHA-256).

**Embedded metadata** (all files): Ultralytics writes class names and input size into each file.
Its `description`, `author` ("Ultralytics") and `license` fields are defaults of the Ultralytics
export tool; they do not name the model's authors.

## Use in FaunaPulse

FaunaPulse works in two steps: a detector finds and boxes each animal, then an identification
model of the FaunaPulse collection (for example BioCLIP 2 with a label pack) names it from the
crop. MegaDetector is the first step: a general detector for animals, people and vehicles, which
leaves naming the species to the second.

In FaunaPulse after v0.8.0-alpha.1, home screen menu, **Download & import models** downloads the
files the app's catalogue lists, from links pinned to one commit of this repository, and checks
their SHA-256. Any file can also be downloaded here and imported by hand (Import model files…).

## How it was made

`detector_export/export_detector.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/detector_export)
(CPU only, Ultralytics 8.4.170, 19 and 20 November 2026):

```bash
python export_detector.py --weights MDV6-yolov10-c.pt --name mdv6-yolov10c --imgsz 320 640 --out out
python export_detector.py --weights MDV6-yolov10-e.pt --name mdv6-yolov10e --imgsz 320 640 --out out
```

## Limitations

- MegaDetector was trained at 640 px; PyTorch-Wildlife runs it at 1280 px. At 320 or 256 px, small or
  distant animals get even smaller. Therefore, The authors' figures (animal recall 76.8 % for c, 82.8 %
  for e) do not apply to these files.
- Conversion and fp16 or especially 8-bit weights can change the outputs.
- It was not trained on phone video of insects on flowers. For insects, see the other models of
  the FaunaPulse collection.
- Results depend on image quality, how large the animals appear, the background, the phone and
  the app's settings (confidence threshold, input size). Choose between files on your own
  footage.

## Licence

AGPL-3.0, the licence the MegaDetector authors give for both variants in the
[MegaDetector Model Zoo](https://microsoft.github.io/MegaDetector/model_zoo/) (the MegaDetector
code is MIT; some other V6 variants are MIT or Apache-2.0). The Zenodo deposit as a whole is
labelled CC BY 4.0; this card follows the authors' licence for these variants. The text is in
`LICENSE`. These files are converted (modified) versions, made with Ultralytics' TFLite export
(see [What it is](#what-it-is)); the export tools are in fauna-pulse-models (AGPL-3.0).

## Citation

If you use these files, please cite MegaDetector as its authors ask
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

Neither paper describes these exact variants (the Pytorch-Wildlife paper introduces V6 with a
YOLOv9 model); they are documented in the MegaDetector Model Zoo and the weights deposit. Also
relevant:

- Weights: Miao, Z., Hernández, A. A., Vargas, L. F., Chacón, I. D., Ruiz, D., & Chalmers, C.
  (2025). Pytorch-wildlife-model-weights (Version v22) [Model weights]. Zenodo.
  https://doi.org/10.5281/zenodo.15398270
- Architecture: Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024).
  YOLOv10: Real-Time End-to-End Object Detection. *Advances in Neural Information Processing
  Systems 37*. https://doi.org/10.52202/079017-3429

## Notes

- Model files here are never replaced: the app pins each download to a commit and checks its
  SHA-256. New weights get a new file name.
- The `.json` manifests, which the app does not download, may be corrected.
