---
license: gpl-3.0
library_name: litert
pipeline_tag: image-classification
tags:
  - tflite
  - litert
  - android
  - on-device
  - faunapulse
  - insects
  - taxonomy
  - hierarchical-classification
---

# insectDCT hierarchical classifier V7 (EfficientNetV2-S), LiteRT (fp16) for FaunaPulse

An unofficial conversion of the **insectDCT hierarchical classifier V7**, EfficientNetV2-S
network, to a LiteRT `.tflite` file plus its class list, made for the Identify screen of the
[FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. Not provided or
endorsed by the insectDCT authors.

## Files

{{FILES_TABLE}}

- `.tflite`: the network. `.fpack`: its class list (the classes of the three levels, 19 + 41 + 104, with
  their taxonomy), which the app needs next to the model. `.json`: the export manifest.

## What it is

- **Original model**: insectDCT hierarchical classifier V7, EfficientNetV2-S
  (`HierarchicalClassifier_EFF2S_V7.pth`), weights as packaged by the
  [InsectAI Model Zoo](https://github.com/InsectAI-COST-Action/insect-model-zoo).
- **Input**: one RGB crop, 224 x 224, NCHW, values 0..1 (no further normalisation, as
  insectDCT's own `ToTensor()`).
- **Output**: the raw scores of the three heads, one after the other (level 1: 19 classes,
  level 2: 41, level 3: 104; output length 164); not normalised. The class list's taxonomy uses the TreeOfLife
  names that come with BioCLIP 2.5 (Imageomics).

## Checks

- **Agreement with insectDCT's own code** (same model, not an accuracy test): on 34 crops,
  34 of 34 same answers, scores within 0.03.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888): 0.02 s per crop on the GPU, 0.13 s on the CPU
  (2 threads); GPU and CPU agree (cosine 0.9998).
- The authors' published per-class test results (insectDCT repository, `metrics/`) put
  EfficientNetV2-S ahead of ResNet50 at every level; ConvNeXt-Base scores higher but does not
  run correctly on the tested phone GPU.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads the model and its class
list from links pinned to one commit of this repository and checks each file's SHA-256.

## How it was made

`classifier_export/export_insectdct_cls.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/classifier_export)
(CPU only):

```bash
python export_insectdct_cls.py --zoo-dir /path/to/weights/insectdct-cls-v7 --backbone eff2s \
    --check-images /path/to/check_crops
```

## Original work, licence and citation

insectDCT: Bjerge, Wogram, Serra-Marin, Sakhiashvili & Høye (2026). InsectDCT: A generalized
pipeline for detection, taxonomic classification, and tracking of insects in camera-trap
recordings. bioRxiv. https://doi.org/10.64898/2026.07.07.736939.
Code: https://github.com/kimbjerge/insectDCT (GPL-3.0).

This file follows the licence of the original (GPL-3.0). Please cite the original paper when
you use this model.

## Notes

- Files in this repository are never replaced: the app pins each download to a commit.
  New weights get a new file name.
