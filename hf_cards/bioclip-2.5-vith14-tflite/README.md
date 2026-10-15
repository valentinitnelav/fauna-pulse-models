---
license: mit
library_name: litert
pipeline_tag: zero-shot-image-classification
base_model: imageomics/bioclip-2.5-vith14
base_model_relation: quantized
datasets:
  - imageomics/TreeOfLife-200M
tags:
  - tflite
  - litert
  - android
  - on-device
  - faunapulse
  - bioclip
  - taxonomy
  - biodiversity
---

# BioCLIP 2.5 (ViT-H/14) image tower, LiteRT (fp16) for FaunaPulse

An unofficial conversion of the image tower of
[BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14) (ViT-H/14) to a
LiteRT `.tflite` file, plus label packs, made for the Identify screen of the
[FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. Not provided or
endorsed by Imageomics.

## Files

{{FILES_TABLE}}

- `.tflite`: the image tower (1.27 GB). The phone compares its embedding of each crop with
  the label packs' name embeddings.
- `.fpack`: label packs, the names the phone can answer with: pre-computed BioCLIP 2.5 text
  embeddings of species names, their taxonomy, and six "none of these" rows.
  - `bioclip25_pollinator_orders_europe_v1`: flies, bees and wasps, beetles, butterflies and
    moths of Europe.
  - `bioclip25_flower_visitors_32fam_v1`: 32 families of common flower visitors, worldwide
    (37,461 names).
- `.json`: the export manifest.

## What it is

- **Input**: one RGB crop, 224 x 224, NCHW, values 0..1 (the colour normalisation is inside
  the graph). **Output**: an L2-normalised 1024-dimensional embedding (logit scale 98.86).
- **Conversion**: fp16 weights, attention with tensors of at most four dimensions.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888, 7.4 GB memory): runs on the CPU only, 5.3 to
  5.6 s per crop. Slow: meant for analysing pictures afterwards, not live. BioCLIP 2 (in the
  same collection) is the fast choice on phones whose GPU can run it.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads the model and a label pack
from links pinned to one commit of this repository and checks each file's SHA-256. Download
on Wi-Fi (1.27 GB).

## How it was made

`bioclip_export/` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/bioclip_export)
(CPU only, about 11 GB RAM): `export_image_tower.py --model bioclip-2.5` for the image tower,
`build_label_pack.py --model bioclip-2.5` for the label packs (from the BioCLIP 2.5 text
embeddings in TreeOfLife-200M), `verify_parity.py` for the PyTorch vs `.tflite` comparison.

## Original work, licence and citation

- Model: https://huggingface.co/imageomics/bioclip-2.5-vith14 (MIT). Please cite it as its
  model card asks, together with BioCLIP 2: Gu, J., Stevens, S., Campolongo, E. G., et al.
  (2025). *BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning.*
  NeurIPS 2025. https://arxiv.org/abs/2505.23883.
- Label packs: derived from the BioCLIP 2.5 text embeddings of
  [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) (CC0-1.0).

The image tower follows the licence of the original (MIT); the label packs follow CC0-1.0.

## Notes

- Files in this repository are never replaced: the app pins each download to a commit.
  New weights or packs get a new file name.
