---
license: mit
library_name: litert
pipeline_tag: zero-shot-image-classification
base_model: imageomics/bioclip-2
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

# BioCLIP 2 image tower, LiteRT (fp16, GPU-friendly) for FaunaPulse

An unofficial conversion of the image tower of [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)
(ViT-L/14) to a LiteRT `.tflite` file, plus label packs, made for the Identify screen of the
[FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. Not provided or
endorsed by Imageomics.

## Files

{{FILES_TABLE}}

- `.tflite`: the image tower. The phone compares its embedding of each crop with the label
  packs' name embeddings.
- `.fpack`: label packs, the names the phone can answer with: pre-computed BioCLIP 2 text
  embeddings of species names, their taxonomy, and six "none of these" rows (flower, leaf,
  shadow, debris, blurry, web; camera-trap prompts for the mammals and birds pack).
  - `bioclip2_pollinator_orders_europe_v1`: flies, bees and wasps, beetles, butterflies and
    moths of Europe.
  - `bioclip2_flower_visitors_32fam_v1`: 32 families of common flower visitors, worldwide
    (38,570 names).
  - `bioclip-2_mammals-birds-world_v1`: mammals and birds of the world (17,130 names).
- `.json`: the export manifest. Its `"file"` field names the file as it was before upload
  (`bioclip-2_image_fp16.tflite`); the SHA-256 is the same.

## What it is

- **Input**: one RGB crop, 224 x 224, NCHW, values 0..1 (the colour normalisation is inside
  the graph). **Output**: an L2-normalised 768-dimensional embedding (logit scale 100).
- **Conversion**: fp16 weights; the attention is written with tensors of at most four
  dimensions so that the phone's GPU delegate can run it ("4d"; same arithmetic).
- **Phone** (Xiaomi 11T Pro, Snapdragon 888): 0.27 s per crop on the GPU, 2.6 s on the CPU.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads the model and a label pack
from links pinned to one commit of this repository and checks each file's SHA-256.

## How it was made

`bioclip_export/` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/bioclip_export)
(CPU only): `export_image_tower.py --model bioclip-2 --precision fp16` for the image tower,
`build_label_pack.py` for the label packs (from the TreeOfLife-200M species embeddings),
`verify_parity.py` for the PyTorch vs `.tflite` comparison.

## Original work, licence and citation

- Model: https://huggingface.co/imageomics/bioclip-2 (MIT). Paper: Gu, J., Stevens, S.,
  Campolongo, E. G., et al. (2025). *BioCLIP 2: Emergent Properties from Scaling Hierarchical
  Contrastive Learning.* NeurIPS 2025. https://arxiv.org/abs/2505.23883. Also consider citing
  the original BioCLIP (Stevens et al., CVPR 2024) and OpenCLIP, as the model card suggests.
- Label packs: derived from the species text embeddings of
  [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) (CC0-1.0).

The image tower follows the licence of the original (MIT); the label packs follow CC0-1.0.

## Notes

- Files in this repository are never replaced: the app pins each download to a commit.
  New weights or packs get a new file name.
