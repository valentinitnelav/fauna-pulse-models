---
license: mit
library_name: litert
pipeline_tag: image-feature-extraction
base_model: imageomics/bioclip-2.5-vith14
base_model_relation: quantized
datasets:
  - imageomics/TreeOfLife-200M
tags:
  - tflite
  - litert
  - android
  - on-device
  - mobile-sensing
  - faunapulse
  - camera-traps
  - biodiversity
  - biodiversity-monitoring
  - taxonomy  
  - bioclip
  - clip
  - zero-shot-image-classification
---

# BioCLIP 2.5 Huge (ViT-H/14) image tower, LiteRT (fp16) for FaunaPulse

An unofficial conversion of the image tower of
[BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14) (ViT-H/14) to a LiteRT
`.tflite` file, plus label packs, made for the Identify screen of the
[FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. BioCLIP 2.5 Huge was
made by Jianyang Gu, Samuel Stevens, Elizabeth G. Campolongo et al., and released by the Imageomics Institute.
See [Original work](#original-work) below. These files are not provided or endorsed by them or by
the Imageomics Institute.

> I did not train BioCLIP 2.5. It is a general model for the whole tree of
> life, not made specifically for FaunaPulse, and it has not yet been tested at scale in
> FaunaPulse, so how accurate it is there is not known yet: treat its answers as suggestions to
> check. It is slow: on the tested phone (7.4 GB memory) it ran on the CPU only, about 5.5 s per
> crop, so it suits analysing pictures afterwards;
> [BioCLIP 2](https://huggingface.co/Valen-Peregrine/bioclip-2-tflite) is the model FaunaPulse
> is meant to use for now. Each label pack here narrows the authors' name list
> (TreeOfLife-200M, 794,878 names) to certain groups of taxa: the BioCLIP software documentation
> notes that limiting the possible names to the taxa that can be expected "may improve the
> accuracy of the predictions"
> ([pybioclip](https://imageomics.github.io/pybioclip/geo-restricted-taxa/)). Packs for other
> groups of species in that list can be built with
> `bioclip_export/build_label_pack.py` of
> [fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models).

## Files

{{FILES_TABLE}}

- `bioclip-2.5_224_fp16.tflite`: the image tower (1.27 GB). The phone compares its embedding of
  each crop with the name embeddings of a label pack.
- `.fpack`: label packs. Each row is a species name with its taxonomy and, where one exists, a
  common name (English where available, otherwise another language), and the BioCLIP 2.5 text
  embedding of that name, pre-computed by the BioCLIP authors (TreeOfLife-200M); stored here as
  float16, re-normalised, a subset of their 794,878 rows. Six "none of these" rows were added for
  FaunaPulse, embedded with the BioCLIP 2.5 text tower (flower, leaf, shadow, debris, blurry,
  spider web).
  - `bioclip-2.5_pollinator-orders-europe_v1.fpack`: flies, bees and wasps, beetles, butterflies
    and moths (Diptera, Hymenoptera, Coleoptera, Lepidoptera) with at least 3 GBIF records in
    Europe (34,704 names; GBIF queried 20 September 2026).
  - `bioclip-2.5_flower-visitors-32fam_v1.fpack`: 32 families of common flower visitors,
    worldwide, including 3 spider families (37,461 names).
  - Inside each pack, its `pack_id` keeps the name it was built with (for example
    `bioclip25_pollinator_orders_europe_v1`); the file names follow the FaunaPulse naming rule.
- `.json`: the export manifest. Its `"file"` field gives the name before upload
  (`bioclip-25_image_fp16.tflite`); same SHA-256.
- `LICENSE`: the MIT licence of BioCLIP, copied from the authors' code repository.

## Original work

- **Authors**: Jianyang Gu, Samuel Stevens, Elizabeth G. Campolongo, Matthew J. Thompson, Net
  Zhang, Jiaman Wu, Andrei Kopanev, Zheda Mai, Alexander E. White, James Balhoff, Wasila M.
  Dahdul, Daniel Rubenstein, Hilmar Lapp, Tanya Berger-Wolf, Wei-Lun Chao and Yu Su; released by
  the Imageomics Institute (version 1.0.0, February 2026).
- **Model**: https://huggingface.co/imageomics/bioclip-2.5-vith14 (MIT). **Code**:
  https://github.com/Imageomics/bioclip-2. There is no separate BioCLIP 2.5 paper; the authors
  ask for the BioCLIP 2 paper to be cited with the model.
- **Training**: fine-tuned by its authors from the CLIP ViT-H/14 model that LAION trained with
  OpenCLIP on LAION-2B
  ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)),
  on an updated
  TreeOfLife-200M (pictures of organisms from GBIF, EOL, BIOSCAN-5M and FathomNet) plus 26
  million LAION-2B samples.
- **Source files**, the authors' own files on Hugging Face: `open_clip_model.safetensors` of
  imageomics/bioclip-2.5-vith14 at commit 6e3d04e (SHA-256
  `ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97`), used unchanged apart from
  the cast to fp16, and the name embeddings `embeddings/txt_emb_bioclip-2.5-vith14.npy` (SHA-256
  `d1cc734330d17ea26e6f713b289b2138b4adc90d4f524349cd16fab42d5c3358`) and `.json` (SHA-256
  `af0cb41ffbfb31e6a2e2d5e3a402529ec8245a4268f42ce45ee4e977b7127443`) of
  imageomics/TreeOfLife-200M at commit 5f2dc49 (the same files as in the cited revision
  94bbc0b). The files used here are byte-identical to these (same SHA-256, checked against a
  download from the authors' repositories on 16 October 2026). Weights and packs were also
  spot-checked by content: several weight tensors were found byte for byte in the `.tflite`,
  every pack name occurs in the authors' name list, and sampled embedding values match theirs
  to within fp16 rounding.
- **Europe list**: species counts from GBIF.org occurrence data, built with the method and the
  TreeOfLife-to-GBIF key table of Maximilian Sittinger's `insect-detect-post` (re-implemented; the table
  is not redistributed).

## What it is

- **Input**: one RGB crop `[1, 3, 224, 224]` (NCHW), float32, values 0..1 (the colour
  normalisation is inside the file).
- **Output**: an L2-normalised 1024-number embedding (logit scale 98.86).
- **Conversion**: image tower only (no text tower), the weights of the large layers in fp16
  (small tensors such as biases stay float32), attention with tensors of at most four
  dimensions. Nothing was retrained.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888, 7.4 GB memory): runs on the CPU only, 5.3 to 5.6 s
  per crop.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads the model and a label pack
from links pinned to one commit of this repository and checks each file's SHA-256 (FaunaPulse
versions after v0.8.0-alpha.1; with any version you can also download the files here and import
them by hand). Download on Wi-Fi (1.27 GB for the model, plus about 75 to 80 MB for a label
pack).

## How it was made

`bioclip_export/` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/bioclip_export)
(CPU only, about 11 GB memory): `export_image_tower.py --model bioclip-2.5` for the image tower,
`build_label_pack.py --model bioclip-2.5` for the label packs, `verify_parity.py` for the
PyTorch vs `.tflite` comparison.

## Limitations

- For bias and limits the authors refer to the BioCLIP 2 card: its training data is long-tailed
  across taxa, so answers may lean towards well-represented species (see the
  [BioCLIP 2 model card](https://huggingface.co/imageomics/bioclip-2#bias-risks-and-limitations)).
- The phone can only answer with names that are in the chosen label pack. The "none of these"
  rows are a FaunaPulse addition that BioCLIP 2.5 was not designed with.
- Names and taxonomy are as published in TreeOfLife-200M, quirks included: the 32-family pack
  has 81 rows with kingdom "Metazoa" instead of "Animalia" and 105 whose species field is a code,
  an author name or a year (for example *Atherigona* "GRG-2020"); the Europe pack has none.
- This file has not yet been tested at scale in FaunaPulse, so how accurate it is there is not
  known yet. The app can run it, but the results depend on many factors: the quality of each
  crop (focus, motion blur, light, resolution, video compression), how large and how complete
  the organism is in the crop (distance, zoom, the detector's box), the label pack chosen (a
  species that is not in the pack cannot be named), the phone and its camera, and the app's
  settings. Try it on your own footage.

## Licence

- **Image tower**: a converted part (the image tower) of BioCLIP 2.5 Huge, under its MIT licence. The copyright and
  permission notice is in `LICENSE`, a verbatim copy of
  https://github.com/Imageomics/bioclip-2/blob/main/LICENSE (the model repository names MIT but
  has no licence file of its own).
- **Label packs**: they follow the licence of their source. The names, taxonomy and embeddings
  come from TreeOfLife-200M, which the BioCLIP authors release as a compilation under CC0 1.0
  (see its dataset card, "Licensing Information"). The common names in it come from the GBIF
  Backbone Taxonomy (CC BY 4.0, https://doi.org/10.15468/39omei), so please credit GBIF when you
  reuse them. The six "none of these" rows (short descriptions such as a flower or a leaf,
  embedded with the BioCLIP 2.5 text tower) were added for FaunaPulse on the same CC0 1.0 terms.

## Citation

If you use these files, please cite the model and the BioCLIP 2 paper, as the model card asks:

- Gu, J., Stevens, S., Campolongo, E. G., Thompson, M. J., Zhang, N., Wu, J., Kopanev, A., Mai,
  Z., White, A. E., Balhoff, J., Dahdul, W. M., Rubenstein, D., Lapp, H., Berger-Wolf, T., Chao,
  W.-L., & Su, Y. (2026). BioCLIP 2.5 Huge (Version 1.0.0) [Model]. Hugging Face.
  https://doi.org/10.57967/hf/10131
- Gu, J., Stevens, S., Campolongo, E., Thompson, M., Zhang, N., Wu, J., Kopanev, A., Mai, Z.,
  White, A., Balhoff, J., Dahdul, W., Rubenstein, D., Lapp, H., Berger-Wolf, T., Chao, W.-L., &
  Su, Y. (2025). BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning.
  *Advances in Neural Information Processing Systems 38* (NeurIPS 2025).
  https://doi.org/10.52202/085713-3436 (preprint: https://arxiv.org/abs/2505.23883)

The BibTeX for both, and for OpenCLIP and the original BioCLIP that the authors also suggest
citing, is in the [Citation section of the model card](https://huggingface.co/imageomics/bioclip-2.5-vith14#citation).

For the label packs:

- Gu, J., Stevens, S., Campolongo, E. G., et al. (2026). TreeOfLife-200M (Revision 94bbc0b)
  [Data set]. Hugging Face. https://doi.org/10.57967/hf/8980
- GBIF Secretariat (2023). GBIF Backbone Taxonomy. Checklist dataset.
  https://doi.org/10.15468/39omei (the common names, as included in TreeOfLife-200M)
- Europe list method: Sittinger, M. (2026). Software for post-processing of data captured with
  the Insect Detect camera trap [Software]. Zenodo. https://doi.org/10.5281/zenodo.21822140

## Notes

- Model files and label packs in this repository are never replaced: the app pins each
  download to a commit and checks its SHA-256. New weights or packs get a new file name.
- The `.json` manifest, which the app does not download, may be corrected.
- BioCLIP 2.5 packs work only with BioCLIP 2.5 (1024 numbers per name), not with BioCLIP 2.
