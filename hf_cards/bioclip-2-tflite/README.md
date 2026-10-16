---
license: mit
library_name: litert
pipeline_tag: image-feature-extraction
base_model: imageomics/bioclip-2
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

# BioCLIP 2 image tower, LiteRT (fp16, GPU-friendly) for FaunaPulse

An unofficial conversion of the image tower of [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)
(ViT-L/14) to a LiteRT `.tflite` file, plus label packs, made for the Identify screen of the
[FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. BioCLIP 2 was made by
Jianyang Gu, Samuel Stevens, Elizabeth G. Campolongo et al. and released by the Imageomics Institute.
See [Original work](#original-work) below. These files are not provided or endorsed by them or by the
Imageomics Institute.

> I did not train BioCLIP 2. It is a general model for the whole tree of
> life, not tuned for insects on flowers or for small crops from phone video, and how accurate
> it is in FaunaPulse is not known yet: treat its answers as suggestions to check. It is the
> identification model FaunaPulse is meant to use for now, with label packs (the names it may
> answer with). The packs here are mine, cut from the authors' name list; you can build your own
> for any list of species that are in that list (TreeOfLife-200M, 867,455 names) with
> `bioclip_export/build_label_pack.py` of
> [fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models). I may build more
> packs on request, but I cannot promise to.

## Files

{{FILES_TABLE}}

- `bioclip-2_224_fp16.tflite`: the image tower. The phone compares its embedding of each crop
  with the name embeddings of a label pack.
- `.fpack`: label packs. Each row is a species name with its taxonomy and, where one exists, a
  common name (English where available, otherwise another language), and the BioCLIP 2 text
  embedding of that name, pre-computed by the BioCLIP 2 authors (TreeOfLife-200M); stored here as
  float16, re-normalised, a subset of their 867,455 rows. Six "none of these" rows were added
  for FaunaPulse, embedded with the BioCLIP 2 text tower (arthropod packs: flower, leaf, shadow,
  debris, blurry, spider web; mammals and birds pack: camera-trap scenes such as bare ground,
  vegetation, a person or a vehicle).
  - `bioclip-2_pollinator-orders-europe_v1.fpack`: flies, bees and wasps, beetles, butterflies
    and moths (Diptera, Hymenoptera, Coleoptera, Lepidoptera) with at least 3 GBIF records in
    Europe (35,264 names; GBIF queried 20 September 2026).
  - `bioclip-2_flower-visitors-32fam_v1.fpack`: 32 families of common flower visitors,
    worldwide, including 3 spider families (38,570 names).
  - `bioclip-2_mammals-birds-world_v1.fpack`: mammals and birds of the world (17,130 names).
  - Inside each pack, its `pack_id` keeps the name it was built with (for example
    `bioclip2_pollinator_orders_europe_v1`); the file names follow the FaunaPulse naming rule.
- `.json`: the export manifest. Its `"file"` field gives the name before upload
  (`bioclip-2_image_fp16.tflite`); same SHA-256.
- `LICENSE`: the MIT licence of BioCLIP 2, copied from the authors' code repository.

## Original work

- **Authors**: Jianyang Gu, Samuel Stevens, Elizabeth G. Campolongo, Matthew J. Thompson, Net
  Zhang, Jiaman Wu, Andrei Kopanev, Zheda Mai, Alexander E. White, James Balhoff, Wasila M.
  Dahdul, Daniel Rubenstein, Hilmar Lapp, Tanya Berger-Wolf, Wei-Lun Chao and Yu Su; released by
  the Imageomics Institute.
- **Model**: https://huggingface.co/imageomics/bioclip-2 (MIT). **Code**:
  https://github.com/Imageomics/bioclip-2. **Paper**: Gu et al. (2025), *BioCLIP 2: Emergent
  Properties from Scaling Hierarchical Contrastive Learning*, NeurIPS 2025,
  https://doi.org/10.52202/085713-3436.
- **Training**: fine-tuned by its authors from the CLIP ViT-L/14 model that LAION trained with
  OpenCLIP on LAION-2B
  ([laion/CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K)),
  on TreeOfLife-200M, over 200 million pictures of organisms from GBIF, EOL, BIOSCAN-5M and
  FathomNet.
- **Source files**, downloaded directly from the authors' Hugging Face repositories:
  `open_clip_model.safetensors` of imageomics/bioclip-2 at commit 2957b32 (SHA-256
  `b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef`), and the name embeddings
  `embeddings/txt_emb_species.npy` (SHA-256
  `c72442de7b0cb7fcb55ab7ca08099d0f42fbd6769efe16ca64c1daa7a8b87db2`) and
  `embeddings/txt_emb_species.json` (SHA-256
  `4648928b006f85d83d28e5a27074ca9363465d82e778d708b369c5eaf54b8ef5`) of
  imageomics/TreeOfLife-200M at commit 5f2dc49 (the same files as in revision a8f38b4, on which
  BioCLIP 2 was trained).
- **Europe list**: species counts from GBIF.org occurrence data, built with the method and the
  TreeOfLife-to-GBIF key table of Maximilian Sittinger's `insect-detect-post` (re-implemented; the table
  is not redistributed).

## What it is

- **Input**: one RGB crop `[1, 3, 224, 224]` (NCHW), float32, values 0..1 (the colour
  normalisation is inside the file).
- **Output**: an L2-normalised 768-number embedding (logit scale 100).
- **Conversion**: image tower only (no text tower), the weights of the large layers in fp16
  (small tensors such as biases stay float32); the attention is written with
  tensors of at most four dimensions so that phone GPUs can run it (same arithmetic). Nothing was
  retrained.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888): 0.27 s per crop on the GPU, 2.6 s on the CPU.

## Use in FaunaPulse

Home screen menu, **Download & import models**: the app downloads the model and a label pack
from links pinned to one commit of this repository and checks each file's SHA-256 (FaunaPulse
versions after v0.8.0-alpha.1; with any version you can also download the files here and import
them by hand). Download on Wi-Fi (610 MB for the model, plus 28 to 63 MB for a label pack).

## How it was made

`bioclip_export/` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/bioclip_export)
(CPU only): `export_image_tower.py --model bioclip-2 --precision fp16` for the image tower,
`build_label_pack.py` for the label packs (with `build_region_species_list.py` for the Europe
list), `verify_parity.py` for the PyTorch vs `.tflite` comparison.

## Limitations

- BioCLIP 2 was trained on TreeOfLife-200M, which is long-tailed across taxa, so its answers may
  lean towards well-represented species (see the
  [BioCLIP 2 model card](https://huggingface.co/imageomics/bioclip-2#bias-risks-and-limitations)).
- The phone can only answer with names that are in the chosen label pack. The "none of these"
  rows are a FaunaPulse addition that BioCLIP 2 was not designed with.
- Names and taxonomy are as published in TreeOfLife-200M, quirks included: in these packs some
  rows have kingdom "Metazoa" instead of "Animalia" (36 to 85 per pack), and some have a
  species field that holds another name (a synonym), a code or a year (111 to 613 per pack).

## Licence

- **Image tower**: a converted part of BioCLIP 2, under its MIT licence. The copyright and
  permission notice is in `LICENSE`, a verbatim copy of
  https://github.com/Imageomics/bioclip-2/blob/main/LICENSE (the model repository names MIT but
  has no licence file of its own).
- **Label packs**: the names, taxonomy and embeddings come from TreeOfLife-200M, which its
  authors release as a compilation under CC0 1.0. The common names in it come from the GBIF
  Backbone Taxonomy (CC BY 4.0), so please credit GBIF when you reuse them. The six "none of
  these" rows are my own and released under CC0 1.0.

## Citation

If you use these files, please cite BioCLIP 2 as its model card asks, the model and the paper:

- Gu, J., Stevens, S., Campolongo, E. G., Thompson, M. J., Zhang, N., Wu, J., Kopanev, A., Mai,
  Z., White, A. E., Balhoff, J., Dahdul, W. M., Rubenstein, D., Lapp, H., Berger-Wolf, T., Chao,
  W.-L., & Su, Y. (2025). BioCLIP 2 (Version 1.0.0) [Model]. Hugging Face.
  https://doi.org/10.57967/hf/5765
- Gu, J., Stevens, S., Campolongo, E., Thompson, M., Zhang, N., Wu, J., Kopanev, A., Mai, Z.,
  White, A., Balhoff, J., Dahdul, W., Rubenstein, D., Lapp, H., Berger-Wolf, T., Chao, W.-L., &
  Su, Y. (2025). BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning.
  *Advances in Neural Information Processing Systems 38* (NeurIPS 2025).
  https://doi.org/10.52202/085713-3436 (preprint: https://arxiv.org/abs/2505.23883)

The BibTeX for both, and for OpenCLIP and the original BioCLIP that the authors also suggest
citing, is in the [Citation section of the BioCLIP 2 model card](https://huggingface.co/imageomics/bioclip-2#citation).

For the label packs:

- Gu, J., Stevens, S., Campolongo, E. G., et al. (2026). TreeOfLife-200M (Revision 94bbc0b)
  [Data set]. Hugging Face. https://doi.org/10.57967/hf/8980 (the embeddings used are identical
  in revision a8f38b4, https://doi.org/10.57967/hf/6786)
- GBIF Secretariat (2023). GBIF Backbone Taxonomy. Checklist dataset.
  https://doi.org/10.15468/39omei (the common names, as included in TreeOfLife-200M)
- Europe list method: Sittinger, M. (2026). Software for post-processing of data captured with
  the Insect Detect camera trap [Software]. Zenodo.
  https://doi.org/10.5281/zenodo.21822140

## Notes

- Files in this repository are never replaced: the app pins each download to a commit and
  checks its SHA-256. New weights or packs get a new file name.
- BioCLIP 2 packs work only with BioCLIP 2 (768 numbers per name), not with BioCLIP 2.5.
