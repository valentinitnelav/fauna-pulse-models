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
> life, not made specifically for FaunaPulse, and it has not yet been tested at scale in
> FaunaPulse, so how accurate it is there is not known yet: treat its answers as suggestions to
> check. It is the identification model FaunaPulse is meant to use for now, with label packs
> (the names it may answer with). Each pack here narrows the authors' name list
> (TreeOfLife-200M, 867,455 names) to certain groups of taxa: the BioCLIP software documentation
> notes that limiting the possible names to the taxa that can be expected "may improve the
> accuracy of the predictions"
> ([pybioclip](https://imageomics.github.io/pybioclip/geo-restricted-taxa/)). Packs for other
> groups of species in that list can be built with
> `bioclip_export/build_label_pack.py` of
> [fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models).

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
  - `bioclip-2_pollinator-orders-europe_v2.fpack`: flies, bees and wasps, beetles, butterflies
    and moths (Diptera, Hymenoptera, Coleoptera, Lepidoptera) with at least 3 GBIF records in
    Europe, counted per accepted species of the Catalogue of Life (34,978 names; GBIF queried
    26 October 2026).
  - `bioclip-2_mammals-birds-world_v2.fpack`: mammals and birds of the world (17,130 names).
  - The v2 packs use the names as described in [Names](#names). The v1 packs (same selections;
    the v1 Europe list was made on 20 September 2026 with 35,264 names and an earlier version of
    the same method) use every row as published. They stay here unchanged because installed
    apps pin them.
  - `bioclip-2_flower-visitors-32fam_v1.fpack` (38,570 names) is a **demo, not maintained**: it
    shows that a pack can be built from a list of taxa given by the user. It holds every species,
    worldwide, of 32 hand-picked families. It is not a justified selection of flower visitors:
    of its insects in the four main orders only 14% have 3 or more GBIF records in Europe, and it
    leaves out flower visitors such as the bee family Melittidae, longhorn beetles and most
    moths. For Europe, use the Europe pack.
  - Inside each v1 pack, its `pack_id` keeps the name it was built with (for example
    `bioclip2_pollinator_orders_europe_v1`); a v2 pack's `pack_id` is its file name. The file
    names follow the FaunaPulse naming rule.
- `.json`: the export manifest. Its `"file"` field gives the name before upload
  (`bioclip-2_image_fp16.tflite`); same SHA-256.
- `LICENSE`: the MIT licence of BioCLIP 2, copied from the authors' code repository.

## Names

The names and their taxonomy are those of TreeOfLife-200M, as the BioCLIP 2 authors publish
them: no name is corrected, added or removed. A few rows are not in the plain form "genus,
epithet". In the v2 packs the app uses such a row only down to the rank where it is still plain:

| Row | Example (row of `txt_emb_species.json`) | Used down to |
|---|---|---|
| kingdom "Metazoa", which the dataset card lists among the unresolved kingdoms ("`Metazoa` instead of `Animalia`"); read as Animalia (NCBI Taxonomy lists Animalia as a synonym of Metazoa) | 515547: Curculionidae, Cionus, "thapsus (Fabricius, J.C., 1792)" | family |
| the species field starts with another genus than the genus field | 53564: Coccinellidae, Ceratomegilla, "Semiadalia rufocincta" | family |
| the species field is not one plain epithet (authors, years, codes, subspecies, placeholders such as "spec") | 347389: Parulidae, Basileuterus, "tristriatus sanlucasensis" | genus |

Rows per pack (Metazoa / two genera / species field): Europe 37 / 72 / 0, mammals and birds
41 / 571 / 1. Each pack lists these rows in its header (`name_rules.rows`: the row of the source
file and its names as published), so every change can be checked and undone;
`source.names_file_sha256` names the source file. The rules look only at the form of a row, so a
plain row keeps whatever placement the source gives it, and a row with an empty order or family
(343 rows of the mammals and birds pack) counts only down to the rank above the gap.

To cite the names of a v2 pack: "TreeOfLife-200M (https://doi.org/10.57967/hf/8980), names file
`embeddings/txt_emb_species.json` (SHA-256 4648928b...), names used as published. The dataset
card lists kingdom Metazoa as unresolved; it is read as Animalia (NCBI Taxonomy lists Animalia as
a synonym). Rows not in the plain form are used only down to genus or family, as listed in the
pack header."

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
  is not redistributed). v2: records counted per accepted species of the Catalogue of Life
  (release 2026-06-19 XR, as GBIF interprets them), with the table of `insect-detect-post` v1.1.0,
  made from the same names file. v1: the earlier table (GBIF Backbone keys).

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
- Names and taxonomy are as published in TreeOfLife-200M, so a name may be outdated or a
  synonym. The v1 packs also use the rows described in [Names](#names) as published (for
  example the species answer "Bombus spec" in the 32-family demo pack); the v2 packs use them
  only down to genus or family.
- This file has not yet been tested at scale in FaunaPulse, so how accurate it is there is not
  known yet. The app can run it, but the results depend on many factors: the quality of each
  crop (focus, motion blur, light, resolution, video compression), how large and how complete
  the organism is in the crop (distance, zoom, the detector's box), the label pack chosen (a
  species that is not in the pack cannot be named), the phone and its camera, and the app's
  settings. Try it on your own footage.

## Licence

- **Image tower**: a converted part of BioCLIP 2, under its MIT licence. The copyright and
  permission notice is in `LICENSE`, a verbatim copy of
  https://github.com/Imageomics/bioclip-2/blob/main/LICENSE (the model repository names MIT but
  has no licence file of its own).
- **Label packs**: they follow the licence of their source. The names, taxonomy and embeddings
  come from TreeOfLife-200M, which the BioCLIP authors release as a compilation under CC0 1.0
  (see its dataset card, "Licensing Information"). The common names in it come from the GBIF
  Backbone Taxonomy (CC BY 4.0, https://doi.org/10.15468/39omei), so please credit GBIF when you
  reuse them. The six "none of these" rows of each pack (short descriptions such as a leaf or
  bare ground, embedded with the BioCLIP 2 text tower) were added for FaunaPulse on the same
  CC0 1.0 terms.

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
  https://doi.org/10.5281/zenodo.21822140 (all versions; v2 lists: v1.1.0,
  https://doi.org/10.5281/zenodo.23404778)
- Bánki, O., Roskov, Y., Döring, M., et al. (2026). Catalogue of Life (Version 2026-06-19 XR).
  Catalogue of Life Foundation. https://doi.org/10.48580/dgy8b (the species counting of the v2
  Europe list)

## Notes

- Model files and label packs in this repository are never replaced: the app pins each
  download to a commit and checks its SHA-256. New weights or packs get a new file name.
- The `.json` manifest, which the app does not download, may be corrected.
- BioCLIP 2 packs work only with BioCLIP 2 (768 numbers per name), not with BioCLIP 2.5.
