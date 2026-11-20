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
  - edge-ai
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

> BioCLIP 2.5 is a general model for the whole tree of life, not made specifically for FaunaPulse.
> Treat its answers as suggestions to check. On our test phone (7.4 GB memory) it ran on the CPU
> only, about 5.5 s per crop, so there it suited analysing pictures afterwards;
> [BioCLIP 2](https://huggingface.co/Valen-Peregrine/bioclip-2-tflite) is the identification model
> FaunaPulse is meant to use for now. Each label pack (name list) here holds the taxonomic names
> BioCLIP 2.5 may answer with, and narrows the authors' name list (TreeOfLife-200M, 794,878 names)
> to certain groups of taxa. The BioCLIP software documentation notes that limiting
> the possible names to the taxa that can be expected "may improve the accuracy of the predictions"
> ([pybioclip](https://imageomics.github.io/pybioclip/geo-restricted-taxa/)).
> Packs for other groups of species in that list can be built with
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
  **Current packs** (use these):
  - `bioclip-2.5_pollinator-orders-europe_v3.fpack`, `bioclip-2.5_pollinator-orders-north-america_v1.fpack`,
    `bioclip-2.5_pollinator-orders-south-america_v1.fpack`, `bioclip-2.5_pollinator-orders-africa_v1.fpack`,
    `bioclip-2.5_pollinator-orders-asia_v1.fpack`, `bioclip-2.5_pollinator-orders-oceania_v1.fpack`:
    flies, bees and wasps, beetles, butterflies and moths (Diptera, Hymenoptera, Coleoptera,
    Lepidoptera) with at least 3 GBIF presence records in the continent (GBIF's continents; North
    America includes Central America and the Caribbean), fossil specimens and records of absence
    left out, counted per accepted species of the Catalogue of Life (35,021 / 56,401 / 23,122 /
    21,765 / 37,202 / 22,995 names, counted as in the app: rows without the six "none of these" rows; GBIF queried 17 November 2026). The lists are made from the
    BioCLIP 2.5 name list itself, so unlike v2 no Europe names are lost to differences between
    the BioCLIP 2 and 2.5 lists. Each holds the GBIF species of every name (Catalogue of Life key
    and accepted name) and, per country, the names with at least 3 presence records there, with
    each country's neighbours (a shared land border, or at most 100 km apart over the sea, plus
    two pairs added by hand), so an app can narrow the names to the country where a session was
    recorded. Species that the Catalogue of Life marks as extinct and of which GBIF holds no
    observation of a living individual are left out (Europe: 116 names).
    **Why one pack per continent:** only for size on the phone. FaunaPulse keeps and compares
    every name of the chosen pack for each crop, so memory and time grow with the number of
    names. A world pack of these four orders (about 189,000 names) is too large for the app on
    an ordinary phone today; a continent pack (22,000 to 56,000 names) is not. The continents
    are GBIF's, a practical cut, not a biogeographic claim: species do not follow them, and a
    country near a continent border can have part of its species only in the neighbouring
    continent's pack. Use the pack that holds the most species of your study area; a single
    world pack is planned once the app can load only the names of the chosen countries.

  **Older versions** (kept unchanged because installed apps pin them; not for new work):
  - `bioclip-2.5_pollinator-orders-europe_v2.fpack`: replaced by Europe v3. The four orders with
    at least 3 GBIF records in Europe (34,392 names; GBIF queried 26 October 2026), matched to the
    BioCLIP 2 name list, so 586 names that the BioCLIP 2.5 list has no row for or writes
    differently are missing; no GBIF species and no country lists.
  - `bioclip-2.5_pollinator-orders-europe_v1.fpack` (list made on 20 September 2026, 34,704
    names, with an earlier version of the same method).
  - The v2 and later packs use the names as described in [Names](#names); the v1 packs use every
    row as published. Inside each v1 pack, its `pack_id` keeps the name it was built with (for
    example `bioclip25_pollinator_orders_europe_v1`); in the later packs the `pack_id` is the
    file name. The file names follow the FaunaPulse naming rule.

  **Demo, not maintained:**
  - `bioclip-2.5_flower-visitors-32fam_v1.fpack` (37,461 names) shows that a pack can be built
    from a list of taxa given by the user. It holds every species, worldwide, of 32 hand-picked
    families. It is not a justified selection of flower visitors: of its insects in the four
    main orders only 14% have 3 or more GBIF records in Europe, and it leaves out flower visitors
    such as the bee family Melittidae, longhorn beetles and most moths. Use a continent pack.
- `.json`: the export manifest. Its `"file"` field gives the name before upload
  (`bioclip-25_image_fp16.tflite`); same SHA-256.
- `LICENSE`: the MIT licence of BioCLIP, copied from the authors' code repository.

## Names

The names and their taxonomy are those of TreeOfLife-200M, as the BioCLIP authors publish them:
no name is corrected, added or removed. A few rows of the name list are not in the plain form
"genus, epithet". In the v2 and later packs the app uses such a row only down to the rank where it is still
plain:

| Row | Example (row of `txt_emb_bioclip-2.5-vith14.json`) | Used down to |
|---|---|---|
| kingdom "Metazoa", which the dataset card lists among the unresolved kingdoms ("`Metazoa` instead of `Animalia`"); read as Animalia (NCBI Taxonomy lists Animalia as a synonym of Metazoa) | 444525: Bombyliidae, Pterygota, "1820" | family |
| the species field starts with another genus than the genus field | (22 rows in the whole list) | family |
| the species field is not one plain epithet (years, codes, placeholders such as "spec") | 101110: Muscidae, Atherigona, "GRG-2020" | genus |

The v2 Europe pack and the continent packs have none of these rows, except one in the North
America pack (species field not plain). Each such pack lists such rows in its header
(`name_rules.rows`: the row of the source file and its names as published), so every change can
be checked and undone; `source.names_file_sha256` names the source file. The rules look only at
the form of a row: some plain BioCLIP 2.5 rows join the genus of a two-genera BioCLIP 2 row with
its epithet (for example "Ceratomegilla rufocincta", where the BioCLIP 2 list has Ceratomegilla,
"Semiadalia rufocincta"), and they are used as published.

To cite the names of a v2 or later pack: "TreeOfLife-200M (https://doi.org/10.57967/hf/8980), names file
`embeddings/txt_emb_bioclip-2.5-vith14.json` (SHA-256 af0cb41f...), names used as published. The
dataset card lists kingdom Metazoa as unresolved; it is read as Animalia (NCBI Taxonomy lists
Animalia as a synonym). Rows not in the plain form are used only down to genus or family, as
listed in the pack header."

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
  is not redistributed). v2: records counted per accepted species of the Catalogue of Life
  (release 2026-06-19 XR, as GBIF interprets them), with the table of `insect-detect-post` v1.1.0
  (made from the BioCLIP 2 name list; a BioCLIP 2.5 pack keeps the names that both lists share).
  v1: the earlier table (GBIF Backbone keys).
- **Continent lists (Europe v3 and the other continents, v1)**: the same method, with GBIF.org
  occurrence data queried on 17 November 2026 (one request per continent and per country). GBIF
  read its records in the Catalogue of Life Extended Release COL26.10 XR
  (https://doi.org/10.48580/dgzlf) on that day, so the TreeOfLife-to-GBIF key table was made for
  that release and for the BioCLIP 2.5 name list itself, by running the mapping script of
  `insect-detect-post` unchanged (commit da050d2) on 17 November 2026; it is not his published
  table v1.1.0, which is of COL26.6 XR and of the BioCLIP 2 name list. The keys in these packs
  are keys of COL26.10 XR; a key can be a different one in a later release although the name
  stays. Extinct species: the Catalogue of Life's extinct mark (ChecklistBank), kept when GBIF
  holds a human or machine observation of the species. Neighbour table from Natural Earth
  country borders (admin-0 map units 1:10m, v5.1.2, public domain).

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
them by hand). Download on Wi-Fi (1.27 GB for the model, plus about 48 to 125 MB for a label
pack).

## How it was made

`bioclip_export/` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/bioclip_export)
(CPU only, about 11 GB memory): `export_image_tower.py --model bioclip-2.5` for the image tower,
`build_label_pack.py --model bioclip-2.5` for the label packs (for the continent packs with
`make_gbif_mapping.py --model bioclip-2.5`, `build_region_species_list.py`,
`build_country_lists.py` and `country_neighbours.py`), `verify_parity.py` for the
PyTorch vs `.tflite` comparison.

## Limitations

- For bias and limits the authors refer to the BioCLIP 2 card: its training data is long-tailed
  across taxa, so answers may lean towards well-represented species (see the
  [BioCLIP 2 model card](https://huggingface.co/imageomics/bioclip-2#bias-risks-and-limitations)).
- The phone can only answer with names that are in the chosen label pack. The "none of these"
  rows are a FaunaPulse addition that BioCLIP 2.5 was not designed with.
- Names and taxonomy are as published in TreeOfLife-200M, so a name may be outdated or a
  synonym. The v1 32-family demo pack also uses the rows described in [Names](#names) as published
  (for example the species answer "Atherigona GRG-2020"); the v2 and later packs use them only
  down to genus or family.
- The country lists of the continent packs show where GBIF holds records, which depends on how
  well a country is recorded (well in Europe, much less in many tropical countries), and species
  do not follow borders: a country list is a practical narrowing, not a range map. The whole
  pack is always available.
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
  The GBIF keys and accepted names in the continent packs come from the Catalogue of Life (CC BY
  4.0, please cite it as below); their country lists are derived from GBIF.org occurrence counts
  and Natural Earth borders (public domain).

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
  the Insect Detect camera trap [Software]. Zenodo. https://doi.org/10.5281/zenodo.21822140 (all
  versions; v2 lists: v1.1.0, https://doi.org/10.5281/zenodo.23404778)
- Bánki, O., Roskov, Y., Döring, M., et al. (2026). Catalogue of Life (Version 2026-06-19 XR).
  Catalogue of Life Foundation. https://doi.org/10.48580/dgy8b (the species counting of the v2
  Europe list)
- Bánki, O., Roskov, Y., Döring, M., et al. (2026). Catalogue of Life (2026-10-16 XR).
  Catalogue of Life Foundation. https://doi.org/10.48580/dgzlf (COL26.10 XR: the species
  counting, keys and accepted names of the continent packs)
- GBIF.org (2026). GBIF Occurrence Search, https://www.gbif.org (queried 26 October 2026 for the
  v2 Europe list and 17 November 2026 for the continent packs)
- Country borders of the neighbour table: Made with Natural Earth (public domain),
  https://www.naturalearthdata.com

## Notes

- Model files and label packs in this repository are never replaced: the app pins each
  download to a commit and checks its SHA-256. New weights or packs get a new file name.
- The `.json` manifest, which the app does not download, may be corrected.
- BioCLIP 2.5 packs work only with BioCLIP 2.5 (1024 numbers per name), not with BioCLIP 2.
