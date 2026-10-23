# BioCLIP export tools: put BioCLIP 2 on the phone (PC side)

FaunaPulse's *Identify organisms* feature runs the BioCLIP 2 image tower on the phone.
The phone cannot read the original PyTorch checkpoint, so two files are prepared once
on a normal computer (no GPU needed) and copied to the phone:

| File | What it is | Size (BioCLIP 2) |
|---|---|---|
| `bioclip-2_224_fp16.tflite` | the image tower converted for the phone (fp16 weights) | 581 MiB |
| `<pack>.fpack` | a **label pack**: the names the model may choose from, their pre-computed embeddings, their taxonomy, plus six "none of these" entries (flower, leaf, shadow, debris, blurry, web) | 60 MiB for 32 flower-visitor families; up to ~430 MB for all Insecta + Arachnida |

Everything below was run end to end on 2026-09-20 (Ubuntu 24.04, Python 3.12, 15 GB RAM,
no GPU). Times and sizes are from that run.

## 0. What you need

- Python 3.11 or 3.12 (tested: 3.12.3). 3.13 is not yet supported by the converter.
- RAM: about 7 GB free during the conversion of BioCLIP 2 (BioCLIP 2.5 needs roughly twice).
- Disk: about 12 GB free in total: 4 GB for the Python packages (CPU build of PyTorch),
  1.7 GB for the BioCLIP 2 checkpoint, 2.8 GB for the TreeOfLife name embeddings (both
  downloaded once into `~/.cache/huggingface`), and 2 GB for the conversion outputs.
- Internet for the first run (Hugging Face downloads; no account or token needed).
- `adb` (Android platform tools) to copy files to the phone, or any USB file transfer.

## 1. Create the Python environment

All three tools share one environment, made once from the repository root as described in
the [repository README](../README.md#setup). Then, from the repository root:

```bash
source .venvs/export/bin/activate
cd bioclip_export
```

The environment uses the CPU build of PyTorch (about 300 MB instead of the 3.5 GB CUDA
build); both give identical results here.

## 2. Export the image tower (about 2 minutes plus the download)

```bash
python export_image_tower.py --model bioclip-2 --precision fp16 --out ./out
```

What happens: the 1.7 GB checkpoint `imageomics/bioclip-2` is downloaded into the
Hugging Face cache (once); the image tower is wrapped so the phone can feed plain RGB
pixels in 0..1 (the colour normalisation and the final L2 normalisation are inside the
graph); `litert-torch` converts it to a float32 `.tflite` (1,216 MB, 74 s, about 7 GB
RAM); `ai-edge-quantizer` casts the weights to fp16 (609 MB, 6 s). Output:

```
out/bioclip-2_224_fp16.tflite   the model for the phone
out/bioclip-2_224_fp16.json     manifest: embedding size 768, input 224x224, logit scale 100, sha256
```

**GPU-friendly attention (round 242, the default).** Every attention layer is exported as
plain matrix steps with tensors of at most 4 dimensions (`--attention 4d`). PyTorch's own
attention layer passes the data through 5-dimensional tensors, which the phone's GPU engine
cannot run: files exported before round 242 therefore always fall back to the CPU (the
phone's log says `RESHAPE ... has bad input dims size`; only 63 of 1488 steps could go to
the GPU). With the 4d export all 1392 steps run on the GPU. The weights and the maths are
the same. On the Xiaomi test phone one crop took 0.27 s on the GPU instead of
2.6 s on the CPU. `--attention torch` makes the old kind of file (for comparison only).
The manifest records which kind a file is (`"attention": "4d"`).

Options: `--precision fp32` (nothing cast, 1.2 GB; also runs on the phone),
`--precision int8` or `int8-weights` (about 0.3 GB; see section 2b before using them),
`--keep-fp32` (keep the float32 intermediate, useful for `verify_parity.py`),
`--model bioclip-2.5` (ViT-H/14, 3.9 GB download, 1.27 GB fp16, about 11 GB RAM; section 2c),
`--weights FILE` (a checkpoint already on disk instead of the Hugging Face download),
`--onnx` (additionally write an ONNX file).

Check the file before copying it anywhere:

```bash
python inspect_tflite.py out/bioclip-2_224_fp16.tflite
```

Expected (4d export): `FULLY_CONNECTED: {'fp16 weights': 145}` (every large layer is
fp16; the attention's in-projection is split into three), input
`serving_default_args_0 [1, 3, 224, 224]`, output `serving_default_output_0_output [1, 768]`.
If it says `weights computed at runtime (unfolded)`, the conversion ran in the
memory-saving mode (see Troubleshooting).

## 2b. Quantisation: what it is and when it makes a model faster

The float32 file stores each of BioCLIP 2's ~300 million weights in 4 bytes.
*Quantising* stores them with fewer bits: the file gets smaller, and some chips compute
faster, at the price of a little rounding. `quantise_tflite.py` does this step (the
export calls it) and works on any float32 `.tflite`, so collaborators can use it for
their own classifiers or embedders. For YOLO detectors, use Ultralytics' own export
instead ([`docs/MODEL_CONVERSION.md`](../docs/MODEL_CONVERSION.md)). Its comments explain each kind in more detail.

| Kind | Size (BioCLIP 2) | Faster where |
|---|---|---|
| fp32 | 1216 MB | nowhere (the baseline) |
| **fp16** (default) | 609 MB | phone GPU (its native 16-bit maths); on a CPU the weights are widened back to 32 bits, so same speed |
| int8-weights | 309 MB | nowhere: weights widened to floats on loading; only the file is smaller |
| int8 (dynamic range) | 309 MB | CPU (8-bit maths); the gain depends on the chip |

Fewer bits round more: fp16 stays closest to the original, int8 (8-bit maths) changes the
outputs the most; check a quantised file with `verify_parity.py` on your own crops. On a laptop CPU (Intel i5-8350U, 4 threads) all four took 1.0 to 2.0 s per crop:
int8 was at most about 20 % faster, which is less than the laptop's own run-to-run
spread. On the Xiaomi test phone (Snapdragon 888) fp16 took 0.27 s per crop on the GPU
and 2.6 s on the CPU. int8 on a phone CPU has not been measured yet.

**Does quantisation need a GPU?** Making the file does not: it runs on any computer's
CPU in seconds. Whether the smaller file then *runs* faster depends on the chip that
runs it. fp16 helps a GPU. On a CPU only int8 with 8-bit maths can help, and for
BioCLIP it cost accuracy on unsure crops. That is why the app ships fp16 and uses the
GPU where it can.

## 2c. BioCLIP 2.5 (round 264)

BioCLIP 2.5 Huge (ViT-H/14, embeddings of 1024 numbers) needs its own model file and its
own label packs (BioCLIP 2 packs do not fit it). Without `--weights` and `--embeddings-dir` the
scripts download the authors' files from Hugging Face. When the checkpoint
(`open_clip_model.safetensors` of imageomics/bioclip-2.5-vith14) and the TreeOfLife name
embeddings (`embeddings/txt_emb_bioclip-2.5-vith14.npy` and `.json` of the dataset
imageomics/TreeOfLife-200M) are already on disk in one folder, point the scripts at them
instead of downloading 7 GB again:

```bash
W=/path/to/bioclip-2.5
python export_image_tower.py --model bioclip-2.5 --weights $W/open_clip_model.safetensors --out out/bioclip25
python build_label_pack.py --model bioclip-2.5 --embeddings-dir $W --weights $W/open_clip_model.safetensors \
  --classes Insecta --species-csv out/species_europe_pollinator_orders.csv \
  --pack-id bioclip-2.5_pollinator-orders-europe_v1 --out out/bioclip25
python verify_parity.py --tflite out/bioclip25/bioclip-2.5_224_fp16.tflite --weights $W/open_clip_model.safetensors \
  --images /path/to/crops --pack out/bioclip25/bioclip-2.5_pollinator-orders-europe_v1.fpack
```

Measured on the 15 GB laptop (2026-10-01): export 4 minutes, 10.7 GB RAM at the peak (it
used swap); the float32 intermediate is 2.5 GB, above the 2 GB limit of a plain `.tflite`,
which the converter and the quantiser handle by themselves; the fp16 file is 1,266 MB with
all 193 large layers in fp16 and all 32 attention layers GPU-friendly. Packs built:
`bioclip-2.5_pollinator-orders-europe_v1` (34,704 names, 74 MB) and
`bioclip-2.5_flower-visitors-32fam_v1` (37,461 names, 80 MB); the 2.5 TreeOfLife table has
794,878 names (BioCLIP 2's: 867,455), so the same filters keep slightly different lists.
On the phone: the app's [`docs/IDENTIFICATION.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/docs/IDENTIFICATION.md), *BioCLIP 2.5* (CPU only on the 7.4 GB test phone,
5.3 to 5.6 s per crop).

int8 was tried too (`--precision int8`, 639 MB): it changed the outputs more than fp16, and on
the phone's CPU it was slower than fp16 (9.2 s against 5.6 s per crop). fp16 stays the choice.

## 3. Build a label pack (about 5 minutes plus the download)

The first pack for a phone test: 32 families of common flower visitors of temperate
Europe (bees, wasps, hoverflies and other flies, ladybirds and other beetles,
butterflies, bugs, lacewings, scorpionflies, crab, orb and jumping spiders), 38,570
species, 60 MiB:

```bash
python build_label_pack.py --model bioclip-2 --pack-id bioclip-2_flower-visitors-32fam_v1 --out ./out \
  --families Apidae,Halictidae,Andrenidae,Megachilidae,Colletidae,Vespidae,Syrphidae,Muscidae,Calliphoridae,Sarcophagidae,Bombyliidae,Empididae,Conopidae,Stratiomyidae,Tachinidae,Coccinellidae,Oedemeridae,Cantharidae,Pieridae,Nymphalidae,Lycaenidae,Hesperiidae,Papilionidae,Zygaenidae,Sphingidae,Pentatomidae,Miridae,Panorpidae,Chrysopidae,Thomisidae,Araneidae,Salticidae
```

What happens: the TreeOfLife-200M name embeddings (`txt_emb_species.npy`, 2.66 GB, plus
a 92 MB json with the taxonomy) are downloaded once into the Hugging Face cache; the
rows of the requested families are kept (867,455 species in total; Insecta 264,036,
Arachnida 16,406); the six "none of these" prompts are embedded with the BioCLIP text
tower; everything is written as one `.fpack` file (format documented in `fpack.py`). Since
round 266 the same container also holds the *class lists* of fixed-class classifiers such as
insectDCT (no name vectors; `fpack.write_class_list`, see [`classifier_export/README.md`](../classifier_export/README.md)).

### 3b. Regional packs (GBIF occurrence lists)

`build_region_species_list.py` writes the species of chosen orders that GBIF has at
least 3 records for in a region (a GBIF continent or a set of countries). Restricting
the candidates to a regional GBIF species list is what BioCLIP's documentation
recommends ("Geo-Restricted Taxon List Predictions"); the API-based way of getting that
list and the TreeOfLife-to-GBIF key mapping (downloaded on first use) follow Maximilian
Sittinger's `insect-detect-post`. One GBIF request per order and region, about 10 to
60 s each, no account needed:

```bash
python build_region_species_list.py --continent EUROPE   --orders Diptera,Hymenoptera,Coleoptera,Lepidoptera --out out/species_europe_pollinator_orders.csv
python build_label_pack.py --model bioclip-2 --classes Insecta   --species-csv out/species_europe_pollinator_orders.csv --pack-id bioclip-2_pollinator-orders-europe_v1 --out ./out
```

Countries instead of a continent: `--countries DE,AT,CH,CZ,PL` (union). Any CSV with a
`species` column of "Genus epithet" names works as `--species-csv`, so a user-supplied
taxon list is the same one-liner.

### 3c. Packs built so far (2026-09-20)

| Pack id | Selection | Names | Size | Runs in the app today |
|---|---|---|---|---|
| `bioclip-2_flower-visitors-32fam_v1` | 32 flower-visitor families, worldwide | 38,570 | 60 MB | yes |
| `bioclip-2_pollinator-orders-europe_v1` | Diptera, Hymenoptera, Coleoptera, Lepidoptera with GBIF records in Europe | 35,264 | 57 MB | yes |
| `bioclip-2_mammalia-world_v1` | class Mammalia, worldwide, camera-trap sink prompts (`--sink-set mammal`) | 5,999 | 9.4 MB | yes (for MegaDetector "animal" boxes) |
| `bioclip-2_mammals-birds-world_v1` | classes Mammalia and Aves, worldwide, camera-trap sink prompts (round 278; offered for "Mammals and birds" on the home screen) | 17,130 | 28 MB | yes |
| `bioclip-2_pollinator-orders-world_v1` | the four orders, worldwide | 204,620 | 318 MB | not yet (needs the native scorer of a later app round) |
| `bioclip-2.5_pollinator-orders-europe_v1` | as the Europe pack above, for BioCLIP 2.5 (round 264) | 34,704 | 74 MB | yes (with the BioCLIP 2.5 model) |
| `bioclip-2.5_flower-visitors-32fam_v1` | the 32 families, for BioCLIP 2.5 (round 264) | 37,461 | 80 MB | yes (with the BioCLIP 2.5 model) |

**Name clean-up (since 2026-10-20, `clean_taxonomy` in `build_label_pack.py`):** a few
TreeOfLife names come in another format: kingdom "Metazoa" instead of "Animalia" (a separate
branch the app's top-down answer never reaches), a lineage shifted by one rank ("Pterygota" in
the genus slot), or authors, years and subgenera in the species slot. The builder now repairs
them and drops the few it cannot (no epithet left) and exact duplicates; the counts go into the
pack header (`source.name_cleaning`). The v1 packs above were built without it; rows affected in
the published ones ("Metazoa" / species slot not a plain epithet): BioCLIP 2 Europe 36 / 111,
32 families 85 / 418 (among them bee flies, hoverflies and bees), mammals and birds 41 / 613;
BioCLIP 2.5 Europe 0 / 0, 32 families 81 / 105 (codes and bare years such as "1820").
`--no-name-cleaning` reproduces them.

Every pack ends with six "none of these" rows; `--sink-set arthropod` (default) uses
flower-scene prompts, `--sink-set mammal` camera-trap prompts, `--no-sink` none.

Other selections:

```bash
# whole classes (large: ~280k names, ~430 MB; needs the native scorer)
python build_label_pack.py --model bioclip-2 --classes Insecta,Arachnida --out ./out
# whole orders
python build_label_pack.py --model bioclip-2 --orders Diptera,Hymenoptera --out ./out
```

Keep packs under about 100,000 names for the current app version: it scores them in
pure Dart (about 50 ms per crop per 30,000 names) and holds the matrix in memory
(4 bytes per number). Larger packs are built the same way and wait for the native
scorer.

## 4. Verify the export (recommended, about 3 minutes)

```bash
python verify_parity.py --tflite out/bioclip-2_224_fp16.tflite --images /path/to/some/crops \
  --pack out/bioclip-2_flower-visitors-32fam_v1.fpack --limit 50
```

Compares the phone model with the original PyTorch model on your images: cosine
similarity of the embeddings and the same answer at order, family, genus and species with
each pack given (`--pack` takes several). Expected `PARITY OK` with mean cosine >= 0.99 and
family agreement >= 95 %. Each rank is taken the way the app reports it (top-down, a taxon's
probabilities added up); the lines also say how many of the images PyTorch itself is sure
about agree, which tells a quantised file's near-ties apart from real errors (section 2b),
and the largest change of the shown percentage. Any folder of insect photos or crops works
(jpg/png, searched recursively). For crops of whole track IDs, `--crops-csv crops.csv` (one row
per crop: `crop_id`, `kind`, `seq_id`, `crop_px`, `new_filename`) takes one random crop per track
ID and one square per insect-free frame, so long tracks do not fill the sample. The ranks and
the top-down answer come from `app_rules.py`, the Python copy of the app's rules.

## 5. Copy the files to the phone and import them

```bash
adb push out/bioclip-2_224_fp16.tflite /sdcard/Download/
adb push out/bioclip-2_flower-visitors-32fam_v1.fpack /sdcard/Download/
```

(or copy them over USB / a file manager into the phone's Downloads folder). Then in
FaunaPulse: home screen Menu, **Download & import models**, *Import model files…* (the model and the
label pack, both at once: the app sorts them). The app copies both files into its private storage; the
Downloads copies can be deleted afterwards. The app's [`docs/IDENTIFICATION.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/docs/IDENTIFICATION.md) explains the run and
the results.

## Why the full model on the phone, and not a distilled one (for now)

Distillation (training a small "student" network / model to imitate the BioCLIP image tower - the "teacher")
was considered and is deliberately not used yet.

The reasoning:

- FaunaPulse is aimed to use identification **after** recording, in bulk, on a 
  plugged-in phone (e.g. overnight session). Speed is not the main goal there, accuracy is. 
  The "teacher" (the full BioCLIP 2 image tower, ~600 MiB fp16) is the accuracy ceiling
  and needs no training data.
- Published "students" lose fidelity, mostly at species level: 
  - the [BioCLIP 2.5 to FastViT-SA12 student][nate_distill] of Nate Hamilton (24 MB,
  label-free MobileCLIP-style recipe) agrees with its teacher on 71.7 % of top-1 and
  88.8 % of top-5 answers on plants; 
  - the "ConvNeXt-tiny+KD" student model of [Gardiner et al.][gardiner_2025]
  (ICCVW 2025, 101 moth species) reaches 64.7 % top-1 target accuracy with no 
  field labels against 88.3 % for BioCLIP 2 (Table 1 in manuscript), 
  and needs about 50% data mix with expert-labelled field data to catch up.
  Those authors recommend BioCLIP 2 itself when compute allows and labelled field data
  are scarce, which is mostly FaunaPulse situation (at least for now).

If it becomes a goal, the path is ready because the app is model-agnostic (any `.tflite` that maps
RGB to an embedding in a pack's space, declared in the manifest). So a student model
distilled from **BioCLIP 2** can be used (keeping the 768-dimension text space, so the packs stay
valid). The label-free recipe of [Nate Hamilton][nate_distill] (MIT; cache teacher
embeddings of insect images and of FaunaPulse's own crops, train a FastViT-class
student with a cosine loss) applies unchanged; export with
`export_image_tower.py` (timm models convert the same way) and check agreement with
the teacher on real crops with `verify_parity.py`.
Vasu et al. (2024) MobileCLIP, presents the distillation recipe.

A related, non-distillation idea worth keeping in mind
is a small insect-versus-background head trained on stored embeddings, as in
https://github.com/lollogiro/zero-shot-insect-detection (the app's "none of these"
rows are the zero-shot version of it).

References:

> Gardiner, R. J., Mougeot, G., Rowlands, S., Simmons, B. I., Helsing, F., & Høye, T. T. (2025, October). Bridging Domain Gaps for Fine-Grained Moth Classification Through Expert-Informed Adaptation and Foundation Model Priors. In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW) (pp. 5169-5174). IEEE. https://doi.org/10.1109/ICCVW69036.2025.00538;

> Vasu, P. K. A., Pouransari, H., Faghri, F., Vemulapalli, R., & Tuzel, O. (2024, June). Mobileclip: Fast image-text models through multi-modal reinforced training. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 15963-15974). IEEE. https://doi.org/10.1109/CVPR52733.2024.01511

[nate_distill]: https://github.com/CrazedCoderNate/bioclip-mobile-distill
[gardiner_2025]: https://doi.org/10.1109/ICCVW69036.2025.00538

## Troubleshooting

- `ModuleNotFoundError: No module named 'tensorflow'`: an old copy of the script;
  the current one does not use TensorFlow (litert-torch 0.9+ plus ai-edge-quantizer).
- `TypeError: can only concatenate str (not "bytes") to str` inside ai-edge-quantizer:
  a text/bytes mismatch between ai-edge-quantizer 0.9.0 and flatbuffers 25.12.
  `quantise_tflite.py` works around it (tensor names are normalised before the cast).
  If it still appears with newer versions, use `--precision fp32`.
- The fp16 file is larger than the float32 one, or `inspect_tflite.py` reports
  "weights computed at runtime": the conversion ran with `--lightweight`, which leaves
  the LayerNorm-scale products of 48 weight matrices unfolded. Delete the float32 file
  in `out/` and run again without `--lightweight`.
- Out of memory during conversion: close other programs (about 7 GB free RAM is needed
  for BioCLIP 2). `--lightweight` is the last resort (see the previous point).
- Disk: keep about 2 GB free during a run (float32 intermediate plus output).
  `pip cache purge` frees the wheels pip downloaded; the Hugging Face cache
  (`~/.cache/huggingface`) can be deleted after the export (it is re-downloaded on the
  next run).
- Slow PC inference in `verify_parity.py` (about 1 s per crop for fp16, 4 s for
  float32) is normal; it only affects the check, not the phone.

## Files in this folder

| File | Purpose |
|---|---|
| `export_image_tower.py` | checkpoint -> `.tflite` (+ manifest) |
| `quantise_tflite.py` | float32 `.tflite` -> fp16 / int8 / int8-weights, with a quick check (any model) |
| `build_label_pack.py` | TreeOfLife embeddings + filters + sink rows -> `.fpack` |
| `fpack.py` | the pack container format (also writes the app's test fixture) |
| `build_region_species_list.py` | GBIF occurrence facets + TreeOfLife-to-GBIF mapping -> regional species CSV |
| `verify_parity.py` | PyTorch vs `.tflite` comparison on real images |
| `app_rules.py` | Python copy of the app's identification defaults and top-down answer (`verify_parity.py`, calibration scripts) |
| `inspect_tflite.py` | ops, constant sizes and how each layer's weights are stored (fp16, int8, float32) |
| `requirements.txt` | packages (version ranges; exact verified versions: `../requirements-lock.txt`) |
| `out/` | outputs (git-ignored) |

## Attribution of methods and data used here

Two sources are credited for different things; please keep both when citing.

**From BioCLIP / pybioclip (Imageomics; model MIT, embeddings CC0, code MIT):**
- zero-shot identification from a hierarchical taxonomic name list, and the
  TreeOfLife-200M name embeddings the packs are cut from (Stevens et al. 2024; Gu et al. 2025);
- the label-subset mechanism the packs implement (pybioclip's `--subset` /
  `create_taxa_filter` / `apply_filter` over precomputed name embeddings);
- the softmax with the model's logit scale and the roll-up of species probabilities to
  higher ranks (pybioclip's `predict` and `format_grouped_probs`);
- the recommendation to restrict candidates to a regional GBIF or Map of Life species
  list ("Geo-Restricted Taxon List Predictions", https://imageomics.github.io/pybioclip/geo-restricted-taxa/).

**From Maximilian Sittinger's `insect-detect-post` (AGPL-3.0; Sittinger, M. 2026, Zenodo
https://doi.org/10.5281/zenodo.21822140), re-implemented in our own code:**
- the API-based way of building the regional list (GBIF occurrence facets, minimum 3
  records) and his TreeOfLife-to-GBIF key mapping (`tol_gbif_taxon_keys_Arthropoda.csv`,
  downloaded from his release, not redistributed);
- the per-track-ID CSV column names (`pred`, `pred_prob_weighted`, `pred_prob_mean`,
  `track_imgs`, `pred_imgs`, `bioclip_<rank>`) of his `_classified_final.csv`, so both
  tools' outputs can be analysed with the same scripts; the app's pooling rule itself
  differs (round 219: certainty-weighted average of the crops' embeddings scored once, the
  "Average Logit" of Dussert et al. 2025; `reproduce_track_conf.py` in this folder recomputes
  a session's track-id results off the phone from the stored embeddings and a pack);
- the square-on-the-longer-side crop rule (`make_bbox_square()`); FaunaPulse pads at
  photo edges instead of shifting the square, and has an optional margin (0 by default
  since round 256).

**Precedent, not origin:** the "none of these" rows follow the common practice of
explicit background classes, as in the `none_*` classes of the Insect Detect
classification dataset (Sittinger, Uhler & Pink 2023, https://doi.org/10.5281/zenodo.8325384).

**Data:** GBIF occurrence facets and backbone taxonomy, https://www.gbif.org.

## Licenses and citation

Model weights: BioCLIP 2 and BioCLIP 2.5 by Imageomics, MIT (BioCLIP 2.5 model card:
https://huggingface.co/imageomics/bioclip-2.5-vith14). Name embeddings: TreeOfLife-200M,
CC0-1.0. 

Important note - the converted files are unofficial conversions, not provided or
endorsed by Imageomics.  

If you use these models, even if they were adapted for smartphone usage with FaunaPulse, 
please note the citation suggestions of the BioCLIP2 model card on Hugging Face at
https://huggingface.co/imageomics/bioclip-2#citation

See also the model cards [`hf_cards/bioclip-2-tflite`](../hf_cards/bioclip-2-tflite/README.md)
and [`hf_cards/bioclip-2.5-tflite`](../hf_cards/bioclip-2.5-tflite/README.md).
