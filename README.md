# AI-model quantisation tools for the FaunaPulse app

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.23436564"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23436564.svg" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://huggingface.co/collections/Valen-Peregrine/faunapulse-6ad22f04766aed6641688544"><img src="https://img.shields.io/badge/Hugging%20Face-models-blue?logo=huggingface" alt="Models on Hugging Face"></a>
</p>

PC-side tools that turn detection and identification models into LiteRT files (`.tflite`)
for [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse), an Android app that detects,
tracks and identifies flower visitors (and other animals) on the phone itself.
This repository holds the conversion and quantisation code and the model cards.

The converted model files themselves are published on Hugging Face
([Valen-Peregrine/FaunaPulse](https://huggingface.co/collections/Valen-Peregrine/faunapulse-6ad22f04766aed6641688544)),
where the app's "Download & import models" screen fetches them.

## Layout

| Folder | What it is for |
|---|---|
| [`detector_export/`](detector_export/README.md) | YOLO detectors (`.pt`) to phone files: fp16 / w8a32 / fp32 / full INT8, several input sizes, segmentation models (flat-bug) turned into box detectors, agreement check against PyTorch |
| [`classifier_export/`](classifier_export/README.md) | Classifiers (insectDCT hierarchical classifier V7) to a phone file plus its class list (`.fpack`) |
| [`bioclip_export/`](bioclip_export/README.md) | BioCLIP 1 / 2 / 2.5 image tower to `.tflite`, label packs (`.fpack`) from the TreeOfLife embeddings, quantiser for any float32 `.tflite`, parity checks. Also defines the `.fpack` format (`fpack.py`) |
| [`legacy/`](legacy/README.md) | The first YOLO export scripts (June to August 2026), superseded by `detector_export/` |
| [`docs/MODEL_CONVERSION.md`](docs/MODEL_CONVERSION.md) | Converting by hand (Ultralytics commands, INT8 calibration, other models) |
| `hf_cards/` | Model card of each Hugging Face repository (sources; placeholders filled at upload) |
| `publish/` | `publish_hf.py` (upload and verify), `hf_repos.json` (which file goes where), `published.json` (pinned links, written by the script) |
| `requirements-lock.txt` | Exact versions of the shared environment |

`classifier_export/` and `detector_export/` import code from `bioclip_export/`, so keep the
three folders side by side. Each tool writes into its own `out/` folder (git-ignored).

## Setup

One Python environment serves all three tools. Python 3.12 on Linux (3.13 is not yet
supported by the converter); no GPU is needed.

```bash
python3.12 -m venv .venvs/export
source .venvs/export/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements-lock.txt   # CPU build of PyTorch, about 2.7 GB in total
```

Then `cd` into a tool's folder and follow its README. `--no-cache-dir` keeps pip from
storing several GB of downloads in `~/.cache/pip`.

**GPU or not?** Exporting and quantising (fp16, w8a32, and full INT8 too) run on the CPU in
minutes for YOLO n/s. Full INT8 needs a few hundred representative, unlabelled images
(calibration), not a GPU. A GPU is only needed for training.

## Which tool for which model

| Model | Tool | Calibration images? |
|---|---|---|
| Ultralytics YOLO detector (`.pt`), fp16 (default) or w8a32 | `detector_export/export_detector.py` | no |
| Ultralytics YOLO detector, full INT8 (fastest on phone CPUs) | `detector_export/export_detector.py --quantize int8 --data data.yaml` | yes, about 200 to 500 of your own images |
| insectDCT classifier V7 | `classifier_export/export_insectdct_cls.py` | no |
| BioCLIP image tower | `bioclip_export/export_image_tower.py` | no |
| Label pack (names the phone can answer with) | `bioclip_export/build_label_pack.py` | no |
| Any float32 `.tflite` to fp16 / int8 weights | `bioclip_export/quantise_tflite.py` | no |

## Your own models for FaunaPulse

The tools work for any model of a supported kind, not only for the ones published here:

- **A detector you trained** (Ultralytics YOLO, detection or segmentation `.pt`):
  `detector_export/export_detector.py` writes the phone file and, with `--check-images` (a folder
  of your own pictures), checks that it finds the same boxes as your `.pt`.
  Import it in the app: Menu, **Download & import models**, **Import model
  files…**. Training is not part of this repository: it needs your own
  labelled pictures and usually a GPU. For example, see the tutorial prepared by
  my colleague Maximilian Sittinger: [Insect Detect ML](https://github.com/maxsitt/insect-detect-ml).
- **A name list for BioCLIP** (the names FaunaPulse can answer with, "*label packs*"), for example the
  pollinators of one country: `bioclip_export/build_label_pack.py`
  keeps the taxa you choose (classes, orders, families, optionally a species list such as a
  GBIF list from `build_region_species_list.py`) from the BioCLIP authors' name list
  (TreeOfLife-200M), so only names in that list can go into a name list (label pack).
  No training and no GPU needed to prepare such a name list.
- **A custom classifier with its own fixed classes**: `classifier_export/` converts *insectDCT*'s
  classifier (and tested only with insectDCT so far). Another type of network, like insectDCT,
  needs its own export script with the same input and output
  layout (described in [`classifier_export/README.md`](classifier_export/README.md)) and a
  class list written with `fpack.write_class_list`.

**Important note:**
> The detectors published here were trained by others for other settings (camera traps, flat
> surfaces, collection pictures), not for FaunaPulse: see the note on each model card.

## File names

Phone files follow the app's naming rule `<model>_<input px>_<precision>[_<extra>].<ext>`
(lower case), label packs `<model>_<list>_v<n>.fpack`. The rule and its reasons live in the
app: [`tool/model_downloads/README.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/tool/model_downloads/README.md#naming-rule-round-276).

## Publishing on Hugging Face

Each published model has its own Hugging Face repository with a model card (licence,
original authors, citation, how it was converted, file sizes and checksums); the cards'
sources are in `hf_cards/`. The app downloads every file through a link pinned to one
Hugging Face commit and checks the file's SHA-256, so:

- **Never replace or delete a published file**, and never squash a repository's history:
  installed apps would get a different file (rejected) or none at all. New weights get a
  new file name (or a new repository) and a new link in the app.
- Editing a model card is fine (the files' links stay valid), and so is correcting a `.json`
  manifest, which the app never downloads (`publish_hf.py --upload --replace-manifests`).

`publish/publish_hf.py` does the upload, one repository per run. It checks every file's size
and SHA-256 against the app's catalogue first (by SHA-256 when a file is published under a new
name), refuses to replace a published file, uploads the original model's `LICENSE`
(`hf_cards/<repo>/LICENSE`) with the files, and writes the pinned links to
`publish/published.json`:

```bash
hf auth login                                      # once, with a write token
python publish/publish_hf.py --catalogue "$APP/assets/model_downloads.json" --write-cards /tmp/cards   # dry run + cards to read
python publish/publish_hf.py --catalogue "$APP/assets/model_downloads.json" --upload --repo <name> --private
python publish/publish_hf.py --verify --full       # anonymous download check of every pinned link
```

(`$APP` = your clone of the app repository.) The code commit the cards link to must be pushed first.

## Keep in sync with the app

Some rules exist on both sides. Change them together:

| Here | In the app (fauna-pulse) |
|---|---|
| `.fpack` format, writer `bioclip_export/fpack.py` | reader `lib/fauna_pulse/identification/label_pack.dart`; fixtures `test/fauna_pulse/fixtures/tiny_pack.fpack`, `tiny_classes.fpack` (regenerate with `python fpack.py "$APP/test/fauna_pulse/fixtures/tiny_pack.fpack" "$APP/test/fauna_pulse/fixtures/tiny_classes.fpack"`) |
| scoring `fpack.class_probabilities` | `lib/fauna_pulse/identification/track_fusion.dart` |
| output names of the export scripts | naming rule in `tool/model_downloads/README.md` |
| the model files' input and output layout (described in each tool's README) | `packages/ultralytics_yolo` (detectors), `Embedder.kt` (identification) |

## Licences

The code in this repository is AGPL-3.0 (see `LICENSE`) - it inherits from the FaunaPulse app.
Each converted model keeps the licence of its original model (AGPL-3.0, GPL-3.0 or MIT),
named on its model card (source in [`hf_cards/`](hf_cards/)).
The conversions are unofficial: not provided or endorsed by the original authors.

## Provenance

The three tool folders and `docs/MODEL_CONVERSION.md` were moved here from the app
repository `fauna-pulse` at commit `c4c367d` (2026-10-15), where their history remains
(`tool/bioclip_export`, `tool/classifier_export`, `tool/detector_export`, commit rounds 208 to 284).

## Citation

<a href="https://doi.org/10.5281/zenodo.23436564"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23436564.svg" alt="DOI"></a>

If you use this repository in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).
GitHub renders a ready-made "Cite this repository" button from that file (top-right of the repository page).

Example:

> Ștefan, V. (2026). Tools to convert and quantise detection and identification models for the FaunaPulse app [Software]. Zenodo. https://doi.org/10.5281/zenodo.23436563. Source code: https://github.com/valentinitnelav/fauna-pulse-models

Every release is archived on Zenodo. The concept DOI (https://doi.org/10.5281/zenodo.23436563) always resolves to the newest version.
Each release also carries its own version DOI, listed on its Zenodo page and in `CITATION.cff`.

If you need to cite the FaunaPulse app, then check it's own page at https://github.com/valentinitnelav/fauna-pulse (section [Citation](https://github.com/valentinitnelav/fauna-pulse#citation)).
