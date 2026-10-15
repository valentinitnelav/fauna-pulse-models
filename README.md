# AI models for the FaunaPulse app

PC-side tools that turn detection and identification models into LiteRT files (`.tflite`)
for [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse), an Android app that detects
and logs flower visitors (and other animals) on the phone itself. This repository holds the
conversion and quantisation code and the model cards. The converted files themselves are
published on Hugging Face ([Valen-Peregrine](https://huggingface.co/Valen-Peregrine)), where
the app's "Download & import models" screen fetches them.

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
- Editing a model card is fine (the files' links stay valid).

`publish/publish_hf.py` does the upload. It checks every file's size and SHA-256 against
the app's catalogue first, refuses to replace a published file, and writes the pinned links
to `publish/published.json`:

```bash
hf auth login                                      # once, with a write token
python publish/publish_hf.py --catalogue "$APP/assets/model_downloads.json"          # dry run
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
| `APP_MAX_BYTES` (30 MiB) in `detector_export/export_detector.py` | `kMaxTfliteModelBytes` in `lib/fauna_pulse/models/model_file_security.dart` |
| output names of the export scripts | naming rule in `tool/model_downloads/README.md` |
| the model files' input and output layout (described in each tool's README) | `packages/ultralytics_yolo` (detectors), `Embedder.kt` (identification) |

## Licences

The code in this repository is AGPL-3.0 (see `LICENSE`); it comes from the FaunaPulse app.
Each converted model keeps the licence of its original model (AGPL-3.0, GPL-3.0 or MIT),
named on its model card and in the app's
[`docs/THIRD_PARTY_MODELS.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/docs/THIRD_PARTY_MODELS.md).
The conversions are unofficial: not provided or endorsed by the original authors.

## Provenance

The three tool folders and `docs/MODEL_CONVERSION.md` were moved here from the app
repository `fauna-pulse` at commit `c4c367d` (2026-10-15), where their history remains
(`tool/bioclip_export`, `tool/classifier_export`, `tool/detector_export`, rounds 208 to 284).
