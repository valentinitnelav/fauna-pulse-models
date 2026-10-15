# Agent overview (state only)

What is true now and where it lives. Replace text when it changes; history goes to
`docs/DECISIONS.md` (append-only) and git. Keep under ~20k characters.

## Purpose

PC side of the FaunaPulse Android app (app repo `valentinitnelav/fauna-pulse`, usually at
`~/InsectDetectApp/fauna-pulse`, called `$APP` below). Converts detectors and identification
models to LiteRT `.tflite` (+ `.fpack` name lists), publishes them on Hugging Face
(`Valen-Peregrine`) with model cards; the app's bundled download catalogue points at them.

## Tools (all CPU-only; Python 3.12; shared venv `.venvs/export`, exact pins `requirements-lock.txt`)

Run each script from its own folder (`cd <tool>`); default output is `./out` (git-ignored).
Use `python -E -s` rather than `python -I` for checks: `-I` drops the script folder from the
import path and breaks `from fpack import ...` and the sibling imports.

| Script | Does | In -> out | Needs |
|---|---|---|---|
| `detector_export/export_detector.py` | YOLO `.pt` -> fp16 (default) / w8a32 / fp32 / int8 `.tflite`; flat-bug segmentation -> box detector; agreement check vs PyTorch; `.json` manifest (sha256, sizes, check) | `--weights --name --imgsz ... --check-images` -> `out/<name>_<px>_<q>.tflite/.json` | int8: `--data` YAML with calibration images; imports `../bioclip_export/quantise_tflite.py` (fp16 cast) |
| `classifier_export/export_insectdct_cls.py` | insectDCT V7 `.pth` -> `.tflite` + `.json` + class-list `.fpack`; checks vs insectDCT's own code; `--draft-taxa` rewrites tracked `taxa/insectdct-cls-v7.csv` | `--zoo-dir` (InsectAI Model Zoo download, GPL code) -> `out/` | imports `../bioclip_export` (fpack, export_image_tower, quantise_tflite) |
| `bioclip_export/export_image_tower.py` | BioCLIP 1/2/2.5 image tower -> `.tflite` (GPU-friendly 4-d attention, normalisation + L2 built in) + `.json` | HF download or `--weights` -> `out/<model>_<px>_<prec>.tflite` | ~7 GB RAM (11 GB for 2.5), internet |
| `bioclip_export/quantise_tflite.py` | fp32 `.tflite` -> fp16 / int8 (dynamic range) / int8 weights; patches an ai-edge-quantizer 0.9 bug | writes next to input | no calibration images |
| `bioclip_export/build_label_pack.py` | TreeOfLife embeddings, filtered, + "none of these" sink rows -> label pack `.fpack` | HF dataset (2.7 GB) or `--embeddings-dir` -> `out/<pack_id>.fpack` | torch (text tower), internet |
| `bioclip_export/build_region_species_list.py` | regional species list from GBIF | GBIF API -> CSV + `.json` | internet, stdlib only |
| `bioclip_export/verify_parity.py` | PyTorch vs `.tflite` embeddings: cosine, top family/species | `--tflite --images [--pack]` | |
| `bioclip_export/inspect_tflite.py` | ops and weight storage of a `.tflite` (positional arg, no `--help`) | prints | |
| `bioclip_export/reproduce_track_conf.py` | recomputes the app's per-track pooling off the phone | session `identification/embeddings_*` + pack | own copy of the fpack reader |
| `bioclip_export/fpack.py` | `.fpack` format: `write_fpack`, `write_class_list`, `read_fpack`, `class_probabilities`; `__main__` writes the app's test fixtures (2 positional paths; no argparse: `--help` would create a file) | | numpy |
| `publish/publish_hf.py` | dry run (default) / `--upload` / `--verify [--full]`; config `publish/hf_repos.json`, cards `hf_cards/<repo>/README.md` (`{{FILES_TABLE}}`, `{{CODE_COMMIT}}` filled at upload), result `publish/published.json` | `--catalogue $APP/assets/model_downloads.json` | huggingface_hub, `hf auth login`; refuses unpushed code commit, catalogue mismatch, replacing a file |
| `legacy/*.py` | first YOLO exports (Jun-Aug 2026), superseded by `detector_export`; very likely the MDV6 int8 recipe (`yolo_pt_to_tflite_general.py`, unconfirmed) | | |

Measured results (phone timings, fp16 vs w8a32 vs int8) are in each tool's README.

## Model files on this computer (git-ignored, never committed)

- Export outputs: `<tool>/out/` here (about 4.5 GB, moved from the app on 2026-10-15).
  `bioclip_export/out/gpu4d/bioclip-2_image_fp16.tflite` is the GPU export published as
  `bioclip-2_image_fp16_4d.tflite`; `bioclip_export/out/bioclip25/` holds BioCLIP 2.5;
  `detector_export/out/MDV6-yolov10-c_int8_256.tflite` is the copy of the app's GitHub release asset.
- Files tried on the phone: `~/InsectDetectApp/ForFaunaPulse/models/` (2.6 GB);
  `~/InsectDetectApp/weights/` is an older subset.
- Those folders also hold collaborator models: publish nothing that is not in the table below
  without the maintainer's explicit clearance.

## Published on Hugging Face

Nothing yet. Planned, one public repo per model under `Valen-Peregrine` (the 6 models of the
app's catalogue; 12 catalogue files, plus each file's `.json` manifest where one exists).
Card sources: `hf_cards/<repo>/README.md`.

| HF repo | Catalogue files | Licence | Status |
|---|---|---|---|
| `megadetector-v6-yolov10c-tflite` | `MDV6-yolov10-c_int8_256.tflite` | AGPL-3.0 | planned |
| `flat-bug-n-tflite` | `flatbug-n_640_fp16.tflite` | as in the app catalogue | planned |
| `insectdct-v8s-tflite` | `insectdct-v8-s_640_fp16.tflite` | GPL-3.0 | planned |
| `insectdct-cls-v7-eff2s-tflite` | `insectdct-cls-v7_eff2s_fp16.tflite`, `.fpack` | GPL-3.0 | planned |
| `bioclip-2-tflite` | `bioclip-2_image_fp16_4d.tflite`, `bioclip2_pollinator_orders_europe_v1.fpack`, `bioclip2_flower_visitors_32fam_v1.fpack`, `bioclip-2_mammals-birds-world_v1.fpack` | MIT (packs CC0-1.0) | planned |
| `bioclip-2.5-vith14-tflite` | `bioclip-25_image_fp16.tflite`, `bioclip25_*_v1.fpack` (2) | MIT | planned |

The MegaDetector file is also a GitHub release asset of the app (`v0.8.0-alpha.1`); that asset
stays online forever. The owner's older HF repo `flower_visitor_order_detector` is unrelated.

## How the app uses published files (app side, for reference)

- Catalogue `$APP/assets/model_downloads.json` is **bundled in the APK** (not fetched): new links
  reach users only with a new app release. Ask before reading the app's `assets/`.
- Schema: `base_url` (required, ends `/`) + per-file optional `url`, `name`, `bytes`, `sha256`;
  parser `$APP/lib/fauna_pulse/models/model_downloads.dart`; spec + upload table
  `$APP/tool/model_downloads/README.md`; sizes/checksums refreshed by
  `$APP/tool/model_downloads/update_catalogue.py`; rules tested by
  `$APP/test/fauna_pulse/model_downloads_test.dart`.
- Downloader `$APP/lib/fauna_pulse/models/file_download.dart`: HTTPS-only redirect chain, final
  HTTP 200, SHA-256 checked when given, no auth headers (HF repos must be public and not gated).
  Use `https://huggingface.co/<owner>/<repo>/resolve/<commit>/<file>`, never `/blob/`.
- Size caps: detector `.tflite` 30 MiB; identification model or `.fpack` 2 GiB.
- Licences and citations of every model: `$APP/docs/THIRD_PARTY_MODELS.md`.

## Keep in sync with the app

| Here | App ($APP) | Check |
|---|---|---|
| `.fpack` writer `bioclip_export/fpack.py` | reader `lib/fauna_pulse/identification/label_pack.dart` (header comment = format notes) | fixtures `test/fauna_pulse/fixtures/tiny_pack.fpack`, `tiny_classes.fpack`, tests `label_pack_test.dart`, `track_fusion_test.dart` |
| `fpack.class_probabilities` | `lib/fauna_pulse/identification/track_fusion.dart` (same formula) | `track_fusion_test.dart` uses `test_scores`/`test_probs` stored in the fixture |
| `APP_MAX_BYTES` in `detector_export/export_detector.py` | `kMaxTfliteModelBytes`, `lib/fauna_pulse/models/model_file_security.dart` | by hand |
| output names of all export scripts | naming rule, `tool/model_downloads/README.md` | `update_catalogue.py --names` |
| model input/output layout (NCHW 0..1; embedder output `[1,n]`; classifier heads concatenated, `dim` = their sum; detector `[1,4+c,N]`; Ultralytics metadata re-appended to fp16 files) | `packages/ultralytics_yolo` (Kotlin), `Embedder.kt`, `model_file_kind.dart` | phone checks: `integration_test/*_check_test.dart` |
| pooling rule copy in `reproduce_track_conf.py` | `track_fusion.dart` | by hand |

Fixture regeneration: `cd bioclip_export && python fpack.py "$APP/test/fauna_pulse/fixtures/tiny_pack.fpack" "$APP/test/fauna_pulse/fixtures/tiny_classes.fpack"`.
Today `tiny_pack.fpack` regenerates byte-identical; the committed `tiny_classes.fpack` predates
round 276 (header `"kind": "classes"`, the writer now writes `"class_list"`; the reader accepts
both), so regenerating it changes the file.

## Invariants

- Never replace or delete a published HF file or squash history (pinned links + SHA-256 in installed apps).
- Collaborator models are never published or committed.
- No usernames, host names, phone serials or collaborators' dataset/run names in tracked files.
- fp16 is the default phone precision (w8a32 changed boxes near the threshold, round 264);
  full INT8 is for CPU-only phones and needs calibration images.
