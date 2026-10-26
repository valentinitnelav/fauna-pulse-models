# Agent overview (state only)

What is true now, where it lives, and the reason where it is not obvious. Replace text when it
changes; history is git. Keep under ~20k characters.

## Purpose

PC side of the FaunaPulse Android app (app repo `valentinitnelav/fauna-pulse`, usually at
`~/InsectDetectApp/fauna-pulse`, called `$APP` below). Converts detectors and identification
models to LiteRT `.tflite` (+ `.fpack` name lists), publishes them on Hugging Face
(`Valen-Peregrine`) with model cards; the app's bundled download catalogue points at them.

The tools came from the app repo by plain copy (fauna-pulse@c4c367d, 2026-10-15); their older
history stays there. Still in the app: `tool/video_eval` (scores the app's tracking output),
`tool/perf_summary.dart` (imported by a CI test), `tool/model_downloads` (edits the bundled
catalogue) and the `.fpack` test fixtures.

## Tools (all CPU-only; Python 3.12; shared venv `.venvs/export`, exact pins `requirements-lock.txt`)

Run each script from its own folder (`cd <tool>`); default output is `./out` (git-ignored).
Use `python -E -s` rather than `python -I` for checks: `-I` drops the script folder from the
import path and breaks `from fpack import ...` and the sibling imports.

| Script | Does | In -> out | Needs |
|---|---|---|---|
| `detector_export/export_detector.py` | YOLO `.pt` -> fp16 (default) / w8a32 / fp32 / int8 `.tflite`; flat-bug segmentation -> box detector; agreement check vs PyTorch; `.json` manifest (sha256, sizes; published) + local `.check.json` (check results, never published) | `--weights --name --imgsz ... --check-images` -> `out/<name>_<px>_<q>.tflite/.json` | int8: `--data` YAML with calibration images; imports `../bioclip_export/quantise_tflite.py` (fp16 cast) |
| `classifier_export/export_insectdct_cls.py` | insectDCT V7 `.pth` -> `.tflite` + `.json` + class-list `.fpack`; checks vs insectDCT's own code (local `.check.json`); `--draft-taxa` rewrites tracked `taxa/insectdct-cls-v7.csv` | `--source-dir` (authors' V7 zip, unpacked) + `--code-dir` (insectDCT's `common/`, GPL code; `--zoo-dir` = old alias) -> `out/` | imports `../bioclip_export` (fpack, export_image_tower, quantise_tflite) |
| `bioclip_export/export_image_tower.py` | BioCLIP 1/2/2.5 image tower -> `.tflite` (GPU-friendly 4-d attention, normalisation + L2 built in) + `.json` | HF download or `--weights` -> `out/<model>_<px>_<prec>.tflite` | ~7 GB RAM (11 GB for 2.5), internet |
| `bioclip_export/quantise_tflite.py` | fp32 `.tflite` -> fp16 / int8 (dynamic range) / int8 weights; patches an ai-edge-quantizer 0.9 bug | writes next to input | no calibration images |
| `bioclip_export/build_label_pack.py` | TreeOfLife embeddings, filtered, names as published (rows not in the plain form used only to genus/family: `--name-rules`, listed in the header; v1 packs: `none`), + "none of these" sink rows -> label pack `.fpack` | HF dataset (2.7 GB) or `--embeddings-dir` -> `out/<pack_id>.fpack` | torch (text tower), internet |
| `bioclip_export/build_region_species_list.py` | regional species list from GBIF (COL XR accepted species, mapping of insect-detect-post v1.1.0; v1 packs used v1.0.0) | GBIF API -> CSV + `.json` | internet, stdlib only |
| `bioclip_export/verify_parity.py` | PyTorch vs `.tflite` embeddings: cosine, same answer per rank | `--tflite --images [--pack ...] [--crops-csv]` | |
| `bioclip_export/inspect_tflite.py` | ops and weight storage of a `.tflite` (positional arg, no `--help`) | prints | |
| `bioclip_export/reproduce_track_conf.py` | recomputes the app's per-track pooling off the phone | session `identification/embeddings_*` + pack | own copy of the fpack reader |
| `bioclip_export/app_rules.py` | the app's identification defaults and top-down answer (Python copy), imported by `verify_parity.py` and the calibration benchmark (separate repository) | | numpy, scipy |
| `bioclip_export/fpack.py` | `.fpack` format: `write_fpack`, `write_class_list`, `read_fpack`, `class_probabilities`; `__main__` writes the app's test fixtures (2 positional paths; no argparse: `--help` would create a file) | | numpy |
| `publish/publish_hf.py` | dry run (default; `--write-cards DIR` writes the rendered cards, metadata checked by the Hub) / `--upload --repo <one>` / `--verify [--full]`; config `publish/hf_repos.json`, cards `hf_cards/<repo>/README.md` (`{{FILES_TABLE}}`, `{{CODE_COMMIT}}` filled at upload), result `publish/published.json` | `--catalogue $APP/assets/model_downloads.json` | huggingface_hub, `hf auth login`; refuses unpushed code commit, catalogue mismatch, replacing a file (except a corrected `.json` with `--replace-manifests`) |
| `legacy/*.py` | first YOLO exports (Jun-Aug 2026), superseded by `detector_export`; not the exact recipe of the MDV6 file (they pass options its Ultralytics 8.3.235 lacks) | | |

Measured results (phone timings, fp16 vs w8a32 vs int8) are in each tool's README.

## Model files (git-ignored, never committed)

Since 2026-10-16 the large files live on the maintainer's portable SSD
(`/media/$USER/PortableSSD/`, ext4), not on the laptop (`/home` was 93 % full). Plug it in
before any task that needs them; the published files can also be fetched again from Hugging
Face (table below) and the source models from their authors' links (model cards).

- Export outputs: `<tool>/out/` is empty here (recreated by the next export). The 2026-10-16
  state (4.9 GB, 66 files, byte-checked) is in `/media/$USER/PortableSSD/fauna-pulse-models/<tool>/out/`
  (the SSD copy of this repo, without `.venvs/`; rebuild a venv from `requirements-lock.txt`
  instead of copying one, its paths are absolute). `bioclip_export/out/gpu4d/bioclip-2_image_fp16.tflite`
  is the GPU export published as `bioclip-2_image_fp16_4d.tflite`; `bioclip_export/out/bioclip25/`
  holds BioCLIP 2.5; `detector_export/out/MDV6-yolov10-c_int8_256.tflite` is the copy of the
  app's GitHub release asset.
- `publish/publish_hf.py --upload` and the dry run read the `local` paths of `publish/hf_repos.json`
  (`<tool>/out/...`): with the SSD plugged in, link them (`ln -s
  /media/$USER/PortableSSD/fauna-pulse-models/<tool>/out <tool>/out`, git-ignored) or copy them back.
  `--verify` needs no local files. The app's real-file test: `FAUNAPULSE_MODELS_REPO=/media/$USER/PortableSSD/fauna-pulse-models`.
- Hugging Face cache (BioCLIP 2 source weights 1.7 GB, TreeOfLife-200M species names 2.8 GB):
  `/media/$USER/PortableSSD/.cache/huggingface/hub`, removed from `$HOME/.cache/huggingface`.
  Use it in place with `export HF_HUB_CACHE=/media/$USER/PortableSSD/.cache/huggingface/hub`
  (tested offline 2026-10-16: open_clip loads BioCLIP 2, `build_label_pack.py` finds the names).
  Set `HF_HUB_CACHE`, not `HF_HOME`: the login token stays in `$HOME/.cache/huggingface`.
  Without the SSD the scripts download the same files again (internet, 4.5 GB).
- BioCLIP 2.5 sources (`open_clip_model.safetensors` 3.9 GB, `txt_emb_bioclip-2.5-vith14.npy`
  3.3 GB): `/media/$USER/PortableSSD/insect-model-zoo/weights/bioclip-2.5/` (`--weights`,
  `--embeddings-dir`); insectDCT V7 zip and `.pth`: `.../weights/insectdct-cls-v7/`.
- Files tried on the phone: `$HOME/InsectDetectApp/ForFaunaPulse/models/` (2.6 GB, still on the laptop);
  `$HOME/InsectDetectApp/weights/` is an older subset.
- Those folders also hold collaborator models: publish nothing that is not in the table below
  without the maintainer's explicit clearance.

## Published on Hugging Face

All six PUBLIC since 2026-10-16 (uploaded private first, checked by the owner, then switched;
code commit `27c23bd`, HF commits in the table; public collection
https://huggingface.co/collections/Valen-Peregrine/faunapulse-6ad22f04766aed6641688544). Every
file was re-downloaded and SHA-256-checked while private; once public, `publish_hf.py --verify`
(anonymous: size and SHA-256 at the pinned commit) passed for all 25 files, and one detector
(flat-bug N) was downloaded through its pinned link like the app does (owner: no need to
download the big BioCLIP files again). Pinned links: `publish/published.json`. Layout:
one public repo per model family under `Valen-Peregrine`, named after the app's model id
without the size or backbone part, plus `-tflite`; sizes, input sizes and precisions are
separate files in it (one card and licence per family; a repo per size would repeat cards that
drift apart). Files are published under the app's naming rule, because a published name can
never change (10 of the 12 catalogue files got a new name, same bytes; `publish_hf.py` matches them to the catalogue by SHA-256), each
repo also gets the original model's `LICENSE` (`hf_cards/<repo>/LICENSE`, verbatim upstream text).
Card sources: `hf_cards/<repo>/README.md`. Provenance (authors, citations, licences, source
weights and their SHA-256) was researched and verified on 2026-10-15; the cards hold the result.
All source files were downloaded again from the authors' own links and SHA-256-checked on
2026-10-16 (byte-identical); the cards name only those links.

| HF repo | Files (published name) | Catalogue name today | Licence | Status |
|---|---|---|---|---|
| `mdv6-tflite` | `mdv6-yolov10c_256_w8a32.tflite` | `MDV6-yolov10-c_int8_256.tflite` | AGPL-3.0 | public, `009a5a9` |
| `flatbug-tflite` | `flatbug-n_640_fp16.tflite` (+ `.json`), `flatbug-s_640_fp16.tflite` (+ `.json`; new, gets a catalogue file when the links go in) | same / none | MIT | public, `0ab3ca4` |
| `insectdct-v8-tflite` | `insectdct-v8s_640_fp16.tflite` (+ `.json`) | `insectdct-v8-s_640_fp16.tflite` | GPL-3.0 | public, `f5ac1bb` |
| `insectdct-cls-v7-tflite` | `insectdct-cls-v7-eff2s_224_fp16.tflite`, `.fpack` (+ `.json`) | `insectdct-cls-v7_eff2s_fp16.*` | GPL-3.0 | public, `8ded6d9` |
| `bioclip-2-tflite` | `bioclip-2_224_fp16.tflite` (+ `.json`), `bioclip-2_pollinator-orders-europe_v1.fpack`, `bioclip-2_flower-visitors-32fam_v1.fpack`, `bioclip-2_mammals-birds-world_v1.fpack`; v2: `bioclip-2_pollinator-orders-europe_v2.fpack`, `bioclip-2_mammals-birds-world_v2.fpack` | `bioclip-2_image_fp16_4d.tflite`, `bioclip2_*_v1.fpack`; v2 none yet | MIT (packs CC0-1.0, common names GBIF CC BY 4.0) | public, `cbf9b26`; v2 `6a2961e` |
| `bioclip-2.5-tflite` | `bioclip-2.5_224_fp16.tflite` (+ `.json`), `bioclip-2.5_pollinator-orders-europe_v1.fpack`, `bioclip-2.5_flower-visitors-32fam_v1.fpack`; v2: `bioclip-2.5_pollinator-orders-europe_v2.fpack` | `bioclip-25_image_fp16.tflite`, `bioclip25_*_v1.fpack`; v2 none yet | MIT | public, `dd1f5b7`; v2 `2f54024` |

Label packs v2 (built, checked and independently reviewed 2026-10-26, uploaded the same day;
local files `bioclip_export/out/packs_v2/` on the SSD; `--verify` passed for all 28 files):
Europe for BioCLIP 2 and 2.5, mammals and birds. The app's catalogue does not offer them yet
(an app round). The 32-family packs are an unmaintained demo (no v2); their v1 files stay.

Facts that are easy to get wrong (verified 2026-10-15, sources on the cards):
- Sources = the authors' own links only (owner's rule, 2026-10-16): flat-bug from the ERDA store
  at Aarhus University; insectDCT from its GitHub repo (detector, code at e459ae8) and the Google
  Drive zip its README links (classifier); BioCLIP 2 / 2.5 and the TreeOfLife embeddings from
  Imageomics on HF; MegaDetector from Zenodo. The InsectAI Model Zoo collects other people's models:
  it is never the author or the distributor, so cards and docs do not cite it as a source (it
  stays only where its design is compared, e.g. `classifier_export/README.md` section 5).
- The MegaDetector file is w8a32 (8-bit weights, float activations), not full INT8: Ultralytics
  8.3 names its dynamic-range TFLite `_int8.tflite`; no calibration pictures were involved.
- flat-bug N and S are the 2024 YOLOv8-seg models of the paper. The authors' default since
  flat-bug 1.2 (July 2026) is `flat_bug_M_v2.pt`: YOLO26m-seg, checkpoint 54.6 MB, trained at
  1024 px for 300 epochs on the flatbug data (no new data named in the release or PR #153);
  on the authors' validation set vs the old M: precision 92.3 vs 92.4, recall 96.4 vs 95.0, F1
  93.7 vs 93.1. No N or S v2 exists. Not converted (owner did not ask for it): about 45-50 MB
  as fp16 (estimate), about 3 times the compute of S; a YOLO26 export should pass
  `nms=None` (raw head; the app reads both heads).
- insectDCT "v8" = the authors' detection dataset version 8 (network YOLO11s), not YOLOv8;
  the preprint has six authors since version 3 (Jepsen added).
- Ultralytics writes `author: Ultralytics`, `license: AGPL-3.0` into every exported file's
  metadata; the cards explain this.
- A classifier's class list names its model inside (header `model_id`), and the app's Identify
  screen refuses the pair when that differs from the model file's stem (`_pairingProblem` in
  `$APP/lib/fauna_pulse/screens/identification_screen.dart`). So the insectDCT class list and
  manifest were REBUILT under the new name on 2026-10-16 (`export_insectdct_cls.py --backbone eff2s`,
  reusing `out/insectdct-cls-v7-eff2s_224_fp16.tflite`, a byte copy of the old file; same check
  result; new `.fpack` SHA-256 `6d5ec961...`, which the catalogue takes when the links go in). Label packs
  pair by file name and size only.

Card update 2026-10-17 (code commit `b4dc9c3`): all six cards re-uploaded; the manifests of
flat-bug N and S, insectDCT v8 and the insectDCT classifier corrected with `--replace-manifests`
(check results moved to local `.check.json` files next to them on the SSD). Model files, class
list and packs unchanged; `--verify` passed for all 25 files. HF commits: `publish/published.json`.

Later uploads: one repo per step, `--upload --repo X --private`, owner checks, then public;
with `--private` a new collection is created private too. The app took the published names and
pinned links in its round 314 (`update_catalogue.py --urls`, all 13 catalogue links match
`published.json`).

The MegaDetector file is also a GitHub release asset of the app (`v0.8.0-alpha.1`); that asset
stays online forever. The owner's older HF repo `flower_visitor_order_detector` is unrelated.

## Model card rules (owner's)

- Original authors, paper, code and source weights (with SHA-256) near the top, from the
  authors' own links only; detector cards carry the owner's demo disclaimer.
- No accuracy claims for the converted files: say that no large-scale test in FaunaPulse has
  been carried out yet and what results depend on (image quality, size of the insect in the
  picture, background, phone, app settings). Do not publish agreement-check results. Phone
  speeds and the authors' own published results are fine.
- Do not cast doubt on the authors' models: "not yet tested at scale in FaunaPulse", not "how it
  does on X is not known".
- Licences: the original `LICENSE` verbatim in each repo. Ultralytics-trained models: quote
  https://www.ultralytics.com/license ("all Ultralytics YOLO trained models fall under the
  AGPL-3.0 License by default") and say AGPL-3.0 applies (or an Enterprise License). Label packs
  follow their source, TreeOfLife-200M (CC0 1.0, the BioCLIP authors' licence); its common names
  are from the GBIF Backbone Taxonomy (CC BY 4.0, credit GBIF).
- Label packs narrow the authors' name list to some taxa (pybioclip: this "may improve the
  accuracy"); they are not presented as own work.

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
- Size limit: 2 GiB for every model file and name list, detectors included
  (`kMaxModelFileBytes`, `$APP/lib/fauna_pulse/models/model_file_security.dart`). Model files
  are downloaded or imported, never shipped in the APK, so a large detector only costs speed.
  App versions before 0.9.0-beta.1 reject detector files above 30 MiB.
- Licences and citations: each model's card in `hf_cards/` (the only source). Models the app
  knows but does not offer (ArthroNat n) keep licence and source in `$APP/assets/model_downloads.json`.

## Keep in sync with the app

| Here | App ($APP) | Check |
|---|---|---|
| `.fpack` writer `bioclip_export/fpack.py` | reader `lib/fauna_pulse/identification/label_pack.dart` (header comment = format notes) | fixtures `test/fauna_pulse/fixtures/tiny_pack.fpack`, `tiny_classes.fpack`, tests `label_pack_test.dart`, `track_fusion_test.dart` |
| `fpack.class_probabilities` | `lib/fauna_pulse/identification/track_fusion.dart` (same formula) | `track_fusion_test.dart` uses `test_scores`/`test_probs` stored in the fixture |
| output names of all export scripts | naming rule, `tool/model_downloads/README.md` | `update_catalogue.py --names` |
| model input/output layout (NCHW 0..1; embedder output `[1,n]`; classifier heads concatenated, `dim` = their sum; detector `[1,4+c,N]`; Ultralytics metadata re-appended to fp16 files) | `packages/ultralytics_yolo` (Kotlin), `Embedder.kt`, `model_file_kind.dart` | phone checks: `integration_test/*_check_test.dart` |
| pooling rule copy in `reproduce_track_conf.py` | `track_fusion.dart` | by hand |
| `bioclip_export/app_rules.py`: defaults (48 px, 10 crops, drop factor 10) and top-down answer | `lib/fauna_pulse/identification/identification_assets.dart`, `track_fusion.dart` | by hand |
| app behaviour named on the cards, kept broad on purpose (owner, 2026-10-16): cls card "How FaunaPulse uses the scores" (no formula, points to the app docs); detector cards "the app does NMS", 640 px input, square area of interest (v8 card); every card's "Use in FaunaPulse" (menu path, "versions after v0.8.0-alpha.1") | the matching app code and screens | by hand; editing a card on the Hub is a new commit and never breaks the pinned file links |

## Used in place by the calibration code

The owner's private calibration code base (`~/fauna-pulse-calibration`, same layout on the
workstation; `classify/models_repo.py` puts `bioclip_export/` on its import path) imports
`fpack.read_fpack`, `export_image_tower` (`MODELS`, `load_open_clip`, `CLIP_MEAN`, `CLIP_STD`)
and `app_rules` (`RANKS`, `EVAL`, `Taxonomy`, `softmax_rows`, `APP_MIN_PX`, `MAX_CROPS`, `DROP`).
Renaming or changing these breaks it: change both code bases together.

Fixture regeneration: `cd bioclip_export && python fpack.py "$APP/test/fauna_pulse/fixtures/tiny_pack.fpack" "$APP/test/fauna_pulse/fixtures/tiny_classes.fpack"`.
Today `tiny_pack.fpack` regenerates byte-identical; the committed `tiny_classes.fpack` predates
round 276 (header `"kind": "classes"`, the writer now writes `"class_list"`; the reader accepts
both), so regenerating it changes the file.

## Invariants

- Never replace or delete a published model file, class list or label pack, or squash history (pinned
  links + SHA-256 in installed apps); `.json` manifests may be corrected (the app never downloads them).
- Collaborator models are never published or committed.
- No usernames, host names, phone serials or collaborators' dataset/run names in tracked files.
- fp16 is the default phone precision (w8a32 rounds more and can change boxes near the threshold);
  full INT8 is for CPU-only phones and needs calibration images.
