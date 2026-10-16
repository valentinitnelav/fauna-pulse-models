# Decisions (append-only)

One entry per decision: date, decision, why, rejected options. 2 to 5 lines each. Search,
don't read in full (`grep -n -i <keyword> docs/DECISIONS.md`).

## 2026-10-15: model tools moved out of the app repo
Moved `tool/{bioclip,classifier,detector}_export` and the conversion half of
`docs/MODEL_CONVERSION.md` from fauna-pulse@c4c367d by plain copy (`git archive`). Old history stays
in the app repo. Rejected: `git filter-repo` (its "Round N" commit messages only make sense in the app).
Kept in the app: `tool/video_eval` (scores the app's tracking output), `tool/perf_summary.dart`
(imported by a CI test), `tool/model_downloads` (edits the app's bundled catalogue), the `.fpack` fixtures.

## 2026-10-15: one shared environment, exact pins at the root
`.venvs/export`, made from `requirements-lock.txt` = `pip freeze` of the environment that ran all three
exports (CPU PyTorch). Replaces `bioclip_export/requirements-lock.txt`, which lacked ultralytics,
pandas and opencv and listed CUDA packages that were never installed. Tools stay siblings because
`classifier_export` and `detector_export` import `../bioclip_export`.

## 2026-10-15: quantization/ kept as legacy/
The first YOLO export scripts (June to August 2026) were never in git. Committed under `legacy/`,
scrubbed of usernames, hosts, phone serials and collaborators' run names, because they are the recipe
of record for early phone files (very likely MegaDetector V6 int8). Rejected: deleting them.

## 2026-10-15: models published on Hugging Face, one repo per model
Owner's choice over GitHub release assets: a model card per model (licence, authors, citation,
recipe), the licence is set per repo (models differ: AGPL, GPL, MIT), `base_model` links, download
counts. App links pinned to commits (`resolve/<sha>/<file>`). First batch: only the 6 models already
in the app catalogue; no collaborator models. Rejected: one repo for all models (one licence field).

## 2026-10-15: HF layout: one repo per model family, file names by the app's naming rule
Repo = the app's model id without size or backbone, plus `-tflite` (`mdv6-tflite`, `flatbug-tflite`,
`insectdct-v8-tflite`, `insectdct-cls-v7-tflite`, `bioclip-2-tflite`, `bioclip-2.5-tflite`); sizes,
input sizes and precisions are files in it (flat-bug N and S share authors, licence and card). Files
are published under the naming rule (owner's choice; HF names can never change): 10 of 12 renamed,
same bytes. Rejected: one repo per size (repeated cards drift), old names (frozen forever).
`MDV6-yolov10-c_int8_256` is published as `mdv6-yolov10c_256_w8a32` because the file is w8a32.

## 2026-10-15: model cards: original authors first, the zoo only as the download tool
After the owner's request, every card names the original authors, paper, code and source weights
(SHA-256, byte-identical to the authors' files) near the top; the InsectAI Model Zoo appears only
where it was the download tool. Detector cards carry the owner's disclaimer (demo, not trained for
FaunaPulse). Each repo gets the original `LICENSE` (MIT, GPL and AGPL require the text with copies).
BioCLIP cards use `pipeline_tag: image-feature-extraction` (the file is the image tower only) with a
`zero-shot-image-classification` tag. Upload one repo per step (`--upload` takes one `--repo`).

## 2026-10-15: insectDCT class list rebuilt under the new name
Renaming only the files would break the pair in the app: the class list's header `model_id` must
equal the model file's stem. The network keeps its bytes (copied to the new name, reused by
`export_insectdct_cls.py`); the class list and manifest are written again by the script, so their
SHA-256 change and the app catalogue takes the new values. Rejected: keeping the old names for these
two files (breaks the naming rule), loosening the app check (older app versions keep the strict one).

## 2026-10-16: sources are the authors' own links, the zoo is not cited as a source
Owner's rule: the InsectAI Model Zoo collects other people's models, so saying files were
"downloaded with the zoo" blurred who made and distributes them. Every source file was downloaded
again from the authors' own links (ERDA, insectDCT GitHub and Google Drive, Imageomics on HF) and
was byte-identical; cards and docs now name only those links, and the zoo citation is gone.
`export_insectdct_cls.py` takes `--source-dir` (authors' zip, unpacked) and `--code-dir`
(insectDCT's `common/`); `--zoo-dir` stays as an alias. The class list was rebuilt from those files.

## 2026-10-16: no size limit for detectors
Owner's decision: detectors are no longer shipped inside the APK; they are downloaded from
Hugging Face or imported, and run live or on recorded video afterwards, so a large file is a
speed choice, not something to forbid. Here: `APP_MAX_BYTES` and its "larger than the app
accepts" note are removed from `export_detector.py`, and the docs drop "up to 30 MiB". The app
change (`kMaxTfliteModelBytes`) is for a separate app session; until then the app still rejects
detector files above 30 MiB.
