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
