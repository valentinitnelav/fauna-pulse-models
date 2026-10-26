# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, codex, etc.) when working with code in this repository. CLAUDE.md points to this file.


## Start here (every session)

1. Read `docs/AGENT_OVERVIEW.md`: a short, state-only map of this repository (what each tool does,
   environment, where model files live, what is published on Hugging Face, the rules for model
   cards, what must stay in sync with the app). It is enough to start most tasks; open other files
   only as the task needs them.
2. Keep it current: when a change alters what is true, replace the affected text (no history
   there, history is git; keep it under ~50k characters). When the reason for a choice is not
   obvious, give it in one short clause next to the fact, or in a code comment.
3. No decision logs, journals or notes about the maintainer's thinking in tracked files: this
   repository and the app's are public. Reports and plans that are not public yet go in
   `private/` (git-ignored). This trickles down from the general rule of avoiding to share private information by accident.

This repository has a sibling, the app [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse/)
(usually cloned at `~/InsectDetectApp/fauna-pulse`). The app reads the files made here; the
"keep in sync" table in `docs/AGENT_OVERVIEW.md` lists every rule shared by both repositories.
The app repository has its own `AGENTS.md` with its own rules: follow them when editing there.
The owner's private calibration code base (`~/fauna-pulse-calibration`) imports a few modules of
`bioclip_export/` in place; `docs/AGENT_OVERVIEW.md` lists them, so change them only together.


## Project goal

Prepare models for deployment to the Android app [FaunaPulse](https://github.com/valentinitnelav/fauna-pulse/) (Flutter, Dart, Kotlin):
convert and quantize existing object detectors and identification models (classifiers, BioCLIP embedders), and train new detectors.
The end goal is to have `*.tflite` weights that can be deployed with LiteRT for inference on Android smartphones,
published with a model card each on Hugging Face (https://huggingface.co/Valen-Peregrine), from where the app downloads them.

The object detectors will be mostly single class, for example: detect pollinators on floral backgrounds.
Some of the object detectors (e.g. YOLO-nano) are needed for real-time inference, ideally running smooth at least at 5 FPS.
More heavy object detector architectures, like YOLO -small or medium can infer in post processing on the video data or imagery collected with FaunaPulse.
Quantization will aim for FP16 and INT8. INT8 requires calibration datasets and will be provided and updated by the maintainer.


## General rules

Less is more: the simplest solution is the better solution as long as core functionality is not lost
and as long as the code remains readable for humans too.

Comment code, especially when complex and focus on the why that code was implemented
especially when past alternatives were unsuccessfully tried and when there are side effects.

If encountered, never print, git commit secrets, digital keys, passwords and give a warning and/or ask for approval.
This stands valid for email addresses too.
Overall, do not place in any git tracked file any private or sensitive data.
If unsure, ask for permission from the maintainer.

Avoid user names if possible in file or folder paths and make use of `$HOME` in path name to indicate "home" without making use of user name.
The same holds for host names of workstations, phone serial numbers and names of collaborators' datasets or training runs:
use placeholders such as `/path/to/...` or `<user>@<workstation>` in tracked files.

**Models and data:**

- Some models belong to collaborators and must never be published or committed. Publish only models
  whose licence allows it and that the maintainer has cleared (the published list is in `docs/AGENT_OVERVIEW.md`).
- Never commit model weights or export outputs (`*/out/`, `*.tflite`, `*.pt`, ...; see `.gitignore`).
- Do not read into `~/InsectDetectApp/sessions/` unless asked (many large session logs); then ask
  first and only search by keyword, never read whole files.
- Ask before reading anything under the app's `assets/` folder.

**Git related:**

- Do not perform destructive Git operations without explicit approval.
- Do not `git commit` or `git push` changes unless requested by maintainer.
- When you edit, make sure to branch away from `main` branch, most probably into the active `develop`, then maintainer will merge to the main after review.
- Suggest commit messages and titles that are short, clear, relevant and easy to understand what the overall edits were about.
- Commit messages contain no author or attribution lines (no "Co-Authored-By", no tool or person names).

**Hugging Face:**

- Never replace, delete or squash a published model file, class list or label pack, or a repository's
  history: the app pins every download to a commit and checks the file's SHA-256. New weights get a new
  file name. Editing model cards is fine, and so is correcting a `.json` manifest (the app never
  downloads it; `publish/publish_hf.py --replace-manifests`).
- Uploading publishes: do it only after the maintainer reviewed the model card, and never without approval.


## Engineering workflow

Plan before making substantial changes.

Before reading very large files, please ask for permission or ask the maintainer where to find the needed information or if the needed information is located.

Try to recycle functionality whenever applicable. For example if same functionality appears across multiple scripts / modules, then generate a reusable "tool" for that particular functionality and mention that in its documentation.

Expect that the scripts facing the user to be executed from a terminal and adjust its documentation to be as clear, simple and easy to understand as possible without over-explaining and being verbose.

When testing code and needing to install dependencies and create python environments please try to check first if the needed dependencies are already part of the project's environments
(the shared one is `.venvs/export`, see README.md "Setup").


## Main operating system and programming languages

The usual operating system is Linux based (e.g Ubuntu, Kubuntu).

The code repository will also be cloned on a workstation with GPU, also Linux based operating system.
File and folder paths, therefore must be relative as much as possible so that the code runs on various computers.

Main language used will most probably be Python.
