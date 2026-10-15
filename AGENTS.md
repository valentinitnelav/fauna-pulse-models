# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, codex, etc.) when working with code in this repository. CLAUDE.md points to this file.


## Project goal

Train object detectors for deployment to the Android app [FaunPulse](https://github.com/valentinitnelav/fauna-pulse/) (Flutter, Dart, Kotlin) and / or quantize existing ones.
The end goal is to have `*.tflite` weights that can be deployed with LiteRT for inference on Android smartphones.

The object detectors will be mostly single class, for example: detect pollinators on floral backgrounds.
Some of the object detectors (e.g. YOLO-nano) are needed for real-time inference, ideally running smooth at least at 5 FPS.
More heavy object detector architectures, like YOLO -small or medium can infer in post processing on the video data or imagery collected with FaunPulse.
Quantitation will aim for FP16 and INT8. INT8 requires calibration datasets and will be provided and updated by the maintainer.


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

**Git related:**

- Do not perform destructive Git operations without explicit approval.
- Do not `git commit` or `git push` changes unless requested by maintainer.
- When you edit, make sure to branch away from `main` branch, most probably into the active `develop`, then maintainer will merge to the main after review.
- Suggest commit messages and titles that are short, clear, relevant and easy to understand what the overall edits were about.


## Engineering workflow

Plan before making substantial changes.

Before reading very large files, please ask for permission or ask the maintainer where to find the needed information or if the needed information is located.

Try to recycle functionality whenever applicable. For example if same functionality appears across multiple scripts / modules, then generate a reusable "tool" for that particular functionality and mention that in its documentation.

Expect that the scripts facing the user to be executed from a terminal and adjust its documentation to be as clear, simple and easy to understand as possible without over-explaining and being verbose.

When testing code and needing to install dependencies and create python environments please try to check first if the needed dependencies are already part of the project's environments (often located in local `.venv` folders).


## Main operating system and programming languages

The usual operating system is Linux based (e.g Ubuntu, Kubuntu).

The code repository will also be cloned on a workstation with GPU, also Linux based operating system.
File and folder paths, therefore must be relative as much as possible so that the code runs on various computers.

Main language used will most probably be Python.
