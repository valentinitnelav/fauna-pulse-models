# Models tried on the phone and parked

Two prompt-based models (you type what to find, e.g. "insect") were tried as slow "AI later"
detectors for FaunaPulse: SAM 3 (September 2026) and MiniCPM-V 4.7 (October 2026). Both are
parked because they are far too slow on today's phones. This page records what was measured,
so the same tests are not repeated without a new reason (a faster phone, a new model or a fix).

All phone numbers come from a **Xiaomi 11T Pro** (Snapdragon 888, Adreno 660 GPU, 8 GB RAM),
the strongest of the test phones. For scale: the app's live detectors aim at 5 or more frames
per second, and BioCLIP 2 identifies one crop in 0.27 s on this phone's GPU.

| Model | What it gives | Phone speed | Why parked |
|---|---|---|---|
| SAM 3 (Meta, SAM License), LiteRT fp16 conversion, 1008 px | every object that matches the prompt: boxes, scores, a "present in the picture" score | GPU: wrong numbers (NaN); CPU: **3.2 to 3.5 min per picture** | far too slow; the GPU route fails |
| EfficientSAM3 EV-M and TV-M (SAM 3 distilled into ~90 M parameters, Apache-2.0) | same as SAM 3 | not converted | tried on the PC only (4 to 6 s per picture); not converted for the phone |
| MiniCPM-V 4.7 1B (OpenBMB, Apache-2.0), llama.cpp | text answers about a picture; one rough box per question | CPU: **4 to 5 s** per question at 448 px, **22 s** at 896 px; no GPU or NPU route | seconds per frame, one box per prompt: not a detector |

## SAM 3

SAM 3 ("Segment Anything with Concepts", Meta; Carion et al. 2025, arXiv:2511.16719, SAM
License) finds every object that matches a short English prompt ("insect", "bee") in a picture,
without training on your data. It has about 840 million parameters, roughly 30 times BioCLIP's
work per picture, so it was only ever meant for saved videos, never for the live camera.

Files: the LiteRT conversion `mlboydaisuke/SAM3-LiteRT` on Hugging Face (fp16; the SAM License
allows use and redistribution with its licence text, and asks for "Built with SAM" in
publications).

| File | Size | Job |
|---|---|---|
| `sam3_vision.tflite` | 930 MB | picture (1008 x 1008) to picture features; the slow part |
| `sam3_text.tflite` | 607 MB | prompt to "text memory"; runs once per new prompt |
| `sam3_head.tflite` | 68 MB | features + text memory to 200 candidate boxes, their scores and a "prompt is in the picture" (presence) score |
| `sam3_token_embed.bin`, `vocab.json`, `merges.txt` | 103 MB | the tokenizer (prompt to numbers) |

A candidate's probability is its score times presence. There is no class list: every box is
named by the prompt.

**App code** (branch [`sam3`](https://github.com/valentinitnelav/fauna-pulse/tree/sam3) of the
app repository, tag `archive/sam3`): `Sam3Detector.kt` and `ClipTokenizer.kt` in the plugin,
a "What to find" field in "Find animals in videos", and the device check
`integration_test/sam3_check_test.dart` (phone against PC on one picture, with load
times, run times and memory; `--dart-define=SAM3_CPU=true` for the CPU route). PC tools in
`tool/sam3/` of that branch: `split_tflite.py` cuts the picture model into 4 parts between
transformer blocks, where only one tensor is passed on (identical numbers); `make_prompts.py`
stores encoded prompts so the text model runs only for a new one; `detect_video.py` runs SAM 3
(or EfficientSAM3) on the PC over the frames of an earlier video run and writes the app's
detection file.

**Measured on the phone:**

| | Result |
|---|---|
| Whole picture model on the GPU | Android closed the app while setting it up (about 5 GB); LiteRT 2.1.5 and 2.2.0 |
| Picture model in 4 parts on the GPU | loads in 42 s (app at 4.2 GB, each part keeps its own GPU memory); 9.6 s per picture; **all numbers NaN** from part 1 on, also with LiteRT 2.2.0, 32-bit sums or overflow clamping |
| Head on the CPU (4 threads) | 7.2 s per picture |
| Text model on the CPU | about 3 s, peak about 2 GB |
| Picture model and head on the CPU, one part at a time | correct (same results as on the PC); **3.2 to 3.5 min per picture** (each part 17 to 26 s to load and 17 to 32 s to run, head 18 to 20 s); app peak 4.4 GB, and Android closes background apps to make room |

About 80 s of each CPU picture is reloading the parts (the CPU library rearranges the weights
at every load). The model card verified a Pixel 8a (Mali GPU) and an iPhone; on the Adreno GPU
something in the first transformer blocks goes wrong, not located yet.

**PC:** about 40 s per picture on a 4-core laptop CPU, both with the LiteRT files and with
Meta's original `sam3.pt` run as in the InsectAI Model Zoo (Hugo Markoff, COST Action CA22129
InsectAI; github.com/HugoMarkoff/Insect_model_zoo; Ultralytics `SAM3SemanticPredictor`, 1008 px).

**EfficientSAM3** (Zeng et al. 2025, arXiv:2511.15833; Apache-2.0) is SAM 3 distilled into a
small picture model and a small text model (MobileCLIP-S0), keeping SAM 3's box-finding part:
EV-M (EfficientViT-B1, 89 million parameters) and TV-M (TinyViT-11M, 95 million), from
`Simon7108528/EfficientSAM3` (folder `efficientsam3_ft/`). Tried on the laptop only: about 4 s
(EV-M) and 5 to 6 s (TV-M) per picture; not converted for the phone. RV-M was not tried.

**Verdict.** If the GPU problem were solved: about 10 to 17 s per picture (picture model 9.6 s on
the GPU, head 7.2 s on the CPU), 1.7 GB of files and about 4.2 GB of memory, so a 1-minute clip
at 1 picture per second would take about 17 minutes. On the CPU today: about 3.3 minutes per
picture, or about 2 with the speed-up below, so 2 to 3 hours for the same minute. Enough to
re-check a few saved pictures, not to scan videos. Mammals were not tried: camera-trap detectors
already find them at YOLO speed.

**Cheapest first check on a new phone** (about 10 minutes): a debug build of branch `sam3`, the
files copied into the app's `files/sam3/` with `adb` and `run-as`, then `sam3_check_test.dart`,
first as it is (GPU), then with `--dart-define=SAM3_CPU=true`. To bring the branch up to date:
`git switch sam3 && git merge develop`.

**If reopened:**

1. Find the operation the Adreno GPU gets wrong: run picture part 1 in smaller pieces on the
   phone and compare each with the PC (its first steps are the "safe layer norm": scaling, sums
   over 1024 channels, square root). Ask the conversion's author whether Adreno was tested. The
   GPU would be about 20 times faster than the CPU route.
2. Speed up the CPU route: run each part over several pictures before loading the next (saves
   most of the 80 s of reloading), or let the CPU library keep its rearranged weights in a file
   (XNNPACK weight cache; needs a guard against a half-written file).
3. An import screen for the files instead of `adb`.

## MiniCPM-V 4.7

A vision-language model: it answers questions about a picture in text. Boxes come only from its
grounding prompt (`...<ref>insect</ref>`): one box per prompt, coordinates 0 to 1000, no score.

Phone setup: the llama.cpp build of OpenBMB's demo app (`MiniCPM-V-Apps`, fork
`tc-mb/llama.cpp-omni`), CPU, 4 threads; language part Q4_K_M (0.67 GB) and picture part f16
(1.11 GB). Loading takes about 5 s once.

| Input | Picture part | Language part | Per question |
|---|---|---|---|
| 1 tile (448 px) | 3.0 s | about 0.9 s | about 4 to 5 s |
| 896 px frame (default: 5 tiles) | 17.5 s | 2.8 s | about 22 s |
| 224 px crop (enlarged to 448 px) | 3.1 s | about 0.9 s | about 5 s |

- No faster route on this phone: llama.cpp's OpenCL GPU engine supports Adreno 750 and 830 and
  states that it does not work on Adreno 6xx; its Hexagon NPU engine needs Snapdragon 8 Gen 2
  or newer. A LiteRT GPU export of the picture part (as done for BioCLIP) was not tried;
  estimated 1.5 to 2 s per frame (not measured).
- Quality, on the PC (BF16), 115 frames of a public-domain clip with one honeybee on a pumpkin
  flower: "Is there an insect?" was answered correctly on 113 of 115 frames (no "yes" on the
  11 frames without the bee), also at one tile. Its boxes overlapped SAM 3's boxes (IoU of 0.5
  or more) in 46 % of the frames at the default setting and 72 % at the finest (run on the
  first 105 frames). With no insect in the picture it still returns a box covering nearly the
  whole picture, so a box is only usable after a "yes". The phone's answers matched the PC's.
- Not tested: small insects, several insects in one picture, other insects than a honeybee, and
  the 35B-A3B version (needs a phone with 12 GB RAM).
- Possible later role: a yes/no second opinion once per track or per photo session, not per
  frame. BioCLIP 2, already in the app, is the cheaper candidate for that job (prompts fixed
  on the PC instead of typed on the phone).

## When to look again

- a phone whose GPU computes SAM 3 correctly, or a LiteRT update that fixes the Adreno NaN;
- a phone with 12 GB RAM or more and a Snapdragon 8 Gen 2 or newer NPU (llama.cpp's Hexagon
  engine);
- a smaller prompt-based model that finds insects as well as SAM 3.
