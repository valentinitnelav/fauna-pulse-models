# Legacy YOLO export scripts (June to August 2026)

**Superseded by [`../detector_export/`](../detector_export/README.md)**, which exports fp16,
w8a32, fp32 or full INT8 (with `--data`), turns segmentation models into box detectors and
checks every phone file against PyTorch. These scripts are kept as the record of how the early
phone files were made (June to August 2026, before `detector_export` existed).

They are not the exact recipe of the MegaDetector V6 file the app offers
(`MDV6-yolov10-c_int8_256.tflite`): in their kept form they pass options (`end2end`,
`format="litert"`, `quantize=8`) that the Ultralytics version recorded in that file (8.3.235)
does not accept. That file is also not full INT8: Ultralytics 8.3 names its dynamic-range output
(8-bit weights, float maths, w8a32) `_int8.tflite`, and the file's tensors confirm it. Check any
"int8" file before trusting the name (see `../docs/MODEL_CONVERSION.md`).

| Script | What it does |
|---|---|
| `pytorch_to_tflite_yolov11.py` | First export (fp16 with `nms=True`). Its baked-in NMS step did not compile on the phone GPU, see below. Kept for the record only. |
| `pytorch_to_tflite_int8_yolov11.py` | YOLOv11 `.pt` to GPU-friendly fp16 and/or INT8 (`nms=False, end2end=False`), INT8 calibrated on a seeded sample from a `.txt` list of image paths. |
| `yolo_pt_to_tflite_general.py` | The general version: any Ultralytics `.pt`, several input sizes in one run, calibration images from a folder, a `.txt` list or a dataset YAML, current LiteRT export with a fallback to the legacy TFLite export, a log file of every run. |

## Why `nms=False` (the reason these scripts were written)

The first export baked a Non-Max-Suppression (NMS) operator into the model graph. The
phone's LiteRT GPU delegate cannot compile that operator, so the model fell back to the
CPU (about 75 to 285 ms per frame) and the phone overheated into a frame-rate collapse.
FaunaPulse does NMS itself, so `nms=False, end2end=False` gives a plain detection graph
that does run on the GPU (about 20 to 45 ms per frame on the Xiaomi 11T Pro), the same
recipe as the Ultralytics plugin's bundled `yolo26n_int8.tflite`.

## Running (CPU is enough, no GPU needed)

The export, including INT8 calibration, runs on the CPU. `CUDA_VISIBLE_DEVICES=-1` forces
the CPU and avoids GPU start-up problems on a workstation; the exported file still runs
on the phone's GPU.

```bash
CUDA_VISIBLE_DEVICES=-1 python3 yolo_pt_to_tflite_general.py \
    --weights /path/to/runs/<run>/weights/best.pt \
    --calib-source /path/to/calibration_images \
    --imgsz 320 640 --mode both --calib-samples 300 --seed 42 \
    --output-dir ./tflite_exports
```

Environment: the shared one of this repository (see the [README](../README.md#setup)) has
Ultralytics and the LiteRT converter.

### Calibration images (INT8 only)

INT8 learns each activation's value range by running the model over a few hundred
**representative** images (your own insect and flower pictures, not a generic set such as
COCO). About 200 to 500 is plenty; all few thousand is slower with no real benefit. No
labels are needed (only the images go through the model). The scripts take a **seeded
random sample** (reproducible) and print, for example,
`Calibration set: sampled 300 of 4123 (seed=42)`. fp16 needs no images at all.

### Outputs

`yolo_pt_to_tflite_general.py` writes `<name>_<size>_int8.tflite` / `_fp16.tflite` into
`--output-dir`, plus the calibration manifest it sampled and a log file of the run.
`pytorch_to_tflite_int8_yolov11.py` leaves Ultralytics' `best_saved_model/best_float16.tflite`
and `best_int8.tflite` next to the weights. Rename the files by the app's naming rule
(`<model>_<input px>_<precision>.tflite`) before copying them to the phone.

## Checking on the phone

Import the file (home screen Menu, Download & import models, Import model files…), start a
session and check the session summary (or `logcat`): the inference engine should read
**GPU**, and the model time should be about 20 to 45 ms. fp16 usually keeps full accuracy;
INT8 is smaller and sometimes faster but can lose a little, so compare detections on your
own footage.

## Troubleshooting

- **Still CPU and slow for both fp16 and INT8**: the model's graph contains another
  operator the GPU delegate cannot compile (not NMS, not precision). Retraining on the
  YOLO26 architecture (as the bundled `yolo26n_int8`) solved it at the time.
- **INT8 export fails with a missing quantisation backend**: install `ai-edge-litert` and run again.
- **Class names look wrong in the app**: the scripts reopen the export and print its class
  names; they must match training, because the app reads them from the file's metadata.
