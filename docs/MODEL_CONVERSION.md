# Converting models for FaunaPulse

*For developers / collaborators who want their model to run inside the FaunaPulse app.*

FaunaPulse runs its models **on the phone itself** (no internet, no server), with LiteRT
(`.tflite` files). Which formats the app accepts, and why it does not run plain ONNX files
(plus the Snapdragon-only `*_qnn.onnx` exception), is explained in the app's
[`docs/MODEL_CONVERSION.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/docs/MODEL_CONVERSION.md). This page covers the conversion
itself; the tools in this repository do it for you (see the [README](../README.md)).

**Name your files by the project's naming rule** (round 276):
`<model>_<input px>_<precision>.tflite`, for example `my-bees-yolo11n_640_fp16.tflite`. The rule
and its reasons are in the app's [`tool/model_downloads/README.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/tool/model_downloads/README.md#naming-rule-round-276).
The app shows every file by its name; a model listed in the app's `assets/model_downloads.json`
(by the `<model>` part) also shows its licence and source.

## Converting: `.pt` → `.tflite` (the recommended route)

On any computer with Python, using the [Ultralytics package](https://docs.ultralytics.com/modes/export/):

```bash
pip install "ultralytics>=8.4.83"

# Recommended: ONE export, no dataset needed:
yolo export model=your_model.pt format=litert quantize=w8a32
```

This writes a `..._w8a32.tflite` file that imports straight into the app
(home screen Menu → Download & import models → Import model files…, or Download from a link…).

Why this is the recommended export (app support added in round 155):

- **No calibration dataset.** `w8a32` is "dynamic-range" quantization: the model's
  weights are stored as 8-bit integers while activations stay float32, so the
  exporter needs nothing but the `.pt` file.
- **Small and GPU-friendly.** It is the smallest format that still compiles on the
  phone-GPU delegate the app uses (this is Ultralytics' own default Android export).
- **Metadata included.** `format=litert` is a native Ultralytics export and embeds
  the class names and input size (the onnx2tf-based fallback below loses them).
- The app auto-detects this export's channel layout (NCHW `[1,3,H,W]`, different
  from the older exports' `[1,H,W,3]`), so `format=litert` and older
  `format=tflite` files both work in the same picker.

### Models by other authors (flat-bug, insectDCT), segmentation models, input sizes (round 264)

[`detector_export/export_detector.py`](../detector_export/export_detector.py) does the export above for any Ultralytics
checkpoint, turns segmentation models (flat-bug) into box detectors, writes one file per
input size and checks every file against PyTorch on your pictures
([`detector_export/README.md`](../detector_export/README.md), with phone timings of flat-bug n/s and insectDCT v8-s).
Its default is fp16 weights, not w8a32: on flat-bug n at 640 px, w8a32 lost 6 of 28 boxes
near the confidence threshold while fp16 kept all 28 with the same confidences; on the
Xiaomi's GPU both run equally fast, on its CPU w8a32 was 2.3 times faster. So w8a32 stays a
good choice for CPU-bound phones, after checking its boxes.

Classifiers (not detectors) such as insectDCT's hierarchical classifier have their own
script, [`classifier_export/`](../classifier_export/README.md) (round 265; the phone file plus its class list, which the
Identify screen uses since round 266).
It also shows two GPU-friendly rewrites with the same arithmetic (a pooling the converter
wrote with GATHER_ND, a max-pool padding with minus infinity): when a converted file runs
on the phone's CPU only, the phone log line "Following operations are not supported by GPU
delegate" names the step to rewrite.

### Older / alternative exports (all still work)

```bash
# Float32 - no dataset needed, most accurate, largest/slowest:
yolo export model=your_model.pt format=tflite

# Float16 ("half") - no dataset needed, half the size:
yolo export model=your_model.pt format=tflite half=True

# Full INT8 - smallest & fastest on CPU, needs a calibration dataset (see below):
yolo export model=your_model.pt format=tflite int8=True data=your_data.yaml
```

Each `format=tflite` command writes a `..._saved_model/` folder containing the
`.tflite` file.

**Check what an "int8" file really is.** Older Ultralytics versions (8.3, used for the first
MegaDetector V6 file in August 2026) name the *dynamic-range* file (8-bit weights, float maths,
i.e. w8a32) `..._int8.tflite`; the fully 8-bit one gets another name (`..._integer_quant` or
`..._full_integer_quant`). So that MegaDetector file is w8a32, although it was called int8.
`detector_export/export_detector.py --quantize int8 --data` (Ultralytics 8.4.170) does write a
full-INT8 file (checked 2026-10-15: 301 of the 302 intermediate tensors are 8-bit). To check a
file, count the 8-bit tensors that operations produce: a w8a32 file has none.

### Notes:

1. **Full INT8 specifically needs calibration images.** INT8 stores the model's
   numbers (weights AND activations) as 8-bit integers, and the exporter must see a
   few hundred *representative* images (your training/validation data) to choose the
   value ranges. Calibrating on unrelated images (e.g. generic COCO photos) can
   noticeably hurt accuracy.
2. **w8a32, fp32 and fp16 exports do not need calibration images**, just the `.pt` file.
3. An ONNX export is a float model too, and converting it wouldn't dodge anything.

#### Why full INT8 still matters (low-end phones)

On phones whose GPU can't run a given model, everything runs on the CPU, and there
full INT8 is faster than the float-activation formats (w8a32 included).
For example, measured in FaunaPulse on my slower test phone (Samsung Galaxy M12 class),
an INT8 YOLO26-nano model (see asset `yolo26n_int8.tflite` in 
[v0.3.5 of ultralytics/yolo-flutter-app](https://github.com/ultralytics/yolo-flutter-app/releases?page=2#release-v0.3.5)) 
allowed a 2-3 fps and that is already rather slow for accurate tracking. 
Therefore, on low-end or older phones, full INT8 might be the only practical option.

Nuance for stronger phones: whether a model runs on the GPU is decided by whether the
GPU backend can compile the model's operations *not* by its float precision. INT8 models
can and do run on the GPU too, and fp16 is perfectly fine there. So when in doubt,
export w8a32 plus a full-INT8 variant. The FaunaPulse app has a built-in benchmark
(Settings → Detection → "Benchmark engines") that times them on the phone.

### Models other than YOLO (classifiers, embedders)

A model that is not an Ultralytics YOLO (for example a classifier converted to a float32
`.tflite` with litert-torch) can be quantised with [`bioclip_export/quantise_tflite.py`](../bioclip_export/quantise_tflite.py)
(fp16, int8 or int8 weights only, plus a quick check). Its comments and
[`bioclip_export/README.md`](../bioclip_export/README.md) §2b explain which kind is faster on a GPU or a CPU, with
the numbers measured on BioCLIP 2.

## Converting: `.onnx` → `.tflite` (when the `.pt` is truly unavailable)

Possible with [onnx2tf](https://github.com/PINTO0309/onnx2tf), the same tool
Ultralytics uses internally, but fussier than exporting from `.pt` (channel-layout
transposes, occasional unsupported operations, and the app expects the standard
Ultralytics YOLO output layout). Treat it as the fallback, not the plan:

```bash
pip install onnx2tf onnx onnx_graphsurgeon sng4onnx
onnx2tf -i your_model.onnx -o converted_model
```

Note that a float `.tflite` produced this way still lacks the Ultralytics metadata
(class names, input size) a native export embeds. The app copes (it reads the input
size from the tensor shape), but class names may be missing. One more reason to
prefer the `.pt` route.

## Quick checklist

1. Preferably share the `.pt`; otherwise run the exports yourself and share `.tflite`.
2. Export **w8a32** (`format=litert quantize=w8a32`, one command, no dataset). Add a
   full-INT8 variant (calibrated on your own data) when low-end phones matter.
3. Keep the input size modest (640 or smaller, e.g. 480/416/320 px) because preprocessing and
   inference cost scale with input size squared, and phones throttle with heat.
4. Only bother with `format=qnn` if it targets a specific Snapdragon phone.
