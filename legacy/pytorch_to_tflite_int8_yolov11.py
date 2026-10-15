"""
Convert a YOLOv11 PyTorch checkpoint (best.pt) to TFLite models that the LiteRT
**GPU** path on the phone can actually run — fp16 and/or int8.

WHY THIS SCRIPT EXISTS
----------------------
The earlier `pytorch_to_tflite_yolov11.py` exported with `nms=True`. That bakes a
Non-Max-Suppression operator into the model graph. On Android the Ultralytics YOLO
Flutter plugin does NMS itself on the *native* side, and — crucially — the baked-in
NMS ops are NOT supported by the LiteRT GPU delegate. On the Xiaomi this made the
GPU compile fail ("Following operations are not supported by GPU delegate ... Failed
to compile model") and the model fell back to the CPU, where it ran ~75-285 ms/frame
and overheated into a frame-rate collapse.

The plugin's own export guide (doc/models.md, scripts/export-tflite-models.py) shows
the correct, GPU-friendly recipe used for the bundled `yolo26n_int8.tflite` that DOES
run on the Xiaomi GPU at ~20-45 ms/frame:

    YOLO("model.pt").export(format="tflite", int8=True, data=<calib>, nms=False,
                            end2end=False, imgsz=...)

  * nms=False     -> no NMS operator in the graph (the app does NMS natively)
  * end2end=False -> plain detection head (GPU-convertible)
  * int8=True     -> full-integer quantization (smallest; the format the device's
                     LiteRT GPU/NPU path handled in our tests)
  * half=True     -> fp16 instead (NO calibration data needed; useful A/B test, and
                     often higher accuracy than int8 if the GPU compiles it)

ABOUT THE CALIBRATION DATA (int8 only) — answers "all images or a sample?"
--------------------------------------------------------------------------
Full-integer quantization rescales every activation to 8-bit, learning the ranges by
running the model over REPRESENTATIVE images ("calibration"). You do NOT need all your
training images — ~200-500 is plenty to capture the activation-range distribution, and
it exports much faster than running thousands. Use YOUR OWN insect/flower images, not a
generic set (coco128 etc.); calibrating on the wrong domain shifts the ranges and loses
accuracy. No labels are needed (only the images are run forward).

This script samples N images from your image-list .txt with a fixed random seed (so the
export is reproducible) and calibrates on exactly those. fp16 needs no calibration data.

HOW TO RUN (on the GPU workstation where the weights + images live)
-------------------------------------------------------------------
    cd /path/to/export_folder/
    source /path/to/export_env/bin/activate

    # BOTH fp16 and int8 in one go (int8 calibrated on a seeded 300-image sample):
    CUDA_VISIBLE_DEVICES=-1 python3 pytorch_to_tflite_int8_yolov11.py \
        --weights /path/to/runs/<run>/weights/best.pt \
        --calib-images /path/to/calibration/train_image_paths.txt \
        --imgsz 320 --mode both --calib-samples 300 --seed 42

    # Just one mode:  --mode fp16   (no calibration data needed)
    #                 --mode int8   (needs --calib-images or --data)
    # Use your real training YAML instead of sampling:  --data /path/dataset.yaml
    # Use ALL images for calibration:  --calib-samples 0
"""

import argparse
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path


class _Tee:
    """Write everything to several streams at once (used to mirror the terminal
    output into a log file), flushing each write so a crash still leaves a
    complete log."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def sample_image_list(image_list: Path, n: int, seed: int) -> Path:
    """Read an image-path .txt, optionally take a seeded random sample of [n]
    paths (n<=0 means use all), and write the selection to a new .txt. Returns
    the path to that selection (also kept for reproducibility/inspection)."""
    paths = [ln.strip() for ln in image_list.read_text().splitlines() if ln.strip()]
    if not paths:
        sys.exit(f"ERROR: no image paths found in {image_list}")
    total = len(paths)
    if n and n > 0 and n < total:
        rng = random.Random(seed)
        chosen = rng.sample(paths, n)  # without replacement, deterministic for a seed
        note = f"sampled {n} of {total} (seed={seed})"
    else:
        chosen = paths
        note = f"all {total} images"
    out = Path(tempfile.gettempdir()) / f"yolov11_calib_{len(chosen)}_seed{seed}.txt"
    out.write_text("\n".join(chosen) + "\n")
    print(f"Calibration set: {note} -> {out}")
    return out


def build_calibration_yaml(image_list: Path, names: dict) -> Path:
    """Write a minimal Ultralytics dataset YAML pointing train/val at [image_list],
    so int8 calibration runs on exactly those images. Returns the YAML path."""
    lines = [
        "# Auto-generated calibration dataset for int8 export.",
        f"train: {image_list.resolve()}",
        f"val: {image_list.resolve()}",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    out = Path(tempfile.gettempdir()) / "yolov11_int8_calibration.yaml"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Path to the trained best.pt")
    ap.add_argument("--imgsz", type=int, default=320, help="Inference size (match training; default 320)")
    ap.add_argument("--mode", choices=["int8", "fp16", "both"], default="both",
                    help="Which TFLite(s) to produce (default: both)")
    ap.add_argument("--data", default=None,
                    help="int8 calibration dataset YAML (e.g. the one you trained with); "
                         "overrides --calib-images/--calib-samples")
    ap.add_argument("--calib-images", default=None,
                    help="int8 fallback: a .txt of image paths to sample calibration images from")
    ap.add_argument("--calib-samples", type=int, default=300,
                    help="How many images to sample for int8 calibration (0 = use all; default 300)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the calibration sample (default 42)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Fraction of the (already-sampled) calibration set Ultralytics uses (default 1.0)")
    ap.add_argument("--device", default="cpu", help="Export device; 'cpu' is the safest (default)")
    ap.add_argument("--log-file", default=None,
                    help="Where to mirror all terminal output. Default: an auto-named, "
                         "timestamped file in the current directory.")
    ap.add_argument("--no-log", action="store_true", help="Disable writing the terminal log file")
    args = ap.parse_args()

    # Mirror everything printed below into a timestamped log file (default on), so
    # the full run can be saved and shared for diagnostics. Timestamp goes in the
    # name at creation time so repeated runs never overwrite each other.
    log_file = None
    if not args.no_log:
        if args.log_file:
            log_path = Path(args.log_file)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = Path.cwd() / f"tflite_export_log_{stamp}.txt"
        log_file = open(log_path, "w", buffering=1)  # line-buffered
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(f"Logging this run to: {log_path.resolve()}")

    try:
        _run(args)
    except SystemExit:
        raise  # validation exits already printed their message (and were logged)
    except BaseException:
        import traceback
        traceback.print_exc()  # write the full traceback into the log + terminal
        raise SystemExit(1)
    finally:
        if log_file is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()


def _run(args) -> None:
    print("Python executable:", sys.executable)
    print("Environment path:", os.environ.get("VIRTUAL_ENV"))
    print("Python version:", sys.version)

    import ultralytics
    from ultralytics import YOLO

    ultralytics.checks()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"ERROR: weights not found: {weights}")

    print(f"\nLoading model: {weights}")
    model = YOLO(str(weights))
    print(f"Classes ({len(model.names)}): {model.names}")

    want_int8 = args.mode in ("int8", "both")
    want_fp16 = args.mode in ("fp16", "both")

    # Resolve int8 calibration data once (if needed).
    calib_data = None
    if want_int8:
        if args.data:
            calib_data = args.data
            print(f"\nint8 calibration: using dataset YAML {calib_data}")
        elif args.calib_images:
            calib_list = Path(args.calib_images)
            if not calib_list.exists():
                sys.exit(f"ERROR: calibration image list not found: {calib_list}")
            sampled = sample_image_list(calib_list, args.calib_samples, args.seed)
            calib_data = str(build_calibration_yaml(sampled, model.names))
            print(f"int8 calibration YAML: {calib_data}")
        else:
            sys.exit("ERROR: int8 needs calibration data — pass --data <dataset.yaml> "
                     "or --calib-images <image_list.txt>.")

    common = dict(
        format="tflite",
        imgsz=args.imgsz,
        nms=False,        # plugin does NMS natively; keeps the graph GPU-convertible
        end2end=False,    # plain detection head
        device=args.device,
    )

    outputs = []
    # Export fp16 first (cheap, no calibration), then int8. Ultralytics writes
    # distinct filenames (*_float16.tflite vs *_int8.tflite), so both can coexist.
    if want_fp16:
        print("\n=== Exporting FP16 TFLite (no calibration data needed) ===")
        outputs.append(("fp16", Path(model.export(half=True, **common))))
        model = YOLO(str(weights))  # reload to avoid any export-state carryover

    if want_int8:
        print(f"\n=== Exporting INT8 TFLite (fraction={args.fraction}) ===")
        outputs.append(("int8", Path(model.export(int8=True, data=calib_data,
                                                   fraction=args.fraction, **common))))

    print("\n--- Results ---")
    for tag, path in outputs:
        print(f"  {tag}: {path.resolve()}  ({path.stat().st_size / 1e6:.1f} MB)")

    print(
        "\nNEXT STEPS\n"
        "  1. Copy BOTH .tflite files to the phone's custom-models folder (where\n"
        "     yolo11n_float16.tflite was placed) and give them distinct, clear names.\n"
        "  2. Test each in the app: start a session and CHECK the summary / logcat —\n"
        "     'Inference engine' should read GPU (not CPU) and inf_ms ~20-45 ms.\n"
        "  3. Compare accuracy (do the bee detections look right?) AND engine/inf_ms\n"
        "     between fp16 and int8; keep whichever runs on GPU with acceptable accuracy.\n"
        "  4. If BOTH still report CPU, the yolo11 op graph itself is GPU-incompatible\n"
        "     -> retrain on the YOLO26 architecture (same as the bundled yolo26n_int8);\n"
        "     this script then works unchanged on the v26 best.pt.\n"
    )


if __name__ == "__main__":
    main()
