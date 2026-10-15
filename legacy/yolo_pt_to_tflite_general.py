#!/usr/bin/env python3
"""
General Ultralytics YOLO .pt -> TFLite/LiteRT export tool.

The tool accepts:
  * any local Ultralytics-compatible YOLO .pt checkpoint path;
  * one or more input resolutions;
  * either a folder of calibration images, a text file listing images, or an
    existing Ultralytics dataset YAML;
  * INT8, FP16, or both export modes.

Important:
  * Raw PyTorch checkpoints from unrelated architectures are not supported.
    The checkpoint must be loadable with ultralytics.YOLO(...).
  * Bounding-box annotations are NOT required for INT8 calibration. Calibration
    runs representative images through the model to estimate activation ranges.
    If labels happen to exist, they are harmless but are not used by calibration.
  * nms=False and end2end=False are the defaults because many Android apps perform
    NMS themselves and GPU delegates are more likely to accept the plain detection
    graph.

Ultralytics compatibility:
  * Legacy TFLite export:
        format="tflite", half=True / int8=True
  * Current LiteRT export:
        format="litert", quantize=8 for static INT8
    Current LiteRT does not create a separate FP16 file; it exports FP32 and lets
    the GPU delegate execute it internally in FP16. In --api auto mode, this tool
    first tries to produce a true legacy FP16 TFLite file and falls back to the
    current FP32-for-GPU-FP16 LiteRT export if necessary.

Examples
--------
Export INT8 and FP16 at 320, 480, and 640:

    python yolo_pt_to_tflite_general.py \\
        --weights /path/to/model.pt \\
        --calib-source /path/to/calibration_images \\
        --imgsz 320 480 640 \\
        --mode both \\
        --calib-samples 300 \\
        --output-dir ./tflite_exports

Use a text file containing one image path per line:

    python yolo_pt_to_tflite_general.py \\
        --weights /path/to/model.pt \\
        --calib-source /path/to/images.txt \\
        --imgsz 640 \\
        --mode int8

Use an existing Ultralytics dataset YAML:

    python yolo_pt_to_tflite_general.py \\
        --weights /path/to/model.pt \\
        --data /path/to/dataset.yaml \\
        --imgsz 320 640 \\
        --mode both

Rectangular input is also accepted:

    python yolo_pt_to_tflite_general.py \\
        --weights /path/to/model.pt \\
        --calib-source /path/to/images \\
        --imgsz 384x640 \\
        --mode int8"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


ImageSize = Union[int, Tuple[int, int]]

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class Tee:
    """Mirror stdout/stderr to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def expand_path(value: str, base: Optional[Path] = None) -> Path:
    """Expand environment variables and '~', then resolve a path."""
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute() and base is not None:
        expanded = base / expanded
    return expanded.resolve()


def parse_imgsz_tokens(tokens: Sequence[str]) -> List[ImageSize]:
    """
    Parse values such as:
      --imgsz 320 480 640
      --imgsz 384x640
      --imgsz 320,480,640
    """
    raw_values: List[str] = []
    for token in tokens:
        raw_values.extend(part.strip() for part in token.split(",") if part.strip())

    parsed: List[ImageSize] = []
    seen = set()

    for raw in raw_values:
        normalized = raw.lower().replace(" ", "")
        if "x" in normalized:
            parts = normalized.split("x")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid resolution '{raw}'. Use INT or HEIGHTxWIDTH."
                )
            try:
                height, width = (int(parts[0]), int(parts[1]))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid resolution '{raw}'. Use INT or HEIGHTxWIDTH."
                ) from exc
            if height <= 0 or width <= 0:
                raise argparse.ArgumentTypeError("Image dimensions must be positive.")
            value: ImageSize = (height, width)
            key = (height, width)
        else:
            try:
                side = int(normalized)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid resolution '{raw}'. Use INT or HEIGHTxWIDTH."
                ) from exc
            if side <= 0:
                raise argparse.ArgumentTypeError("Image dimensions must be positive.")
            value = side
            key = (side, side)

        if key not in seen:
            parsed.append(value)
            seen.add(key)

    if not parsed:
        raise argparse.ArgumentTypeError("At least one input resolution is required.")
    return parsed


def size_tag(imgsz: ImageSize) -> str:
    if isinstance(imgsz, tuple):
        return f"{imgsz[0]}x{imgsz[1]}"
    return str(imgsz)


def normalize_names(names) -> Dict[int, str]:
    """Normalize Ultralytics model.names into an integer-keyed dictionary."""
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                idx = len(normalized)
            normalized[idx] = str(value)
        return dict(sorted(normalized.items()))

    if isinstance(names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(names)}

    return {0: "object"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def unique_existing_images(paths: Iterable[Path]) -> List[Path]:
    """Resolve, validate, and de-duplicate image paths while preserving order."""
    output: List[Path] = []
    seen = set()
    missing = 0
    unsupported = 0

    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            missing += 1
            continue
        if not is_image(resolved):
            unsupported += 1
            continue

        key = str(resolved)
        if key not in seen:
            output.append(resolved)
            seen.add(key)

    if missing:
        print(f"WARNING: ignored {missing} missing image path(s).")
    if unsupported:
        print(f"WARNING: ignored {unsupported} unsupported/non-image path(s).")

    return output


def read_image_list(list_file: Path) -> List[Path]:
    """
    Read one image path per line.

    Relative paths are resolved relative to the text file. Blank lines and lines
    beginning with '#' are ignored.
    """
    paths: List[Path] = []
    for line in list_file.read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        paths.append(expand_path(item, base=list_file.parent))
    return unique_existing_images(paths)


def collect_calibration_images(source: Path, recursive: bool = True) -> List[Path]:
    """Collect calibration images from a directory, text file, or single image."""
    if not source.exists():
        raise FileNotFoundError(f"Calibration source does not exist: {source}")

    if source.is_dir():
        iterator = source.rglob("*") if recursive else source.glob("*")
        return unique_existing_images(path for path in iterator if path.is_file())

    if is_image(source):
        return [source.resolve()]

    if source.suffix.lower() in {".txt", ".lst"}:
        return read_image_list(source)

    raise ValueError(
        "--calib-source must be an image directory, one image file, or a .txt/.lst "
        "file containing one image path per line. Use --data for a dataset YAML."
    )


def choose_sample(images: Sequence[Path], samples: int, seed: int) -> List[Path]:
    """Choose a deterministic sample without replacement; 0 means all images."""
    if samples < 0:
        raise ValueError("--calib-samples must be >= 0.")
    if not images:
        raise ValueError("No calibration images were found.")

    if samples == 0 or samples >= len(images):
        chosen = list(images)
        print(f"Calibration set: using all {len(chosen)} images.")
        return chosen

    rng = random.Random(seed)
    chosen = rng.sample(list(images), samples)
    print(
        f"Calibration set: sampled {len(chosen)} of {len(images)} images "
        f"without replacement (seed={seed})."
    )
    return chosen


def write_calibration_files(
    images: Sequence[Path],
    names: Dict[int, str],
    output_dir: Path,
    seed: int,
) -> Tuple[Path, Path]:
    """
    Write an absolute-path image manifest and a minimal Ultralytics dataset YAML.

    Labels are deliberately not generated: calibration consumes image tensors, not
    bounding boxes. Ultralytics may print a harmless missing-label warning.
    """
    calibration_dir = output_dir / "calibration_manifests"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    digest_input = "\n".join(str(path) for path in images).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:12]
    base_name = f"calibration_n{len(images)}_seed{seed}_{digest}"

    manifest_path = calibration_dir / f"{base_name}.txt"
    yaml_path = calibration_dir / f"{base_name}.yaml"

    manifest_path.write_text(
        "\n".join(str(path) for path in images) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Auto-generated by yolo_pt_to_tflite_general.py",
        "# Bounding-box labels are not required for INT8 calibration.",
        f"train: {json.dumps(str(manifest_path.resolve()), ensure_ascii=False)}",
        f"val: {json.dumps(str(manifest_path.resolve()), ensure_ascii=False)}",
        "names:",
    ]
    for idx, name in sorted(names.items()):
        lines.append(f"  {idx}: {json.dumps(name, ensure_ascii=False)}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Calibration manifest: {manifest_path.resolve()}")
    print(f"Calibration YAML:     {yaml_path.resolve()}")
    return manifest_path, yaml_path


def normalize_export_result(result, search_root: Path, precision_hint: str) -> Path:
    """Resolve the .tflite file returned by different Ultralytics versions."""
    raw = result
    if isinstance(raw, (tuple, list)) and raw:
        raw = raw[0]

    candidates: List[Path] = []

    if raw is not None:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()

        if candidate.is_file() and candidate.suffix.lower() == ".tflite":
            candidates.append(candidate)
        elif candidate.is_dir():
            candidates.extend(candidate.rglob("*.tflite"))

    candidates.extend(search_root.rglob("*.tflite"))

    unique: Dict[str, Path] = {}
    for path in candidates:
        if path.exists() and path.is_file():
            unique[str(path.resolve())] = path.resolve()
    candidates = list(unique.values())

    if not candidates:
        raise FileNotFoundError(
            "Ultralytics reported export completion, but no .tflite file could be found."
        )

    hint_terms = {
        "int8": ("int8",),
        "fp16": ("float16", "fp16"),
        "fp32_gpu_fp16": ("float32", "fp32"),
    }.get(precision_hint, ())

    def score(path: Path):
        name = path.name.lower()
        hint_score = sum(term in name for term in hint_terms)
        return (hint_score, path.stat().st_mtime_ns)

    return max(candidates, key=score)


def safe_link_or_copy(source: Path, destination: Path) -> None:
    """Create a symlink when possible; otherwise copy the checkpoint."""
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def copy_export(
    source: Path,
    destination: Path,
    overwrite: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}\n"
            "Use --overwrite or choose another --output-dir."
        )

    if destination.exists():
        destination.unlink()

    shutil.copy2(source, destination)
    return destination.resolve()


def run_export_attempt(
    yolo_class,
    weights: Path,
    task: Optional[str],
    export_kwargs: dict,
    output_dir: Path,
    destination: Path,
    precision_hint: str,
    overwrite: bool,
) -> Path:
    """
    Run one export in an isolated temporary directory.

    Keeping the source checkpoint in a temporary working directory prevents
    Ultralytics intermediate SavedModel/ONNX files from cluttering the checkpoint
    directory.
    """
    with tempfile.TemporaryDirectory(prefix=".yolo_export_", dir=output_dir) as tmp:
        work_dir = Path(tmp)
        work_weights = work_dir / weights.name
        safe_link_or_copy(weights, work_weights)

        model = yolo_class(str(work_weights), task=task) if task else yolo_class(str(work_weights))
        result = model.export(**export_kwargs)
        exported = normalize_export_result(result, work_dir, precision_hint)
        return copy_export(exported, destination, overwrite=overwrite)


def common_export_kwargs(args, imgsz: ImageSize) -> dict:
    return {
        "imgsz": imgsz,
        "batch": args.batch,
        "device": args.device,
        "nms": args.nms,
        "end2end": args.end2end,
    }


def export_int8(
    yolo_class,
    weights: Path,
    task: Optional[str],
    imgsz: ImageSize,
    calib_data: str,
    args,
    output_dir: Path,
) -> Tuple[str, Path, str]:
    """Export static INT8, trying current LiteRT and legacy TFLite APIs as requested."""
    base = common_export_kwargs(args, imgsz)
    size = size_tag(imgsz)
    destination = output_dir / f"{args.output_name}_{size}_int8.tflite"

    attempts = []
    if args.api in {"auto", "litert"}:
        attempts.append(
            (
                "current LiteRT static INT8",
                {
                    **base,
                    "format": "litert",
                    "quantize": 8,
                    "data": calib_data,
                    "fraction": args.fraction,
                },
            )
        )
    if args.api in {"auto", "legacy"}:
        attempts.append(
            (
                "legacy TFLite static INT8",
                {
                    **base,
                    "format": "tflite",
                    "int8": True,
                    "data": calib_data,
                    "fraction": args.fraction,
                },
            )
        )

    errors = []
    for label, kwargs in attempts:
        print(f"\n=== {label}: input {size} ===")
        try:
            path = run_export_attempt(
                yolo_class=yolo_class,
                weights=weights,
                task=task,
                export_kwargs=kwargs,
                output_dir=output_dir,
                destination=destination,
                precision_hint="int8",
                overwrite=args.overwrite,
            )
            return "int8", path, label
        except Exception as exc:
            errors.append((label, exc))
            print(f"WARNING: {label} failed: {exc}")
            if args.api != "auto":
                raise

    details = "\n".join(f"  - {label}: {exc}" for label, exc in errors)
    raise RuntimeError(f"All INT8 export methods failed:\n{details}")


def export_fp16(
    yolo_class,
    weights: Path,
    task: Optional[str],
    imgsz: ImageSize,
    args,
    output_dir: Path,
) -> Tuple[str, Path, str]:
    """
    Try a true legacy FP16 TFLite export.

    If unavailable in a current Ultralytics installation, export current LiteRT
    FP32. GPU delegates normally execute that model internally in FP16.
    """
    base = common_export_kwargs(args, imgsz)
    size = size_tag(imgsz)
    errors = []

    if args.api in {"auto", "legacy"}:
        label = "legacy TFLite true FP16"
        destination = output_dir / f"{args.output_name}_{size}_fp16.tflite"
        kwargs = {**base, "format": "tflite", "half": True}
        print(f"\n=== {label}: input {size} ===")
        try:
            path = run_export_attempt(
                yolo_class=yolo_class,
                weights=weights,
                task=task,
                export_kwargs=kwargs,
                output_dir=output_dir,
                destination=destination,
                precision_hint="fp16",
                overwrite=args.overwrite,
            )
            return "fp16", path, label
        except Exception as exc:
            errors.append((label, exc))
            print(f"WARNING: {label} failed: {exc}")
            if args.api == "legacy":
                raise

    if args.api in {"auto", "litert"}:
        label = "current LiteRT FP32 (GPU delegate may execute internally in FP16)"
        destination = output_dir / f"{args.output_name}_{size}_fp32_gpu_fp16.tflite"
        kwargs = {**base, "format": "litert", "quantize": 32}
        print(f"\n=== {label}: input {size} ===")
        try:
            path = run_export_attempt(
                yolo_class=yolo_class,
                weights=weights,
                task=task,
                export_kwargs=kwargs,
                output_dir=output_dir,
                destination=destination,
                precision_hint="fp32_gpu_fp16",
                overwrite=args.overwrite,
            )
            return "fp32_gpu_fp16", path, label
        except Exception as exc:
            errors.append((label, exc))
            print(f"WARNING: {label} failed: {exc}")
            if args.api == "litert":
                raise

    details = "\n".join(f"  - {label}: {exc}" for label, exc in errors)
    raise RuntimeError(f"All FP16-compatible export methods failed:\n{details}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to an Ultralytics-compatible YOLO .pt checkpoint.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        default=["320"],
        metavar="SIZE",
        help=(
            "One or more input sizes. Examples: --imgsz 320 480 640 or "
            "--imgsz 384x640. Default: 320."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["int8", "fp16", "both"],
        default="both",
        help="Export INT8, FP16-compatible, or both. Default: both.",
    )
    parser.add_argument(
        "--api",
        choices=["auto", "legacy", "litert"],
        default="auto",
        help=(
            "Ultralytics export API. auto tries compatible fallbacks; legacy uses "
            "format=tflite with half/int8; litert uses current quantize API. "
            "Default: auto."
        ),
    )

    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument(
        "--calib-source",
        "--calib-images",
        dest="calib_source",
        help=(
            "Calibration image directory, one image file, or .txt/.lst image list. "
            "Images are sampled deterministically. No annotations are required."
        ),
    )
    calibration.add_argument(
        "--data",
        help=(
            "Existing Ultralytics dataset YAML for INT8 calibration. When supplied, "
            "--calib-samples is not applied; use --fraction to use a subset."
        ),
    )

    parser.add_argument(
        "--calib-samples",
        type=int,
        default=300,
        help="Number of raw calibration images to sample; 0 uses all. Default: 300.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibration sampling. Default: 42.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction passed to Ultralytics after any explicit sampling. Must be "
            "in (0, 1]. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search calibration image directories recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default="./tflite_exports",
        help="Directory for final .tflite files, logs, and calibration manifests.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename prefix. Default: checkpoint stem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device accepted by Ultralytics. Default: cpu.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export/calibration batch size. Default: 1.",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "classify", "pose", "obb"],
        default=None,
        help="Optional task override if Ultralytics cannot infer it from the checkpoint.",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Bake NMS into the graph. Default is false and is recommended for the app.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Export an end-to-end/NMS-free model where supported. Default is false.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit log-file path.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable terminal-output logging.",
    )
    return parser


def run(args) -> None:
    weights = expand_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    if weights.suffix.lower() != ".pt":
        print(
            f"WARNING: weights file does not end in .pt: {weights}. "
            "Ultralytics may still load it, but this tool is intended for .pt checkpoints."
        )

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and at most 1.")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1.")

    args.imgsz = parse_imgsz_tokens(args.imgsz)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_name = args.output_name or weights.stem

    print("Command:", " ".join(sys.argv))
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Virtual environment:", os.environ.get("VIRTUAL_ENV"))
    print("Weights:", weights)
    print("Output directory:", output_dir)
    print("Input sizes:", ", ".join(size_tag(value) for value in args.imgsz))

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in this Python environment. "
            "Install a compatible version before running this tool."
        ) from exc

    print("Ultralytics version:", getattr(ultralytics, "__version__", "unknown"))

    # Load once to validate the checkpoint, discover the task, and read class names.
    probe_model = YOLO(str(weights), task=args.task) if args.task else YOLO(str(weights))
    detected_task = getattr(probe_model, "task", args.task)
    model_names = normalize_names(getattr(probe_model, "names", None))
    del probe_model

    print("Detected task:", detected_task)
    print(f"Classes ({len(model_names)}):", model_names)

    wants_int8 = args.mode in {"int8", "both"}
    wants_fp16 = args.mode in {"fp16", "both"}

    calibration_data: Optional[str] = None
    if wants_int8:
        if args.data:
            data_path = expand_path(args.data)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
            calibration_data = str(data_path)
            print("INT8 calibration: existing dataset YAML:", calibration_data)
        elif args.calib_source:
            if detected_task == "classify":
                raise ValueError(
                    "Raw label-free --calib-source handling is implemented for "
                    "detection-style datasets. For classification models, provide "
                    "an existing classification dataset using --data."
                )
            source = expand_path(args.calib_source)
            images = collect_calibration_images(
                source,
                recursive=not args.no_recursive,
            )
            chosen = choose_sample(images, args.calib_samples, args.seed)
            _, generated_yaml = write_calibration_files(
                images=chosen,
                names=model_names,
                output_dir=output_dir,
                seed=args.seed,
            )
            calibration_data = str(generated_yaml.resolve())
        else:
            raise ValueError(
                "INT8 export requires representative calibration images. Pass "
                "--calib-source PATH or --data DATASET.yaml."
            )

    results = []

    for imgsz in args.imgsz:
        if wants_fp16:
            results.append(
                export_fp16(
                    yolo_class=YOLO,
                    weights=weights,
                    task=args.task,
                    imgsz=imgsz,
                    args=args,
                    output_dir=output_dir,
                )
            )
        if wants_int8:
            results.append(
                export_int8(
                    yolo_class=YOLO,
                    weights=weights,
                    task=args.task,
                    imgsz=imgsz,
                    calib_data=calibration_data,
                    args=args,
                    output_dir=output_dir,
                )
            )

    print("\n--- Export results ---")
    for precision, path, method in results:
        size_mb = path.stat().st_size / 1_000_000
        print(f"{precision:17s} {size_mb:8.2f} MB  {path}")
        print(f"{'':17s} method: {method}")

    print(
        "\nCalibration labels were not required. Validate every exported model on a "
        "held-out, annotated test set because conversion and quantization can change "
        "precision, recall, and output behavior."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if not args.no_log:
        if args.log_file:
            log_path = expand_path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_path = output_dir / f"tflite_export_{stamp}.log"

        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, log_handle)
        print("Logging to:", log_path.resolve())

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nExport interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
