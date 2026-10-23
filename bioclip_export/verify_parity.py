#!/usr/bin/env python3
"""Check that the exported TFLite image tower matches the PyTorch model.

Why: a wrong resize, colour order or quantisation would silently lower accuracy on
the phone. This compares embeddings of real crops from both paths and reports the
cosine similarity (1.0 = identical direction). With a label pack it also reports
how often the two give the same answer at order, family, genus and species, taken the way
the app reports them (top-down: the best family inside the chosen order, and so on; the
probabilities of a taxon's names added up), and the largest change of the shown percentage.

Usage:
    python verify_parity.py --tflite out/bioclip-2_224_fp16.tflite --images ./crops [--pack out/x.fpack] [--limit 200]
    # crops of whole track IDs (--crops-csv: one row per crop with crop_id = file <crop_id>.png
    # in --images, kind = insect/background, seq_id = track ID, crop_px, new_filename = frame):
    # one random crop per track ID and one random square per insect-free frame, so long tracks
    # do not fill the sample
    python verify_parity.py --tflite ... --images DIR/crops --crops-csv DIR/crops.csv --pack ...

Acceptance (the bar set when BioCLIP was first brought to the phone): mean cosine >= 0.99
and >= 95 % family agreement.

Reading the numbers for a quantised file (quantise_tflite.py): the agreement is also shown
for the images where PyTorch itself is sure (50 % or more at that rank). Disagreements only
on unsure images are near-ties that small rounding flips; disagreements on sure images mean
the file is not fit for use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fpack import read_fpack

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_rgb01(path: Path, size: int) -> np.ndarray:
    """pybioclip's TreeOfLife transform: direct resize to size x size (no centre crop)."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0  # HWC 0..1


def torch_embed(model_key: str, images: list[np.ndarray], weights: Path | None = None) -> np.ndarray:
    import torch

    from export_image_tower import load_open_clip

    model = load_open_clip(model_key, weights)
    out = []
    with torch.no_grad():
        for hwc in images:
            x = torch.from_numpy(hwc).permute(2, 0, 1).unsqueeze(0)
            x = (x - torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD).view(1, 3, 1, 1)
            f = model.encode_image(x)
            out.append(torch.nn.functional.normalize(f, dim=-1)[0].numpy())
    return np.stack(out)


def tflite_embed(tflite_path: Path, images: list[np.ndarray]) -> tuple[np.ndarray, float]:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
    interp = Interpreter(model_path=str(tflite_path), num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]
    nchw = inp["shape"][1] == 3
    out, ms = [], []
    for hwc in images:
        x = hwc[None]
        if nchw:
            x = np.transpose(x, (0, 3, 1, 2))
        interp.set_tensor(inp["index"], x.astype(np.float32))
        t0 = time.time()
        interp.invoke()
        ms.append((time.time() - t0) * 1000)
        v = interp.get_tensor(outp["index"])[0].astype(np.float32)
        out.append(v / max(np.linalg.norm(v), 1e-9))
    return np.stack(out), float(np.median(ms))


def one_per_track(crops_csv: Path, images: Path, seed: int, below_px: int | None = None) -> tuple[list[Path], list[str]]:
    """One random crop per track ID (seq_id) and one random square per insect-free frame;
    with below_px only crops smaller than that (tracks without one left out, no squares)."""
    import pandas as pd

    from app_rules import APP_MIN_PX

    c = pd.read_csv(crops_csv, low_memory=False)
    ins = c[c.kind == "insect"]
    if below_px is not None:
        ins, c = ins[ins.crop_px < below_px], c[c.kind == "insect"]
    pick = pd.concat([ins.groupby("seq_id").sample(1, random_state=seed),
                      c[c.kind == "background"].groupby("new_filename").sample(1, random_state=seed)])
    groups = np.where(pick.kind == "background", "backgrounds",
                      np.where(pick.crop_px >= APP_MIN_PX, f"insects, {APP_MIN_PX} px or more",
                               f"insects, under {APP_MIN_PX} px"))
    return [images / f"{cid}.png" for cid in pick.crop_id], list(groups)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tflite", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder of crop images (jpg/png), searched recursively")
    ap.add_argument("--model", default=None, help="model key (default: read from the .json manifest next to the tflite)")
    ap.add_argument("--weights", type=Path, default=None,
                    help="checkpoint file already on disk (default: the Hugging Face download)")
    ap.add_argument("--pack", type=Path, nargs="+", default=[], help="label pack(s) to compare the answers per rank")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--crops-csv", type=Path, default=None,
                    help="one row per crop (crop_id, kind, seq_id, crop_px, new_filename): one crop per track ID from --images (no --limit)")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--below-px", type=int, default=None,
                    help="with --crops-csv: one crop smaller than this per track that has one (tiny crops are rare)")
    args = ap.parse_args()

    manifest_path = args.tflite.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    model_key = args.model or manifest.get("model_id", "bioclip-2")
    size = int(manifest.get("input_size", 224))

    if args.crops_csv is not None:
        paths, groups = one_per_track(args.crops_csv, args.images, args.seed, args.below_px)
    else:
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.limit]
        groups = ["all images"] * len(paths)
    if not paths:
        print("no images found", file=sys.stderr)
        return 2
    images = [load_rgb01(p, size) for p in paths]
    print(f"{len(images)} images, model {model_key}, input {size}")

    ref = torch_embed(model_key, images, args.weights)
    got, med_ms = tflite_embed(args.tflite, images)
    cos = np.sum(ref * got, axis=1)
    print(f"cosine similarity: mean {cos.mean():.4f}  min {cos.min():.4f}  (tflite median {med_ms:.0f} ms/crop on this PC)")
    ok = cos.mean() >= 0.99
    groups = np.array(groups)
    for pack in args.pack:
        from app_rules import EVAL, Taxonomy, softmax_rows

        hdr, mat = read_fpack(pack)
        scale = float(hdr.get("logit_scale", 100.0))
        tax = Taxonomy(hdr["labels"])
        # Softmax over the pack's names with the model's scale, then the app's top-down path
        # (Taxonomy.ladder_batch gives the same answers as the app's Dart code).
        k_ref, m_ref = tax.ladder_batch(softmax_rows(scale * (ref @ mat.T)))
        k_got, m_got = tax.ladder_batch(softmax_rows(scale * (got @ mat.T)))
        print(f"same answer as PyTorch per rank (pack {hdr['pack_id']}, scale {scale:.1f}); in brackets where "
              "PyTorch is sure (50 % or more), then the largest change of the shown percentage:")
        for g in sorted(set(groups)):
            sel = groups == g
            cells = []
            for k, name in EVAL.items():
                same = k_ref[sel, k] == k_got[sel, k]
                sure = np.nan_to_num(m_ref[sel, k]) >= 0.5
                d = np.abs(m_ref[sel, k] - m_got[sel, k])
                shift = np.nanmax(d) * 100 if np.isfinite(d).any() else 0.0
                cells.append(f"{name} {same.sum()}/{sel.sum()} ({same[sure].sum()}/{sure.sum()}, {shift:.2f} points)")
            print(f"  {g}: " + "; ".join(cells))
        ok = ok and np.mean(k_ref[:, 4] == k_got[:, 4]) >= 0.95
    print("PARITY OK" if ok else "PARITY FAILED (see thresholds in the docstring)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
