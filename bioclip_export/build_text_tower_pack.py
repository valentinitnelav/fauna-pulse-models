#!/usr/bin/env python3
"""Build a label pack for another CLIP-type model with the SAME rows as an existing pack, by
running every row through that model's own text tower (for benchmark comparisons, not for
publishing).

Why: build_label_pack.py takes the ready-made TreeOfLife text vectors of BioCLIP 2 and 2.5. Other
models (BioCAP) have none, and a comparison needs exactly the rows of the reference pack, which a
new species list would not give (GBIF's Catalogue of Life release has moved since). So the rows
(the header's `labels`, sink rows included) are copied and only the vectors are new.

Text per row, following the zero-shot evaluation of BioCLIP and BioCAP (BioCAP repository,
train_and_eval/imageomics/naming_eval.py, text type `taxon_com`, and
evaluation/zero_shot_iid.py): the taxonomic name "Kingdom Phylum Class Order Family Genus
epithet" (empty slots left out), plus " with common name <common>" when the row has one, put into
each of OpenAI's 80 ImageNet templates ("a photo of a {}." ...); the row's vector is the mean of
the 80 unit vectors, made unit length again. --prompt sci_com uses "Genus epithet" instead of the
whole taxonomy (rows without a genus fall back to the taxonomic name); --prompt taxon leaves the
common name out. Sink rows ("none of these", kingdom `none`) are embedded from their own prompt
(the 8th slot) once, without templates, as build_label_pack.py does.

The header is the reference pack's, with model_id, logit_scale (the model's own), temperature 1.0,
pack_id, licence and a `text_vectors` record replaced; `rows_from` names the reference pack and
its SHA-256.

Usage (needs torch and open_clip; a GPU makes it minutes instead of hours):
  python build_text_tower_pack.py --rows-from REF.fpack --model biocap --out OUT.fpack
      [--prompt taxon_com|sci_com|taxon] [--pack-id ID] [--device cuda] [--batch 4096] [--limit N]
--limit embeds only the first N rows (smoke test; the pack is then not complete).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from export_image_tower import MODELS, load_open_clip
from fpack import read_fpack, write_fpack


def openai_templates():
    """OpenAI's 80 ImageNet prompt templates, as open_clip ships them."""
    try:
        from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES as T
    except ImportError:  # older open_clip
        from open_clip.zero_shot_metadata import openai_imagenet_template as T
    T = list(T)
    if len(T) != 80:
        sys.exit(f"expected 80 OpenAI templates, open_clip has {len(T)}")
    return T


def row_text(label: list[str], prompt: str) -> str:
    """The class name of one taxon row (BioCAP naming_eval.Taxon, empty slots left out)."""
    l = (list(label) + [""] * 8)[:8]
    upper = [s.capitalize() for s in l[:6] if s]
    taxon = " ".join(upper + ([l[6].lower()] if l[6] else []))
    name = taxon
    if prompt == "sci_com" and l[5]:
        name = " ".join([l[5].capitalize()] + ([l[6].lower()] if l[6] else []))
    common = l[7].strip()
    if prompt in ("taxon_com", "sci_com") and common:
        name += " with common name " + common
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-from", type=Path, required=True, help="reference pack whose rows are copied")
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--prompt", choices=["taxon_com", "sci_com", "taxon"], default="taxon_com")
    ap.add_argument("--pack-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4096, help="texts per forward pass")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N rows")
    a = ap.parse_args()

    import open_clip
    import torch

    ref_hdr, _ = read_fpack(a.rows_from)
    labels = ref_hdr["labels"]
    n = len(labels) if not a.limit else min(a.limit, len(labels))
    sink = [r for r in range(n) if labels[r][0] == "none"]
    taxa = [r for r in range(n) if labels[r][0] != "none"]
    templates = openai_templates()

    model = load_open_clip(a.model).to(a.device)
    tokenizer = open_clip.get_tokenizer(MODELS[a.model])
    logit_scale = float(model.logit_scale.exp().item())

    def encode(texts):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), a.batch):
                t = model.encode_text(tokenizer(texts[i:i + a.batch]).to(a.device))
                out.append(torch.nn.functional.normalize(t.float(), dim=-1))
        return torch.cat(out)

    dim = int(encode(["a photo of a bee."]).shape[1])
    mat = np.zeros((n, dim), np.float32)
    t0 = time.time()
    per_batch = max(1, a.batch // len(templates))  # rows per forward pass
    for i in range(0, len(taxa), per_batch):
        rows = taxa[i:i + per_batch]
        texts = [tpl(row_text(labels[r], a.prompt)) for r in rows for tpl in templates]
        v = encode(texts).view(len(rows), len(templates), dim).mean(1)
        mat[rows] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        if (i // per_batch) % 50 == 0:
            done = i + len(rows)
            el = time.time() - t0
            print(f"{done}/{len(taxa)} rows, {el:.0f} s, about {el / done * (len(taxa) - done):.0f} s left", flush=True)
    if sink:
        mat[sink] = encode([labels[r][7] for r in sink]).cpu().numpy()
    seconds = time.time() - t0

    # what describes the reference pack's own vectors goes under rows_from; with --limit the
    # per-row blocks (gbif, regions) would no longer fit the rows, so they are left out
    stale = ("format", "version", "rows", "dim", "dtype", "source", "built_with")
    hdr = {k: v for k, v in ref_hdr.items() if k not in stale and not (a.limit and k in ("gbif", "regions"))}
    hdr["labels"] = labels[:n]
    hdr["sink_rows"] = len(sink)
    hdr.update({
        "pack_id": a.pack_id or f"{a.model}_{ref_hdr.get('pack_id', a.rows_from.stem)}_{a.prompt}",
        "model_id": a.model,
        "logit_scale": logit_scale,
        "temperature": 1.0,
        "license": f"rows from {ref_hdr.get('pack_id')}; vectors made with the {a.model} text tower "
                   f"({MODELS[a.model]}), for benchmarks, not published",
        "rows_from": {"pack_id": ref_hdr.get("pack_id"), "file": a.rows_from.name,
                      "sha256": hashlib.sha256(a.rows_from.read_bytes()).hexdigest(),
                      "source": ref_hdr.get("source"), "built_with": ref_hdr.get("built_with")},
        "text_vectors": {"model": MODELS[a.model], "prompt": a.prompt,
                         "templates": "OpenAI ImageNet (80), mean of unit vectors; sink rows: own prompt, no templates",
                         "open_clip": open_clip.__version__, "torch": torch.__version__,
                         "device": torch.cuda.get_device_name(torch.device(a.device)) if a.device.startswith("cuda") else a.device,
                         "date": date.today().isoformat(), "seconds": round(seconds, 1),
                         "complete": n == len(labels)},
    })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_fpack(a.out, hdr, mat, dtype=ref_hdr.get("dtype", "f16"))
    print(f"{n} rows ({len(sink)} sink) x {dim} -> {a.out} in {seconds:.0f} s; logit_scale {logit_scale:.2f}; "
          + (f"example text: {templates[0](row_text(labels[taxa[0]], a.prompt))!r}" if taxa else "no taxon rows"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
