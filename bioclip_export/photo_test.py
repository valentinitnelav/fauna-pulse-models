#!/usr/bin/env python3
"""Test label packs on photos the model cannot have seen in training, answered as the app answers.

For each species of a photo set (a CSV, see photo_sets/), the script downloads research-grade
iNaturalist photos uploaded after a date. The default is 2025-07-01: BioCLIP 2 was published in
May 2025, and its training data (TreeOfLife-200M) holds GBIF's occurrence snapshot of 2024-05-01
plus EOL images, so a photo uploaded later cannot be in it. For another model, give its own date.
Each photo is cut out two ways:
  - "photo": the whole photo as one box;
  - "detector": the best box of a detector (e.g. MegaDetector for mammals and birds).
Both are squared on the longer side and padded with the CLIP mean colour, as the app does
(crop_worker.dart). Each cut-out is embedded with BioCLIP (PyTorch) and scored against every
pack given, with the app's answer (app_rules.py): one softmax over the rows that compete, then
top-down (best order, best family inside it, ...), stopping at the deepest rank whose share
reaches --tau (the app's 60 %). Two ways of narrowing are scored:
  - "whole": every row of the pack;
  - "country": the set's country and its neighbours, the app's default when the session's
    country is known. The "none of these" rows keep their share of the softmax over
    the whole pack and the listed names share the rest, as in the app (track_fusion.dart).
    Only for packs with a regions block, and only for species that have a country in the set.

Every stage is kept in the work folder, so a rerun with another pack only scores again:
  photos/manifest.csv   one row per photo: species, observation, licence, attribution, upload date
  photos/*.jpg          iNaturalist "large" size (1024 px on the longer side)
  crops.csv             one row per photo and cut-out kind (box in photo pixels, detector score)
  embeddings_<model>.npz
  results_<pack>.csv    one row per cut-out and narrowing
  summary.md, summary.json

    python photo_test.py --set photo_sets/mammals_birds_common.csv --work out/photo_tests/mammals_birds \\
        --pack out/packs_v2/bioclip-2_mammals-birds-world_v2.fpack \\
        --pack out/packs_2026-11/bioclip-2_mammals-birds-world_v4.fpack \\
        --detector MDV6-yolov10-c.pt --detector-class animal

Photos are for testing only and stay in the work folder (licences CC0, CC BY and CC BY-NC by
default; the attribution of each photo is in the manifest). iNaturalist asks for at most about
one API request per second, so the script waits --sleep seconds between requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_rules import Taxonomy, softmax_rows  # noqa: E402
from fpack import bits_to_rows, read_fpack  # noqa: E402

INAT_API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "fauna-pulse-models photo_test.py"}
PAD = (123, 117, 104)  # the CLIP mean colour, as the app pads (crop_worker.dart kPadR/G/B)
RANK_NAMES = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def get(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read() if binary else json.load(r)


def read_set(path: Path) -> list[dict]:
    """The photo set: columns species (as the pack writes it, "Genus epithet"), and optional
    inat_name (when iNaturalist uses another name), country (ISO code for the narrowing) and group
    (a label for the summary)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("species") or "").strip() and not r["species"].startswith("#")]
    for r in rows:
        r["species"] = r["species"].strip()
        r["inat_name"] = (r.get("inat_name") or "").strip() or r["species"]
        r["country"] = (r.get("country") or "").strip().upper()
        r["group"] = (r.get("group") or "").strip() or "all"
    return rows


# ---------------------------------------------------------------- stage 1: photos

MANIFEST_COLS = ["species", "inat_name", "file", "observation_id", "photo_id", "licence", "attribution",
                 "observer_id", "uploaded", "observed", "url"]


def fetch_photos(sp: dict, have: list[dict], args, photo_dir: Path) -> list[dict]:
    """Up to --per-species photos of one species: one photo per observation, observers spread
    (each observer once before anyone twice), newest uploads first."""
    if len(have) >= args.per_species:
        return []
    q = {"taxon_name": sp["inat_name"], "quality_grade": "research", "photos": "true", "captive": "false",
         "created_d1": args.uploaded_after, "photo_license": ",".join(args.licences),
         "hrank": "species", "lrank": "subspecies", "per_page": 200, "order_by": "created_at", "order": "desc"}
    res = get(INAT_API + "?" + urllib.parse.urlencode(q)).get("results", [])
    time.sleep(args.sleep)
    seen_obs = {h["observation_id"] for h in have}
    by_user: dict[int, list[dict]] = defaultdict(list)
    for obs in res:
        tax = obs.get("taxon") or {}
        if " ".join(tax.get("name", "").split()[:2]) != sp["inat_name"] or str(obs["id"]) in seen_obs:
            continue  # another species, or one already taken
        photo = next((p for p in obs.get("photos", []) if p.get("license_code") in args.licences and p.get("url")), None)
        if photo is None:
            continue
        by_user[(obs.get("user") or {}).get("id", 0)].append({
            "species": sp["species"], "inat_name": sp["inat_name"], "observation_id": str(obs["id"]),
            "photo_id": str(photo["id"]), "licence": photo["license_code"], "attribution": photo.get("attribution", ""),
            "observer_id": str((obs.get("user") or {}).get("id", "")), "uploaded": obs.get("created_at", "")[:10],
            "observed": obs.get("observed_on") or "", "url": photo["url"].replace("/square.", "/large.")})
    users_have = {h["observer_id"] for h in have}
    queue = [c for u, cs in by_user.items() if str(u) not in users_have for c in cs[:1]]
    queue += [c for u, cs in by_user.items() if str(u) not in users_have for c in cs[1:]]
    queue += [c for u, cs in by_user.items() if str(u) in users_have for c in cs]
    new = []
    for c in queue:
        if len(have) + len(new) >= args.per_species:
            break
        c["file"] = f"{c['species'].replace(' ', '_')}_{c['observation_id']}_{c['photo_id']}.jpg"
        try:
            (photo_dir / c["file"]).write_bytes(get(c["url"], binary=True))
            new.append(c)
        except Exception as e:  # a missing photo: take the next one
            print(f"  skip {c['url']}: {e}")
        time.sleep(args.sleep)
    return new


def stage_photos(species: list[dict], args, work: Path) -> list[dict]:
    photo_dir = work / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    man_path = photo_dir / "manifest.csv"
    manifest = []
    if man_path.exists():
        with open(man_path, newline="", encoding="utf-8") as f:
            manifest = [r for r in csv.DictReader(f) if (photo_dir / r["file"]).exists()]
    for sp in species:
        have = [m for m in manifest if m["species"] == sp["species"]]
        if not args.offline:
            new = fetch_photos(sp, have, args, photo_dir)
            manifest += new
            have += new
        print(f"{sp['species']}: {len(have)} photos")
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(manifest)
    return manifest


# ---------------------------------------------------------------- stage 2: cut-outs

def square(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float]:
    """The app's crop of a box (margin 0): a square on the box's longer side, centred on the box."""
    side = max(x1 - x0, y1 - y0)
    return (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side


def cut(im, x: float, y: float, side: float, size: int):
    """The square, padded with the CLIP mean colour where it leaves the photo, resized as the app
    does: area average when shrinking, bilinear when enlarging."""
    from PIL import Image

    s = max(1, round(side))
    x, y = round(x), round(y)
    canvas = Image.new("RGB", (s, s), PAD)
    box = (max(0, x), max(0, y), min(im.width, x + s), min(im.height, y + s))
    canvas.paste(im.crop(box), (box[0] - x, box[1] - y))
    return canvas.resize((size, size), Image.BOX if s > size else Image.BILINEAR)


def stage_crops(manifest: list[dict], args, work: Path) -> list[dict]:
    from PIL import Image

    path = work / "crops.csv"
    crops: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            crops = list(csv.DictReader(f))
    # Detector boxes are kept only while the detector options stay the same.
    opts = {"detector": args.detector.name if args.detector else None, "class": args.detector_class,
            "conf": args.detector_conf, "size": args.detector_size}
    opts_path = work / "crops_options.json"
    if opts_path.exists() and json.loads(opts_path.read_text()) != opts:
        crops = [c for c in crops if c["kind"] == "photo"]
    opts_path.write_text(json.dumps(opts) + "\n")
    done = {(c["file"], c["kind"]) for c in crops}
    todo = [m for m in manifest if (m["file"], "photo") not in done]
    for m in todo:
        with Image.open(work / "photos" / m["file"]) as im:
            crops.append({"crop_id": f"{Path(m['file']).stem}__photo", "file": m["file"], "species": m["species"],
                          "kind": "photo", "x0": 0, "y0": 0, "x1": im.width, "y1": im.height, "score": ""})
    if args.detector:
        from ultralytics import YOLO

        model = YOLO(str(args.detector))
        want = [k for k, v in model.names.items() if v == args.detector_class]
        if not want:
            raise SystemExit(f"{args.detector.name} has no class {args.detector_class!r}: {model.names}")
        todo = [m for m in manifest if (m["file"], "detector") not in done and (m["file"], "no box") not in done]
        for i in range(0, len(todo), 16):
            part = todo[i:i + 16]
            res = model.predict([str(work / "photos" / m["file"]) for m in part], imgsz=args.detector_size,
                                conf=args.detector_conf, classes=want, verbose=False)
            for m, r in zip(part, res):
                b = r.boxes
                if b is None or len(b) == 0:
                    crops.append({"crop_id": f"{Path(m['file']).stem}__none", "file": m["file"], "species": m["species"],
                                  "kind": "no box", "x0": "", "y0": "", "x1": "", "y1": "", "score": ""})
                    continue
                j = int(b.conf.argmax())
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[j].tolist())
                crops.append({"crop_id": f"{Path(m['file']).stem}__det", "file": m["file"], "species": m["species"],
                              "kind": "detector", "x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1),
                              "y1": round(y1, 1), "score": round(float(b.conf[j]), 3)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "file", "species", "kind", "x0", "y0", "x1", "y1", "score"])
        w.writeheader()
        w.writerows(crops)
    return crops


# ---------------------------------------------------------------- stage 3: embeddings

def crop_key(c: dict, args) -> str:
    """Cache key of a cut-out's embedding: its id and box, so a new box is embedded again."""
    return f"{c['crop_id']}|{c['x0']},{c['y0']},{c['x1']},{c['y1']}"


def stage_embeddings(crops: list[dict], args, work: Path) -> dict[str, np.ndarray]:
    path = work / f"embeddings_{args.model}_{args.size}.npz"
    have: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path)
        have = dict(zip(z["ids"].tolist(), z["emb"]))
    todo = [c for c in crops if c["kind"] in ("photo", "detector") and crop_key(c, args) not in have]
    if todo:
        import torch
        from PIL import Image
        from export_image_tower import CLIP_MEAN, CLIP_STD, load_open_clip

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_open_clip(args.model).to(dev).eval()
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        for i in range(0, len(todo), 64):
            part, xs = todo[i:i + 64], []
            for c in part:
                with Image.open(work / "photos" / c["file"]) as im:
                    im = im.convert("RGB")
                    x, y, s = square(float(c["x0"]), float(c["y0"]), float(c["x1"]), float(c["y1"]))
                    xs.append(np.asarray(cut(im, x, y, s, args.size), dtype=np.float32) / 255.0)
            x = (torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2) - mean) / std
            with torch.no_grad():
                e = torch.nn.functional.normalize(model.encode_image(x.to(dev)).float(), dim=-1).cpu().numpy()
            have.update(zip((crop_key(c, args) for c in part), e))
        ids = sorted(have)
        np.savez(path, ids=np.array(ids), emb=np.stack([have[k] for k in ids]))
    return have


# ---------------------------------------------------------------- stage 4: scores

def shown(label: list[str]) -> str:
    return " ".join(x for x in label[5:7] if x) or next((x for x in reversed(label[:5]) if x), "")


def country_rows(hdr: dict, n: int, country: str) -> np.ndarray | None:
    reg = hdr.get("regions")
    if not reg or country not in reg.get("countries", {}):
        return None
    ok = np.zeros(n, bool)
    for c in [country] + reg.get("neighbours", {}).get(country, []):
        if c in reg["countries"]:
            ok |= bits_to_rows(reg["countries"][c]["rows"], n)
    ok[n - hdr.get("sink_rows", 0):] = True
    return ok


def score_pack(pack: Path, crops: list[dict], emb: dict, species: dict, left_out: set[str], args) -> list[dict]:
    hdr, W = read_fpack(pack)
    labels, sink = hdr["labels"], hdr.get("sink_rows", 0)
    n = len(labels)
    tax = Taxonomy(labels)
    names = [shown(l) if i < n - sink else "none: " + l[-1] for i, l in enumerate(labels)]
    rows_of = defaultdict(list)
    for i in range(n - sink):
        rows_of[names[i]].append(i)
    cs = [c for c in crops if crop_key(c, args) in emb]
    scale = float(hdr["logit_scale"]) / float(hdr.get("temperature", 1.0))  # as label_pack.dart
    logits = scale * np.stack([emb[crop_key(c, args)] for c in cs]) @ W.T
    species_rows = np.arange(n) < n - sink
    out = []
    for narrowing in ("whole", "country"):
        masks = {}
        for i, c in enumerate(cs):
            sp = species[c["species"]]
            p = softmax_rows(logits[i][None])[0]
            if narrowing == "country":
                if not sp["country"]:
                    continue
                if sp["country"] not in masks:
                    masks[sp["country"]] = country_rows(hdr, n, sp["country"])
                mask = masks[sp["country"]]
                if mask is None:
                    continue
                # The app's rule (track_fusion.dart, round 373): the "none of these" rows keep
                # their share of the softmax over every row; the listed species rows share the
                # rest, (1 - none) x a softmax over them; other rows get 0. Masking before the
                # softmax would let "none of these" grow as the list gets shorter.
                listed_rows = mask & species_rows
                q = softmax_rows(logits[i][listed_rows][None])[0]
                none = float(p[~species_rows].sum())
                p = np.where(species_rows, 0.0, p)
                p[listed_rows] = (1.0 - none) * q
            else:
                mask = None
            keys, mass = tax.ladder_batch(p[None])
            keys, mass = keys[0], mass[0]
            true_rows = rows_of.get(c["species"], [])
            listed = [r for r in true_rows if mask is None or mask[r]]
            # A species the pack lacks can still be right at genus and above: take those
            # ranks from the rows of its genus.
            truth_rows = true_rows or [r for r in range(n - sink) if labels[r][5] == c["species"].split()[0]]
            true_keys = [{int(tax.ids[k, r]) for r in truth_rows} for k in range(7)]
            if not true_rows:
                true_keys[6] = set()
            stop = max([k for k in range(7) if keys[k] >= 0 and mass[k] >= args.tau], default=-1)
            top = int(p.argmax())
            row = {"crop_id": c["crop_id"], "species": c["species"], "group": sp["group"], "kind": c["kind"],
                   "narrowing": narrowing, "rows_competing": int(mask.sum()) if mask is not None else n,
                   "in_pack": bool(true_rows), "on_list": bool(listed),
                   "top1": names[top], "top1_p": round(float(p[top]), 4), "top1_right": names[top] == c["species"],
                   "top1_left_out_elsewhere": names[top] in left_out,
                   "answer_rank": RANK_NAMES[stop] if stop >= 0 else "",
                   "answer": (" ".join(tax.names[stop][keys[stop]].split("|")[5:7]) if stop == 6
                              else tax.names[stop][keys[stop]].split("|")[-1]) if stop >= 0 else "",
                   "answer_share": round(float(mass[stop]), 4) if stop >= 0 else ""}
            for k in (3, 4, 5, 6):
                # a share of 0 is never right (the ladder can pass a child without mass)
                row[f"{RANK_NAMES[k]}_right"] = keys[k] >= 0 and mass[k] > 0 and int(keys[k]) in true_keys[k]
                row[f"{RANK_NAMES[k]}_share"] = round(float(mass[k]), 4) if keys[k] >= 0 else ""
            # the app's answer is right when it stops at order or below and that taxon is the true one
            row["answer_right"] = stop >= 3 and int(keys[stop]) in true_keys[stop]
            row["answer_species_right"] = stop == 6 and row["answer_right"]
            out.append(row)
    return out


def summarise(results: dict[str, list[dict]], crops: list[dict], manifest: list[dict], args) -> tuple[str, dict]:
    n_photos = len(manifest)
    no_box = sum(1 for c in crops if c["kind"] == "no box")
    lines = [f"# Photo test: {args.set.name}", "",
             f"{date.today().isoformat()}; model {args.model}; photos uploaded to iNaturalist after "
             f"{args.uploaded_after} (research grade, wild, licences {', '.join(args.licences)}); {n_photos} photos of "
             f"{len({m['species'] for m in manifest})} species; detector "
             f"{args.detector.name if args.detector else 'none'} ({no_box} photos without a box); stop rule {args.tau:.0%}.",
             "", "Columns: share of cut-outs whose answer is right. *Species*: the app names the right species; *answer "
             "right*: the app stops at some rank and that taxon is right; *genus/family/order*: the top-down path passes the "
             "right taxon; *top-1*: the single best row is the right species; *fossil top-1*: the best row is one another "
             "pack left out (extinct or fossil). *photo, boxed only*: the whole-photo cut-outs of the photos where the "
             "detector found a box, to compare both cut-outs on the same photos.", ""]
    table = {}
    head = "| Pack | Narrowing | Cut-out | Group | N | Species | Answer right | Genus | Family | Order | Top-1 | Fossil top-1 |"
    lines += [head, "|" + "---|" * 12]
    boxed = {Path(c["file"]).stem for c in crops if c["kind"] == "detector"}
    for pack, rows in results.items():
        groups = defaultdict(list)
        for r in rows:
            kinds = [r["kind"]]
            if r["kind"] == "photo" and boxed and r["crop_id"].rsplit("__", 1)[0] in boxed:
                kinds.append("photo, boxed only")
            for kind in kinds:
                groups[(r["narrowing"], kind, r["group"])].append(r)
                groups[(r["narrowing"], kind, "all")].append(r)
        for (nar, kind, grp), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "whole", kv[0][1], kv[0][2] != "all", kv[0][2])):
            if grp != "all" and len({r["group"] for r in rows}) == 1:
                continue
            k = len(rs)
            def share(col):
                return sum(1 for r in rs if r[col]) / k
            cells = {c: share(c) for c in ("answer_species_right", "answer_right", "genus_right", "family_right",
                                            "order_right", "top1_right", "top1_left_out_elsewhere")}
            table[f"{pack} | {nar} | {kind} | {grp}"] = {"n": k, **{c: round(v, 4) for c, v in cells.items()}}
            lines.append(f"| {pack} | {nar} | {kind} | {grp} | {k} | " + " | ".join(f"{cells[c]:.0%}" for c in cells) + " |")
    return "\n".join(lines) + "\n", table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True, help="photo set CSV (species, inat_name, country, group)")
    ap.add_argument("--work", type=Path, required=True, help="work folder (photos, cut-outs, embeddings, results)")
    ap.add_argument("--pack", type=Path, action="append", default=[], help="label pack to score (repeat)")
    ap.add_argument("--model", default="bioclip-2", help="model key of export_image_tower.py (default bioclip-2)")
    ap.add_argument("--size", type=int, default=224, help="input size of the image tower")
    ap.add_argument("--per-species", type=int, default=4)
    ap.add_argument("--uploaded-after", default="2025-07-01",
                    help="only photos uploaded to iNaturalist after this day (default: after BioCLIP 2 was published)")
    ap.add_argument("--licences", default="cc0,cc-by,cc-by-nc", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between iNaturalist requests")
    ap.add_argument("--offline", action="store_true", help="use only the photos already in the work folder")
    ap.add_argument("--detector", type=Path, default=None, help="Ultralytics detector weights for the 'detector' cut-out")
    ap.add_argument("--detector-class", default="animal")
    ap.add_argument("--detector-size", type=int, default=640)
    ap.add_argument("--detector-conf", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.6, help="the app's stop rule: deepest rank with at least this share")
    args = ap.parse_args()
    args.licences = [x.lower() for x in args.licences]

    species_list = read_set(args.set)
    species = {s["species"]: s for s in species_list}
    args.work.mkdir(parents=True, exist_ok=True)
    manifest = stage_photos(species_list, args, args.work)
    manifest = [m for m in manifest if m["species"] in species]
    crops = stage_crops(manifest, args, args.work)
    crops = [c for c in crops if c["species"] in species and c["file"] in {m["file"] for m in manifest}]
    emb = stage_embeddings(crops, args, args.work)

    # Names a pack left out (extinct or fossil rows): a name that pack still shows on another row
    # (e.g. a row named only "Bovidae") is not one of them.
    left_out: set[str] = set()
    for pack in args.pack:
        hdr = read_fpack(pack)[0]
        info = hdr.get("built_with", {}).get("rows_and_vectors_from", {}).get("species_rows_left_out", {})
        left_out |= {r[1] for r in info.get("rows", [])} - {shown(l) for l in hdr["labels"]}
    results = {}
    for pack in args.pack:
        rows = score_pack(pack, crops, emb, species, left_out, args)
        missing = sorted({r["species"] for r in rows if not r["in_pack"]})
        if missing:
            print(f"{pack.name}: not in the pack (only genus and higher can be right): {', '.join(missing)}")
        with open(args.work / f"results_{pack.stem}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        results[pack.stem] = rows
    text, table = summarise(results, crops, manifest, args)
    (args.work / "summary.md").write_text(text, encoding="utf-8")
    (args.work / "summary.json").write_text(json.dumps({
        "date": date.today().isoformat(), "set": args.set.name, "model": args.model, "uploaded_after": args.uploaded_after,
        "licences": args.licences, "photos": len(manifest), "detector": args.detector.name if args.detector else None,
        "tau": args.tau, "packs": [p.name for p in args.pack], "table": table}, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
