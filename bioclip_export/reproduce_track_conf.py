#!/usr/bin/env python3
"""Recompute FaunaPulse's per-track-id identification from a session's stored
embeddings, off the phone (round 219).

The app stores one embedding vector per crop in
<session>/identification/embeddings_<model>.{jsonl,bin}. This script applies
the same pooling rule as the app (see the app's docs/IDENTIFICATION.md, "How a track
id's Conf. is computed") against a label pack and prints, per track id, the
ladder with Conf., the plain mean, the maximum and the agreeing-crops mean of
the crops' own values, and the agreement share, so every number on the
results screen can be checked independently.

Rule (defaults = the app's defaults):
  p_i   = softmax(scale * M @ e_i)        per crop, scale = logit_scale / temperature
  w_i   = max(p_i)                        certainty weight (the crop's top-1 probability)
  counted_i = w_i >= max(w) / drop_factor (drop_factor <= 1: every crop counts)
  m     = sum(w_i e_i) / sum(w_i)         over counted crops, NOT re-normalised
  pbar  = softmax(scale * M @ m)          scored once; masses = species summed per taxon
  ladder: top-down best child of the chosen parent; identified rank = deepest
          rank with mass >= tau; none_p = mass of the "none of these" rows of pbar

Country list (--countries, the app's round 373 rule, track_fusion.dart Scorer.narrow): the
weights w_i, the drop rule and none_p stay over EVERY row of the pack; only what is shown (the
ladder, the identified rank, each crop's top-1 and own masses) is narrowed, for each crop's p_i
and for pbar alike: q = p on the sink rows, 0 on species rows outside the list, and
(1 - none) * p_i / sum(p over the listed species rows) on the listed ones (from the scores in
log space when that sum is below 1e-30). A list without a single row of the pack never narrows.

Orders and families (--taxa, the app's round 377 rule, name_list_region.dart resolveNameList and
label_pack.dart PackTaxa): a row (not a "none of these" row) has an order when its kingdom,
phylum, class and order are all filled in, and a family when the family is filled in too. The
chosen names that are an order or a family of some row are used (matching by exact name); a row
is kept when its order or its family is one of them. With --countries as well, only rows in both
are kept; when no row is in both, the whole pack is used (the app records `all`, `all` then).
No usable name: no narrowing. The scoring is the country rule above, with these rows as the
listed species rows.

Usage (from this folder):
  python3 reproduce_track_conf.py <session_dir> <pack.fpack> [--model STEM]
      [--tau 0.6] [--drop-factor 10] [--temperature T] [--countries DE+AT+...]
      [--taxa Apidae+Syrphidae] [--track ID] [--crops]

--countries takes what the app used, the last column of tracks_<pack>.csv
(`name_list_countries`: `DE+AT+...`, or `all` for the whole list), --taxa the very last one
(`name_list_taxa`: `Apidae+Syrphidae`, or `all`). Needs numpy and the sibling
fpack.py. Compare with <session>/identification/tracks_<pack>.json.
"""
import argparse
import json
import struct
import sys
from collections import defaultdict

import numpy as np

from fpack import bits_to_rows

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def load_pack(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"FPK1":
        sys.exit("not a label pack (missing FPK1 magic)")
    hlen = struct.unpack("<I", raw[4:8])[0]
    hdr = json.loads(raw[8:8 + hlen].decode())
    dim, rows = hdr["dim"], hdr["rows"]
    dtype = np.float32 if hdr.get("dtype", "f16") == "f32" else np.float16
    nbytes = rows * dim * np.dtype(dtype).itemsize
    matrix = np.frombuffer(raw[8 + hlen:8 + hlen + nbytes], dtype=dtype).astype(np.float32).reshape(rows, dim)
    labels = [(list(l) + [""] * 8)[:7] for l in hdr["labels"]]
    return hdr, matrix, labels


def load_embeddings(session_dir, stem):
    base = f"{session_dir}/identification/embeddings_{stem}"
    recs = [json.loads(l) for l in open(base + ".jsonl") if l.strip()]
    dim = next((r["dim"] for r in recs if r.get("type") == "identify_start" and "dim" in r), None)
    crops = [r for r in recs if r.get("type") == "crop"]
    vectors = np.fromfile(base + ".bin", dtype=np.float32)
    if dim is None:
        sys.exit("embedding dimension not found in the jsonl header")
    return crops, vectors.reshape(-1, dim)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def listed_rows(hdr, codes, sink):
    """Mask of the species rows GBIF lists in any of [codes] (the pack's `regions` block), or
    None = the whole pack: no codes, or a list without a single row (the app never narrows then)."""
    if not codes:
        return None
    countries = (hdr.get("regions") or {}).get("countries") or {}
    if not countries:
        sys.exit("--countries: this pack has no country lists (no `regions` block)")
    mask = np.zeros(hdr["rows"], bool)
    for c in codes:
        if c not in countries:
            print(f"note: the pack has no list for {c}; it adds no names (as in the app)")
            continue
        mask |= bits_to_rows(countries[c]["rows"], hdr["rows"])
    mask &= ~sink
    return mask if mask.any() else None


def taxa_rows(labels, names, sink):
    """(mask, used) for the app's PackTaxa.rowBits: rows whose order or family is one of the
    chosen [names]. mask None = no narrowing (no chosen name is an order or a family here)."""
    order = [l[3] if all(l[:4]) else "" for l in labels]
    family = [l[4] if all(l[:5]) else "" for l in labels]
    known = {o for o, s in zip(order, sink) if o and not s} | {f for f, s in zip(family, sink) if f and not s}
    used = sorted(set(n for n in names if n in known))
    for n in names:
        if n not in known:
            print(f"note: no order or family of this pack is named {n}; it adds no names (as in the app)")
    if not used:
        return None, []
    wanted = set(used)
    mask = np.array([not s and (o in wanted or f in wanted) for o, f, s in zip(order, family, sink)])
    return mask, used


def narrow(logits, whole, kept, sink):
    """The app's Scorer.narrow: [whole] = softmax(logits) over every row, narrowed to [kept]."""
    if kept is None:
        return whole
    out = np.zeros_like(whole)
    out[sink] = whole[sink]
    none = whole[sink].sum()
    listed = whole[kept].sum()
    if listed > 1e-30:
        out[kept] = whole[kept] * max(0.0, 1 - none) / listed
    else:  # every listed probability underflows: from the scores, in log space
        ex = np.exp(logits[kept] - logits[kept].max())
        out[kept] = ex * max(0.0, 1 - none) / ex.sum()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("pack")
    ap.add_argument("--model", default="bioclip-2_224_fp16", help="embeddings file stem (default: bioclip-2_224_fp16)")
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--drop-factor", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=None, help="override the pack's calibration temperature")
    ap.add_argument("--countries", default="all",
                    help="narrow to these country lists, as the app records them: DE+AT+... (or DE,AT); all = whole pack")
    ap.add_argument("--taxa", default="all",
                    help="narrow to these orders and families, as the app records them: Apidae+Syrphidae (or a,b); all = every one")
    ap.add_argument("--track", type=int, default=None, help="only this track id")
    ap.add_argument("--crops", action="store_true", help="also print the per-crop table")
    a = ap.parse_args()

    hdr, M, labels = load_pack(a.pack)
    temperature = a.temperature if a.temperature is not None else hdr.get("temperature", 1.0)
    scale = hdr["logit_scale"] / temperature
    keys = [["|".join(l[:k + 1]) if l[k] else None for k in range(7)] for l in labels]
    rows_under = [defaultdict(list) for _ in range(7)]
    for r, kk in enumerate(keys):
        for k in range(7):
            if kk[k]:
                rows_under[k][kk[k]].append(r)
    rows_under = [{key: np.array(v) for key, v in d.items()} for d in rows_under]
    sink = np.array([l[0] == "none" for l in labels])
    codes = [] if a.countries.strip().lower() == "all" else \
        [c.strip().upper() for c in a.countries.replace(",", "+").split("+") if c.strip()]
    in_countries = listed_rows(hdr, codes, sink)
    names = [] if a.taxa.strip().lower() == "all" else \
        [t.strip() for t in a.taxa.replace(",", "+").split("+") if t.strip()]
    in_taxa, used = taxa_rows(labels, names, sink)
    if in_countries is not None and in_taxa is not None:
        both = in_countries & in_taxa
        mask = both if both.any() else None  # no common names: the whole pack, as in the app
        if mask is None:
            print("note: no name is in both the countries and the taxa: the whole pack is used "
                  "(the app records `all`, `all`, `no_common_names`)")
    else:
        mask = in_countries if in_countries is not None else in_taxa
    kept = None if mask is None else np.nonzero(mask)[0]

    crops, E = load_embeddings(a.session_dir, a.model)
    by_track = defaultdict(list)
    for c in crops:
        by_track[c.get("track_id")].append(c)
    print(f"pack {hdr.get('pack_id')} ({len(labels)} names), scale {scale:.1f} (temperature {temperature}), "
          f"tau {a.tau}, drop factor {a.drop_factor}, {len(crops)} crops in {len(by_track)} track ids")
    if kept is None:
        offer = "the whole pack"
    else:
        within = [x for x, m in (("+".join(used), in_taxa), ("+".join(codes), in_countries)) if m is not None]
        offer = f"{len(kept)} species rows of {' within '.join(within)} (+ {int(sink.sum())} 'none of these' rows)"
    print("names on offer: " + offer)

    for tid in sorted(by_track, key=lambda t: (t is None, t)):
        if a.track is not None and tid != a.track:
            continue
        cr = by_track[tid]
        e = np.stack([E[c["row"]] for c in cr]).astype(np.float64)
        L = scale * (e @ M.T)  # each crop's scores against every row
        P_whole = np.stack([softmax(l) for l in L])
        w = P_whole.max(axis=1)  # weights and the drop rule: over every row, whatever the list
        counted = w >= w.max() / a.drop_factor if a.drop_factor > 1 else np.ones(len(cr), bool)
        # pooled scores = the weighted mean of the counted crops' scores (= scale * M @ m)
        pooled = ((w[counted, None] / w[counted].sum()) * L[counted]).sum(0)
        pbar_whole = softmax(pooled)
        none_p = float(pbar_whole[sink].sum())
        pbar = narrow(pooled, pbar_whole, kept, sink)
        P = np.stack([narrow(L[i], P_whole[i], kept, sink) for i in range(len(cr))])  # shown
        top1 = P.argmax(axis=1)

        def mass(p, key, k):
            return float(p[rows_under[k][key]].sum()) if key in rows_under[k] else 0.0

        # ladder: best child of the chosen parent
        ladder = []
        parent = ""
        for k in range(7):
            best_key, best = None, -1.0
            for key, idx in rows_under[k].items():
                if k > 0 and not key.startswith(parent + "|"):
                    continue
                v = float(pbar[idx].sum())
                if v > 0 and v > best:  # the app rolls up only rows with mass (rollUp)
                    best, best_key = v, key
            if best_key is None:
                break
            own = np.array([mass(P[i], best_key, k) for i in range(len(cr))])
            agree = np.array([keys[top1[i]][k] == best_key for i in range(len(cr))])
            ladder.append((RANKS[k], best_key.split("|")[-1] if k < 6 else " ".join(best_key.split("|")[5:7]),
                           best, own.mean(), own.max(), own[agree].mean() if agree.any() else 0.0, agree.mean()))
            parent = best_key
        identified = None
        for rank, _, p, *_ in ladder:
            if p >= a.tau and ladder[0][1] != "none":
                identified = rank
            else:
                break
        print(f"\n== track id {tid}: {len(cr)} crops, {int(counted.sum())} counted; identified rank: {identified}; "
              f"none_p {none_p:.4f}")
        print(f"   {'rank':8s} {'taxon':28s} {'Conf.':>6s} {'p_mean':>7s} {'p_max':>6s} {'p_agree':>8s} {'agree':>6s}")
        for rank, taxon, p, pm, px, pa, ag in ladder:
            print(f"   {rank:8s} {taxon[:28]:28s} {p*100:5.1f}% {pm*100:6.1f}% {px*100:5.1f}% {pa*100:7.1f}% {ag*100:5.0f}%")
        if a.crops:
            lkeys = []
            parent = ""
            for k, (_, taxon, *_rest) in enumerate(ladder):
                cand = [key for key in rows_under[k] if (k == 0 or key.startswith(parent + "|")) and key.split("|")[-1] == (taxon.split(" ")[-1] if k == 6 else taxon)]
                lkeys.append(cand[0] if cand else None)
                parent = lkeys[-1] or parent
            print(f"   {'no':>3s} {'counted':>7s} {'top-1 species':30s} {'top1_p':>7s} {'weight':>7s} {'none_p':>7s} "
                  + " ".join(f"{r[:5]:>6s}" for r, *_ in ladder) + "  photo")
            for i, c in enumerate(cr):
                own = " ".join(f"{mass(P[i], key, k)*100:5.0f}%" if key else "    -" for k, key in enumerate(lkeys))
                print(f"   {i+1:>3d} {'yes' if counted[i] else 'no':>7s} {' '.join(labels[top1[i]][5:7])[:30]:30s} "
                      f"{P[i].max()*100:6.1f}% {w[i]*100:6.1f}% {P_whole[i][sink].sum()*100:6.1f}% {own}  {c['src']}")


if __name__ == "__main__":
    main()
