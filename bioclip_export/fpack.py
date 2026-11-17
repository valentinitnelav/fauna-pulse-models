"""FaunaPulse label-pack container ("fpack"): one file = header JSON + embedding matrix.

A label pack is what the phone compares a crop's image embedding against. It holds one
L2-normalised text embedding per candidate name (species rows from the TreeOfLife
embeddings, plus a few "none of these" sink rows) and the taxonomy of every row.

Layout (all little-endian):
    bytes 0..3      ASCII magic "FPK1"
    bytes 4..7      uint32 header length H (bytes)
    bytes 8..8+H    UTF-8 JSON header (see write_fpack for the keys)
    then            rows x dim numbers, row-major, dtype "f16" or "f32" per header

The Dart reader lives in the app repo (fauna-pulse): lib/fauna_pulse/identification/label_pack.dart.
Both sides are covered by the app's test/fauna_pulse/label_pack_test.dart through the fixtures
this module writes, so a format change here needs the Dart reader and the fixtures updated too.
Only numpy is needed here, so the format can be tested without torch.

Optional header blocks (since 2026-10-27, written by build_label_pack.py and add_pack_blocks.py;
"version" stays 1 because the byte layout is unchanged). Readers that do not know them ignore
them, so older app versions load such a pack as before, as long as the whole header stays
within the app's limits (16 MB; check_pack_blocks.app_accepts_header, run by both writers):
    "gbif"     GBIF species of every row, as lists with one entry per row (null for sink rows
               and rows without one): "species_key" (Catalogue of Life Extended Release key,
               a string, some digits only: never turn it into a number) and "species" (its
               accepted name), plus "source" (the mapping file and COL release they come from).
               Rows listed in "name_rules" carry the key of their published name, for the
               country lists; the app answers them only at genus or family, so it is not a
               track's species there.
    "regions"  which rows GBIF lists per country: "countries" = {ISO 3166-1 alpha-2 code:
               {"rows": rows_to_bits(...), "names": distinct names among the rows set (a name in
               two families has two rows), "names_in_country": names of the pack's orders that
               the country's list holds in total}}, "neighbours" = {code:
               [codes]} with the rule in "neighbours_rule", "min_occurrences", "source".
               Sink rows are never set. A row whose records are spread over several countries
               with fewer than min_occurrences in each is in no country: only the whole pack has it.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from pathlib import Path

import numpy as np

MAGIC = b"FPK1"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
_PLAIN_NAME = re.compile(r"[A-Z][a-z]+ [a-z][a-z-]+")  # "Genus epithet"


def write_raw(path: Path, header: dict, payload: bytes) -> None:
    """Write [header] as the pack's JSON header, then [payload] unchanged (the matrix bytes)."""
    hdr_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(payload)


def read_raw(path: Path) -> tuple[dict, bytes]:
    """(header dict, matrix bytes untouched), for copying a pack's vectors bit for bit."""
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{path} is not an fpack file (bad magic)")
    (hlen,) = struct.unpack("<I", data[4:8])
    return json.loads(data[8:8 + hlen].decode("utf-8")), data[8 + hlen:]


def write_fpack(path: Path, header: dict, matrix: np.ndarray, dtype: str = "f16") -> None:
    """Write [matrix] (rows x dim, any float dtype) with [header] into [path].

    Rows are re-normalised to unit length so the phone can use plain dot products.
    The header gets "rows", "dim" and "dtype" filled in; everything else is the caller's.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D (rows x dim), got shape {matrix.shape}")
    mat = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    if dtype == "f16":
        payload = mat.astype("<f2").tobytes()
    elif dtype == "f32":
        payload = mat.astype("<f4").tobytes()
    else:
        raise ValueError("dtype must be 'f16' or 'f32'")
    hdr = dict(header)
    hdr.update({"format": "fpack", "version": 1, "rows": int(mat.shape[0]),
                "dim": int(mat.shape[1]), "dtype": dtype})
    labels = hdr.get("labels")
    if labels is None or len(labels) != mat.shape[0]:
        raise ValueError("header['labels'] must have one entry per matrix row")
    write_raw(path, hdr, payload)


def write_class_list(path: Path, header: dict) -> None:
    """Write a class list (round 266): the label pack of a fixed-class classifier such as
    insectDCT. No name vectors: the model itself scores every class. The header holds, per
    row (finest class), its taxonomy in "labels" and the model's own name in "classes", plus
    "heads" (the model's output layers: [{"name", "size", "classes"}], one after another in
    the model's output) and "head_index" (per row, its class in every head). "dim" is the
    model's output length, so the app's size check pairs model and class list.
    """
    hdr = dict(header)
    rows = len(hdr["labels"])
    sizes = [h["size"] for h in hdr["heads"]]
    if len(hdr["classes"]) != rows or len(hdr["head_index"]) != rows:
        raise ValueError("labels, classes and head_index need one entry per row")
    for idx in hdr["head_index"]:
        if len(idx) != len(sizes) or any(not 0 <= i < n for i, n in zip(idx, sizes)):
            raise ValueError(f"head_index entry {idx} does not fit the heads {sizes}")
    # "class_list" since round 276 (the word of the app's assets/model_downloads.json; the app also
    # reads the older "classes").
    hdr.update({"format": "fpack", "version": 1, "kind": "class_list", "rows": rows,
                "dim": int(sum(sizes)), "dtype": "none"})
    write_raw(path, hdr, b"")


def class_probabilities(scores, head_sizes, head_index, heads=None) -> np.ndarray:
    """One probability per class-list row from a classifier's raw output (the rule the app
    uses, track_fusion.dart): each head's scores become log-probabilities (log-softmax); a
    row scores the mean of its classes' log-probabilities over the heads used (default: all),
    so a fine class only scores high when every level agrees and K heads do not count as K
    independent votes; then one softmax over the rows."""
    scores = np.asarray(scores, dtype=np.float64)
    use = range(len(head_sizes)) if heads is None else heads
    logp, start = [], 0
    for n in head_sizes:
        h = scores[start:start + n]
        m = h.max()
        logp.append(h - m - np.log(np.exp(h - m).sum()))
        start += n
    s = np.array([np.mean([logp[k][idx[k]] for k in use]) for idx in head_index])
    e = np.exp(s - s.max())
    return e / e.sum()


def read_fpack(path: Path) -> tuple[dict, np.ndarray]:
    """Read a pack back: (header dict, float32 matrix rows x dim; rows x 0 for a class list)."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not an fpack file (bad magic)")
        (hlen,) = struct.unpack("<I", f.read(4))
        hdr = json.loads(f.read(hlen).decode("utf-8"))
        rows, dim, dtype = hdr["rows"], hdr["dim"], hdr["dtype"]
        if dtype == "none":
            return hdr, np.zeros((rows, 0), np.float32)
        np_dtype = "<f2" if dtype == "f16" else "<f4"
        mat = np.frombuffer(f.read(), dtype=np_dtype, count=rows * dim).astype(np.float32)
    return hdr, mat.reshape(rows, dim)


def shown_name(label: list[str]) -> str:
    """The lowest name of a pack row as the app shows it (species, else genus, family, order, class)."""
    return " ".join(x for x in label[5:7] if x) or label[4] or label[3] or label[2]


def row_names(label: list[str], published: list[str] | None) -> list[str]:
    """The "Genus epithet" names of a pack row, plain names only, in order: the pack's species
    name and, for a row of a name rule ([published]: its TreeOfLife names), the genus slot with the
    species slot's first word, and the species slot's own first two words when it starts with a
    genus (rules two_genera and metazoa). For list_fossil_rows.py and the row checks of --drop-rows."""
    names = [f"{label[5]} {label[6]}"]
    if published:
        genus, slot = published[5], published[6]
        words = slot.split()
        if words:
            names.append(f"{genus} {words[0]}")
            if len(words) >= 2 and words[0][:1].isupper():
                names.append(f"{words[0]} {words[1]}")
                names.append(f"{genus} {words[1]}")
    out = []
    for n in names:
        n = n.strip()
        if _PLAIN_NAME.fullmatch(n) and n not in out:
            out.append(n)
    return out


def leave_out_species_rows(hdr: dict, payload: bytes, gone: list[int]) -> bytes:
    """Leave the species rows [gone] (row numbers of [hdr]) out of [hdr] and [payload] (raw matrix
    bytes); every other row keeps its bytes and order. Updates "labels", "rows" and "name_rules"
    (row numbers moved down, counts recounted). The "gbif" and "regions" blocks are row lists of
    the old numbering: they are removed here and must be written again by the caller.
    For add_pack_blocks.py --drop-rows and its check in check_pack_blocks.py --same-as."""
    row_bytes = hdr["dim"] * {"f16": 2, "f32": 4}[hdr["dtype"]]
    n, sinks = hdr["rows"], hdr.get("sink_rows", 0)
    if len(payload) != n * row_bytes:
        raise ValueError(f"payload {len(payload)} bytes, expected {n} rows of {row_bytes}")
    gone_set = set(gone)
    if len(gone_set) != len(gone) or any(not 0 <= i < n - sinks for i in gone_set):
        raise ValueError(f"rows to leave out must be distinct species rows (0 to {n - sinks - 1})")
    kept = [i for i in range(n) if i not in gone_set]
    new_row = {old: new for new, old in enumerate(kept)}
    hdr["labels"] = [hdr["labels"][i] for i in kept]
    hdr["rows"] = len(kept)
    rules = hdr.get("name_rules")
    if rules and rules.get("rows"):
        rules["rows"] = [{**r, "pack_row": new_row[r["pack_row"]]} for r in rules["rows"] if r["pack_row"] not in gone_set]
        counted = {k: 0 for k in rules.get("counts", {})}
        for r in rules["rows"]:
            counted[r["rule"]] = counted.get(r["rule"], 0) + 1
        rules["counts"] = counted
    hdr.pop("gbif", None)
    hdr.pop("regions", None)
    return b"".join(payload[i * row_bytes:(i + 1) * row_bytes] for i in kept)


def rows_to_bits(rows, n: int) -> str:
    """Rows [rows] of a pack with [n] rows as base64 text of ceil(n / 8) bytes: bit i = row i,
    least significant bit first (row 0 = bit 0 of byte 0, row 8 = bit 0 of byte 1)."""
    idx = np.asarray(list(rows), dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"row numbers must be 0 to {n - 1}")
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return base64.b64encode(np.packbits(mask, bitorder="little").tobytes()).decode("ascii")


def bits_to_rows(bits: str, n: int) -> np.ndarray:
    """The reverse of rows_to_bits: a boolean array of length [n]. Refuses text that is not
    plain base64 and set bits after the last row."""
    raw = np.frombuffer(base64.b64decode(bits, validate=True), dtype=np.uint8)
    if raw.size != (n + 7) // 8:
        raise ValueError(f"bit string holds {raw.size} bytes, a pack of {n} rows needs {(n + 7) // 8}")
    all_bits = np.unpackbits(raw, bitorder="little")
    if all_bits[n:].any():
        raise ValueError("bits set after the last row")
    return all_bits[:n].astype(bool)


def sink_label(name: str, prompt: str) -> list[str]:
    """Taxonomy row for a "none of these" candidate: kingdom 'none', the key in the
    species slot, the prompt as the common name."""
    return ["none", "", "", "", "", "", name, prompt]


def _write_class_fixture(path: Path) -> None:
    """Tiny two-level class list for the Dart tests (round 266), with two raw score vectors and
    the row probabilities class_probabilities gives for them (the app's rule, all heads)."""
    level1 = ["Diptera", "Hymenoptera", "Vegetation"]
    level2 = ["Syrphidae", "Eristalis tenax", "Apis mellifera", "Vegetation"]
    head_index = [[0, 0], [0, 1], [1, 2], [2, 3]]
    sizes = [len(level1), len(level2)]
    scores = [[2.0, -1.0, 0.5, 1.5, 0.2, -0.3, 0.1], [-0.5, 3.0, 0.0, 0.4, 0.1, 2.5, -1.0]]
    write_class_list(path, {
        "pack_id": "tiny-classes", "model_id": "tiny-classes", "logit_scale": 1.0, "temperature": 1.0,
        "ranks": RANKS, "sink_rows": 1,
        "labels": [["Animalia", "Arthropoda", "Insecta", "Diptera", "Syrphidae", "", "", ""],
                   ["Animalia", "Arthropoda", "Insecta", "Diptera", "Syrphidae", "Eristalis", "tenax", ""],
                   ["Animalia", "Arthropoda", "Insecta", "Hymenoptera", "Apidae", "Apis", "mellifera", ""],
                   sink_label("Vegetation", "")],
        "classes": level2,
        "heads": [{"name": "level 1", "size": sizes[0], "classes": level1},
                  {"name": "level 2", "size": sizes[1], "classes": level2}],
        "head_index": head_index,
        "test_scores": scores,
        "test_probs": [class_probabilities(s, sizes, head_index).round(6).tolist() for s in scores],
    })


def _write_regions_fixture(path: Path, tiny: dict, mat: np.ndarray) -> None:
    """tiny_pack with the "gbif" and "regions" blocks (2026-10-27) for the Dart tests: the five
    species' real COL keys (insect-detect-post mapping v1.1.0), made-up country lists and
    neighbours, and in "test_union" the species rows of a country with its neighbours, without
    the sink row (the app offers the sink rows on top, always)."""
    n = len(tiny["labels"])
    lists = {"AT": [1, 3, 4], "DE": [0, 1, 2, 3], "MT": [2]}
    blocks = {
        "gbif": {"species_key": ["3B7X4", "6FY28", "FN46", "MFWK", "43P3V", None],
                 "species": ["Eristalis tenax", "Episyrphus balteatus", "Apis mellifera", "Bombus terrestris",
                             "Misumena vatia", None],
                 "source": {"mapping": "bioclip_gbif_species_mapping_Arthropoda.csv v1.1.0"}},
        "regions": {"min_occurrences": 3,
                    "countries": {c: {"rows": rows_to_bits(r, n), "names": len(r), "names_in_country": len(r) + 1}
                                  for c, r in lists.items()},
                    "neighbours": {"AT": ["DE"], "DE": ["AT"]}, "neighbours_rule": "made-up test table",
                    "source": {}},
        "test_union": {"AT": [0, 1, 2, 3, 4], "DE": [0, 1, 2, 3, 4], "MT": [2]},
    }
    write_fpack(path, {**tiny, **blocks, "pack_id": "tiny-test-regions"}, mat, dtype="f16")


if __name__ == "__main__":
    # Writes the tiny cross-language fixtures used by the Dart unit tests:
    # (APP = your clone of the app repo fauna-pulse, e.g. APP=~/InsectDetectApp/fauna-pulse)
    #   python fpack.py "$APP/test/fauna_pulse/fixtures/tiny_pack.fpack" "$APP/test/fauna_pulse/fixtures/tiny_classes.fpack"
    # An optional third path also writes tiny_pack.fpack with the "gbif" and "regions" blocks
    # (tiny_pack_regions.fpack, for the app round that reads them).
    import sys

    if len(sys.argv) > 2:
        _write_class_fixture(Path(sys.argv[2]))
        hdr, _ = read_fpack(Path(sys.argv[2]))
        print("wrote", sys.argv[2], "rows", hdr["rows"], "dim", hdr["dim"], "probs", hdr["test_probs"])
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tiny.fpack")
    rng = np.random.default_rng(7)
    labels = [
        ["Animalia", "Arthropoda", "Insecta", "Diptera", "Syrphidae", "Eristalis", "tenax", "Drone fly"],
        ["Animalia", "Arthropoda", "Insecta", "Diptera", "Syrphidae", "Episyrphus", "balteatus", "Marmalade hoverfly"],
        ["Animalia", "Arthropoda", "Insecta", "Hymenoptera", "Apidae", "Apis", "mellifera", "Western honey bee"],
        ["Animalia", "Arthropoda", "Insecta", "Hymenoptera", "Apidae", "Bombus", "terrestris", "Buff-tailed bumblebee"],
        ["Animalia", "Arthropoda", "Arachnida", "Araneae", "Thomisidae", "Misumena", "vatia", "Goldenrod crab spider"],
        sink_label("flower", "a photo of a flower."),
    ]
    mat = rng.standard_normal((len(labels), 4)).astype(np.float32)
    tiny = {
        "pack_id": "tiny-test", "model_id": "test-model", "logit_scale": 100.0,
        "temperature": 1.0, "ranks": RANKS, "sink_rows": 1, "labels": labels,
    }
    write_fpack(out, tiny, mat, dtype="f16")
    hdr, back = read_fpack(out)
    assert hdr["rows"] == 6 and hdr["dim"] == 4
    print("wrote", out, "rows", hdr["rows"], "dim", hdr["dim"], "first row", back[0])
    if len(sys.argv) > 3:
        _write_regions_fixture(Path(sys.argv[3]), tiny, mat)
        print("wrote", sys.argv[3], "with blocks gbif and regions; test_union", read_fpack(Path(sys.argv[3]))[0]["test_union"])
