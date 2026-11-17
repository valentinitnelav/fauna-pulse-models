#!/usr/bin/env python3
"""Checks of the optional "gbif" and "regions" header blocks of a label pack (see fpack.py).

Without a pack: a self-test on tiny packs in a temporary folder (bits round trip, the blocks
change nothing else in the file, the header passes the app's limits). With a pack: checks it;
with --same-as OLD also that it holds the same rows, names, name rules and vectors as an older
pack (v3 against v2), so that only the blocks are new. Needs numpy only.

    python check_pack_blocks.py
    python check_pack_blocks.py out/packs_v3/bioclip-2_pollinator-orders-europe_v3.fpack \
        --same-as out/packs_v2/bioclip-2_pollinator-orders-europe_v2.fpack
"""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from fpack import MAGIC, RANKS, bits_to_rows, leave_out_species_rows, read_fpack, row_names, rows_to_bits, shown_name, sink_label, write_fpack

# The app refuses larger or oddly shaped headers (label_pack.dart, round 327): an old app version
# must still open every new pack.
APP_MAX_HEADER_BYTES = 16 << 20


def raw_parts(path: Path) -> tuple[bytes, bytes]:
    """(header bytes, everything after the header) of a pack."""
    data = path.read_bytes()
    assert data[:4] == MAGIC, f"{path}: not an fpack file"
    (hlen,) = struct.unpack("<I", data[4:8])
    return data[8:8 + hlen], data[8 + hlen:]


def app_accepts_header(hdr: bytes) -> None:
    """Python copy of LabelPack._checkJsonShape and the size limit (app round 327). Raises
    ValueError (not assert, so it also works under python -O; the writers call it)."""
    if len(hdr) > APP_MAX_HEADER_BYTES:
        raise ValueError(f"header {len(hdr)} bytes > the app's limit {APP_MAX_HEADER_BYTES}")
    depth = containers = 0
    in_string = escaped = False
    limit = 64 + len(hdr) // 16
    for b in hdr:
        if in_string:
            if escaped:
                escaped = False
            elif b == 0x5C:
                escaped = True
            elif b == 0x22:
                in_string = False
        elif b == 0x22:
            in_string = True
        elif b in (0x5B, 0x7B):
            depth += 1
            containers += 1
            if depth > 16 or containers > limit:
                raise ValueError("the app would refuse this header's shape")
        elif b in (0x5D, 0x7D):
            depth -= 1


SAMPLE_COUNTRIES = ("DE", "FR", "GB", "MT", "CY", "RO")


def check_pack(path: Path, same_as: Path | None = None, countries: tuple[str, ...] = SAMPLE_COUNTRIES) -> bool:
    """Checks [path]; False when the blocks are fine but no released app version can open the file
    (a pack for PC use, e.g. a world pack), True when the app can."""
    hdr_bytes, payload = raw_parts(path)
    try:
        app_accepts_header(hdr_bytes)
        app = "ok"
    except ValueError as e:
        app = f"NO released app version can open it ({e})"
    hdr, mat = read_fpack(path)
    n, sinks = hdr["rows"], hdr.get("sink_rows", 0)
    assert len(hdr["labels"]) == n and mat.shape == (n, hdr["dim"])
    print(f"{path.name}: {n} rows ({sinks} sink rows), header {len(hdr_bytes) / 1e6:.2f} MB, "
          f"file {path.stat().st_size / 1e6:.1f} MB; app limits: {app}")
    sink = np.zeros(n, dtype=bool)
    sink[n - sinks:] = True
    assert all(hdr["labels"][i][0] == "none" for i in range(n - sinks, n)), "sink rows must be last"

    # Names as published (the species list's names): a row with a name rule keeps them in the header.
    published = {r["pack_row"]: r["published"] for r in hdr.get("name_rules", {}).get("rows", [])}
    names = ["" if sink[i] else "{} {}".format(*published.get(i, lab)[5:7]).strip() for i, lab in enumerate(hdr["labels"])]
    if "gbif" in hdr:
        g = hdr["gbif"]
        keys, species = g["species_key"], g["species"]
        assert len(keys) == n and len(species) == n, "gbif lists need one entry per row"
        assert all(keys[i] is None and species[i] is None for i in range(n - sinks, n)), "sink rows have no GBIF species"
        assert all(k is None or (isinstance(k, str) and k) for k in keys), "keys are non-empty strings or null"
        assert all((k is None) == (s is None) for k, s in zip(keys, species)), "key and name come together"
        with_key = sum(k is not None for k in keys)
        print(f"  gbif: {with_key} of {n - sinks} species rows have a key, {len(set(keys) - {None})} distinct keys")
    if "regions" in hdr:
        assert "gbif" in hdr, "regions come from the keys"
        reg = hdr["regions"]
        for code, c in reg["countries"].items():
            mask = bits_to_rows(c["rows"], n)
            assert mask.any() and not (mask & sink).any(), f"{code}: no rows or a sink row set"
            assert all(hdr["gbif"]["species_key"][i] is not None for i in np.flatnonzero(mask)), f"{code}: row without key"
            assert c["names"] == len({names[i] for i in np.flatnonzero(mask)}), f"{code}: names count"
            assert c["names"] <= c["names_in_country"], f"{code}: more pack names than the country list holds"
        for code, nb in reg["neighbours"].items():
            assert code in reg["countries"] and code not in nb and len(set(nb)) == len(nb), f"neighbours of {code}"
            for other in nb:
                assert other not in reg["neighbours"] or code in reg["neighbours"][other], f"{code}-{other} one-sided"
        anywhere = np.zeros(n, dtype=bool)
        for c in reg["countries"].values():
            anywhere |= bits_to_rows(c["rows"], n)
        print(f"  regions: {len(reg['countries'])} countries, {int((~anywhere & ~sink).sum())} species rows in no "
              f"country (whole pack only); rule: {reg['neighbours_rule']}")
        for code in countries:
            if code in reg["countries"]:
                union = bits_to_rows(reg["countries"][code]["rows"], n).copy()
                for other in reg["neighbours"].get(code, []):
                    if other in reg["countries"]:
                        union |= bits_to_rows(reg["countries"][other]["rows"], n)
                c = reg["countries"][code]
                print(f"    {code}: {c['names']} names ({c['names_in_country']} on its GBIF list), with "
                      f"{','.join(reg['neighbours'].get(code, [])) or 'no neighbours'}: {len({names[i] for i in np.flatnonzero(union)})}")
    if same_as is not None:
        old_hdr_bytes, old_payload = raw_parts(same_as)
        old, _ = read_fpack(same_as)
        source = hdr.get("built_with", {}).get("rows_and_vectors_from") or {}
        # add_pack_blocks.py --drop-rows: those species rows are left out (checked by name), every other row stays.
        left_out = (source.get("species_rows_left_out") or {}).get("rows") or []
        if left_out:
            old_published = {r["pack_row"]: r["published"] for r in old.get("name_rules", {}).get("rows", [])}
            for i, name, tested, _ in left_out:
                assert shown_name(old["labels"][i]) == name and \
                    " | ".join(row_names(old["labels"][i], old_published.get(i))) == tested, f"{same_as.name}: row {i} is not '{name}' ({tested})"
            old_payload = leave_out_species_rows(old, old_payload, [row[0] for row in left_out])
            print(f"  {same_as.name} without the {len(left_out)} species rows listed in the header:")
        # add_pack_blocks.py --drop-sink: those sink rows are left out, every other row stays.
        dropped = source.get("sink_rows_left_out") or []
        if dropped:
            row_bytes = len(old_payload) // old["rows"]
            gone = [i for i in range(old["rows"] - old["sink_rows"], old["rows"]) if old["labels"][i][6] in dropped]
            assert len(gone) == len(dropped), f"{same_as.name} has no sink rows {dropped}"
            old_payload = b"".join(old_payload[i * row_bytes:(i + 1) * row_bytes] for i in range(old["rows"]) if i not in gone)
            old["labels"] = [l for i, l in enumerate(old["labels"]) if i not in gone]
            old["rows"], old["sink_rows"] = old["rows"] - len(gone), old["sink_rows"] - len(gone)
            print(f"  {same_as.name} without its sink rows {dropped} (rows {gone}):")
        assert payload == old_payload, "the vectors differ from the older pack"
        skip = {"gbif", "regions", "built_with", "pack_id", "source"}
        for k in sorted((set(hdr) | set(old)) - skip):
            assert hdr.get(k) == old.get(k), f"header key '{k}' differs from {same_as.name}"
        for k in sorted(set(hdr["source"]) | set(old["source"])):
            assert hdr["source"].get(k) == old["source"].get(k), f"source.{k} differs from {same_as.name}"
        print(f"  same rows, names, name rules and vectors (byte-identical) as {same_as.name}")
    return app == "ok"


def self_test() -> None:
    rng = np.random.default_rng(3)
    for n in (1, 7, 8, 9, 64, 34984):
        rows = sorted(set(rng.integers(0, n, size=max(1, n // 3)).tolist()))
        back = bits_to_rows(rows_to_bits(rows, n), n)
        assert back.sum() == len(rows) and back[rows].all(), f"bits round trip failed for {n} rows"
    assert rows_to_bits([0, 9], 10) == "AQI=", "bit order: row 0 = bit 0 of byte 0, row 9 = bit 1 of byte 1"
    labels = [["Animalia", "Arthropoda", "Insecta", "Diptera", "Syrphidae", "Eristalis", "tenax", ""],
              ["Animalia", "Arthropoda", "Insecta", "Hymenoptera", "Apidae", "Apis", "mellifera", ""],
              sink_label("flower", "a photo of a flower.")]
    base = {"pack_id": "t", "model_id": "m", "logit_scale": 100.0, "temperature": 1.0, "ranks": RANKS,
            "sink_rows": 1, "labels": labels}
    mat = rng.standard_normal((3, 4)).astype(np.float32)
    blocks = {"gbif": {"species_key": ["4ZKN", "123", None], "species": ["Eristalis tenax", "Apis mellifera", None]},
              "regions": {"min_occurrences": 3, "countries": {"DE": {"rows": rows_to_bits([0, 1], 3), "names": 2,
                                                                      "names_in_country": 5},
                                                               "AT": {"rows": rows_to_bits([1], 3), "names": 1,
                                                                      "names_in_country": 4}},
                          "neighbours": {"AT": ["DE"], "DE": ["AT"]}, "neighbours_rule": "test"}}
    with tempfile.TemporaryDirectory() as tmp:
        plain, extra = Path(tmp) / "plain.fpack", Path(tmp) / "blocks.fpack"
        write_fpack(plain, base, mat)
        write_fpack(extra, {**base, **blocks}, mat)
        (h0, m0), (h1, m1) = read_fpack(plain), read_fpack(extra)
        assert h0["labels"] == h1["labels"] and np.array_equal(m0, m1) and raw_parts(plain)[1] == raw_parts(extra)[1]
        check_pack(extra)
    # leave_out_species_rows (add_pack_blocks.py --drop-rows): other rows keep bytes, order and name rules.
    hdr = {**base, "rows": 3, "dim": 4, "dtype": "f16", **blocks,
           "name_rules": {"counts": {"two_genera": 1}, "rows": [{"pack_row": 1, "rule": "two_genera", "published": labels[1]}]}}
    raw = mat.astype("<f2").tobytes()
    out = leave_out_species_rows(hdr, raw, [0])
    assert out == raw[8:] and hdr["rows"] == 2 and hdr["labels"] == labels[1:] and "gbif" not in hdr and "regions" not in hdr
    assert hdr["name_rules"]["rows"][0]["pack_row"] == 0 and hdr["name_rules"]["counts"] == {"two_genera": 1}
    assert shown_name(labels[0]) == "Eristalis tenax" and shown_name(["Animalia", "Chordata", "Aves", "", "Emeidae", "", "", ""]) == "Emeidae"
    for bad in ([2], [0, 0]):  # a sink row, a row twice
        try:
            leave_out_species_rows({**base, "rows": 3, "dim": 4, "dtype": "f16"}, raw, bad)
            raise AssertionError(f"rows {bad} must be refused")
        except ValueError:
            pass
    print("self-test: ok")


def main() -> int:
    if not __debug__:
        raise SystemExit("run without -O: the checks are assert statements")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path, nargs="?", help="pack to check (default: self-test only)")
    ap.add_argument("--same-as", type=Path, default=None, help="older pack that must hold the same rows and vectors")
    ap.add_argument("--countries", default=",".join(SAMPLE_COUNTRIES),
                    help="countries whose list sizes are printed (default: %(default)s)")
    args = ap.parse_args()
    self_test()
    if args.pack and not check_pack(args.pack, args.same_as, tuple(c.strip().upper() for c in args.countries.split(","))):
        return 1  # the blocks are fine, but the app cannot open the file
    return 0


if __name__ == "__main__":
    sys.exit(main())
