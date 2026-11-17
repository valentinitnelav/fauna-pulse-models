#!/usr/bin/env python3
"""Add the GBIF header blocks ("gbif", "regions"; see fpack.py) to an existing label pack as a new
pack, keeping its rows, names and vectors byte for byte (e.g. the v3 Europe pack from v2).

Why not rebuild with build_label_pack.py --country-lists: the species rows come out the same,
but the six "none of these" rows are embedded again by the BioCLIP text tower, and their float16
values can differ in the last bit from one computer to another (v2 of 2026-10-26 against a
rebuild of 2026-10-27: 3 of 6 rows, at most 3e-5). Copying keeps every score of the old pack, and
needs neither the TreeOfLife embeddings (2.7 GB) nor torch: a few seconds.

The keys are joined by the TreeOfLife names of the species list the pack was built from, so
--species-csv must be that list (its SHA-256 is checked against the pack header). A pack built
from whole classes without a species list (mammals and birds v2) takes the key list of
build_region_species_list.py --every-mapped-name for the same classes instead.

--drop-sink KEY leaves out a "none of these" row (its label and its vector), keeping every other
row byte for byte: the mammals-and-birds pack v3 is v2 without the "person" row (humans are named
by the Homo sapiens row since v3, see SINK_SETS in build_label_pack.py).

    python add_pack_blocks.py bioclip-2_mammals-birds-world_v2.fpack --species-csv species_mammals-birds_keys.csv \
        --country-lists countries_mammals-birds.json --in-every-country people-and-domestic --short-list-names 300 --drop-sink person \
        --pack-id bioclip-2_mammals-birds-world_v3 --out out/packs_2026-11

    python add_pack_blocks.py out/packs_v2/bioclip-2_pollinator-orders-europe_v2.fpack \
        --species-csv out/packs_v2/species_europe_4orders_col.csv --country-lists out/packs_v3/countries_4orders_col.json \
        --pack-id bioclip-2_pollinator-orders-europe_v3 --out out/packs_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from build_label_pack import EVERY_COUNTRY_SETS, SINK_SETS, gbif_block, read_species_csv, regions_block, taxa_of
from check_pack_blocks import app_accepts_header, raw_parts
from fpack import read_raw, write_raw


def drop_sink_rows(hdr: dict, payload: bytes, keys: list[str]) -> bytes:
    """Leave the sink rows named [keys] out of [hdr] (labels, rows, sink_rows) and [payload] (their
    vectors); every other row keeps its bytes. Sink rows are the last rows and hold no gbif or
    regions entry, so the species rows keep their numbers."""
    row_bytes = hdr["dim"] * {"f16": 2, "f32": 4}[hdr["dtype"]]
    if len(payload) != hdr["rows"] * row_bytes:
        raise SystemExit(f"payload {len(payload)} bytes, expected {hdr['rows']} rows of {row_bytes}")
    first_sink = hdr["rows"] - hdr["sink_rows"]
    gone = [i for i in range(first_sink, hdr["rows"]) if hdr["labels"][i][6] in keys]
    if len(gone) != len(set(keys)):
        raise SystemExit(f"--drop-sink {keys}: the pack's sink rows are {[l[6] for l in hdr['labels'][first_sink:]]}")
    kept = [i for i in range(hdr["rows"]) if i not in gone]
    hdr["labels"] = [hdr["labels"][i] for i in kept]
    hdr["rows"], hdr["sink_rows"] = len(kept), hdr["sink_rows"] - len(gone)
    print(f"left out sink rows {keys} (rows {gone})")
    return b"".join(payload[i * row_bytes:(i + 1) * row_bytes] for i in kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path, help="existing label pack (.fpack)")
    ap.add_argument("--species-csv", type=Path, required=True, help="the species list the pack was built from")
    ap.add_argument("--country-lists", type=Path, default=None, help="output of build_country_lists.py")
    ap.add_argument("--neighbours", type=Path, default=Path(__file__).with_name("country_neighbours.csv"))
    ap.add_argument("--in-every-country", choices=sorted(EVERY_COUNTRY_SETS), default=None,
                    help="with --country-lists: put these rows on every country's list (see build_label_pack.py)")
    ap.add_argument("--short-list-names", type=int, default=None,
                    help="with --country-lists: the pack's own short-list threshold (see build_label_pack.py)")
    ap.add_argument("--drop-sink", action="append", default=[], metavar="KEY",
                    help="leave out this 'none of these' row (e.g. person); may be given more than once")
    ap.add_argument("--pack-id", required=True, help="the new pack's id and file name, e.g. <model>_<list>_v3")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    hdr, payload = read_raw(args.pack)
    sha = hashlib.sha256(args.species_csv.read_bytes()).hexdigest()
    built_from = hdr.get("source", {}).get("species_csv_sha256")
    if built_from is None:
        # Whole classes, no species list: the CSV must be the key list of exactly those classes.
        meta_path = args.species_csv.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not str(meta.get("regions", "")).startswith("none") or taxa_of(meta) != {"classes": hdr["source"]["classes"]} \
                or hdr["source"].get("orders") or hdr["source"].get("families"):
            raise SystemExit(f"{args.pack.name} holds whole classes {hdr['source'].get('classes')}: --species-csv must be "
                             "build_region_species_list.py --every-mapped-name --classes of the same classes")
    elif built_from != sha:
        raise SystemExit(f"{args.pack.name} was not built from {args.species_csv} (SHA-256 differs)")
    if not args.pack_id.startswith(f"{hdr['model_id']}_"):
        raise SystemExit(f"--pack-id must start with '{hdr['model_id']}_' (the app's naming rule)")
    _, gbif = read_species_csv(args.species_csv)
    if not gbif:
        raise SystemExit(f"{args.species_csv} has no gbif_species_key column (build_region_species_list.py)")

    # The species list holds the names as published; a row with a name rule keeps them in the header.
    sink_rows = hdr.get("sink_rows", 0)
    published = {r["pack_row"]: r["published"] for r in hdr.get("name_rules", {}).get("rows", [])}
    row_names = []
    for i in range(hdr["rows"] - sink_rows):
        lab = published.get(i, hdr["labels"][i])
        row_names.append(f"{lab[5]} {lab[6]}".strip())

    if args.drop_sink:
        args.drop_sink = sorted(set(args.drop_sink))
        payload = drop_sink_rows(hdr, payload, args.drop_sink)
        sink_rows = hdr["sink_rows"]
        left = [l[6] for l in hdr["labels"][len(hdr["labels"]) - sink_rows:]]
        if hdr.get("sink_set") in SINK_SETS and left != [k for k, _ in SINK_SETS[hdr["sink_set"]]]:
            raise SystemExit(f"the sink rows left are not today's sink set '{hdr['sink_set']}' of build_label_pack.py")
    old_id, old_built = hdr["pack_id"], hdr.get("built_with")
    hdr["pack_id"] = args.pack_id
    hdr["gbif"] = gbif_block(row_names, gbif, sink_rows, args.species_csv, hdr["source"]["names_file_sha256"])
    print(f"gbif: {sum(k is not None for k in hdr['gbif']['species_key'])} of {len(row_names)} species rows "
          "have a GBIF species key")
    hdr.pop("regions", None)  # an old block would not match the new keys and date
    if args.country_lists:
        hdr["regions"] = regions_block(hdr["gbif"]["species_key"], row_names, args.country_lists,
                                       args.neighbours, hdr["gbif"]["source"], args.in_every_country,
                                       args.short_list_names)
        print(f"regions: {len(hdr['regions']['countries'])} countries with rows, "
              f"neighbours of {len(hdr['regions']['neighbours'])}")
    hdr["built_with"] = {"script": "add_pack_blocks.py", "date": date.today().isoformat(),
                         "rows_and_vectors_from": {"pack_id": old_id, "file": args.pack.name,
                                                   "sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest(),
                                                   "built_with": old_built,
                                                   **({"sink_rows_left_out": args.drop_sink} if args.drop_sink else {})}}
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.pack_id}.fpack"
    if out_path.resolve() == args.pack.resolve():
        raise SystemExit("--pack-id and --out would overwrite the input pack")
    tmp = out_path.with_name(out_path.name + ".part")
    write_raw(tmp, hdr, payload)
    try:
        app_accepts_header(raw_parts(tmp)[0])
    except ValueError as e:
        tmp.unlink()
        raise SystemExit(f"not written: no released app version could open it ({e})")
    tmp.replace(out_path)
    print(f"Pack written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {hdr['rows']} rows, vectors of {args.pack.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
