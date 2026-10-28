#!/usr/bin/env python3
"""Add the GBIF header blocks ("gbif", "regions"; see fpack.py) to an existing label pack as a new
pack, keeping its rows, names and vectors byte for byte (e.g. the v3 Europe pack from v2).

Why not rebuild with build_label_pack.py --country-lists: the species rows come out the same,
but the six "none of these" rows are embedded again by the BioCLIP text tower, and their float16
values can differ in the last bit from one computer to another (v2 of 2026-10-26 against a
rebuild of 2026-10-27: 3 of 6 rows, at most 3e-5). Copying keeps every score of the old pack, and
needs neither the TreeOfLife embeddings (2.7 GB) nor torch: a few seconds.

The keys are joined by the TreeOfLife names of the species list the pack was built from, so
--species-csv must be that list (its SHA-256 is checked against the pack header).

    python add_pack_blocks.py out/packs_v2/bioclip-2_pollinator-orders-europe_v2.fpack \
        --species-csv out/packs_v2/species_europe_4orders_col.csv --country-lists out/packs_v3/countries_4orders_col.json \
        --pack-id bioclip-2_pollinator-orders-europe_v3 --out out/packs_v3
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

from build_label_pack import gbif_block, read_species_csv, regions_block
from check_pack_blocks import app_accepts_header, raw_parts
from fpack import read_raw, write_raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path, help="existing label pack (.fpack)")
    ap.add_argument("--species-csv", type=Path, required=True, help="the species list the pack was built from")
    ap.add_argument("--country-lists", type=Path, default=None, help="output of build_country_lists.py")
    ap.add_argument("--neighbours", type=Path, default=Path(__file__).with_name("country_neighbours.csv"))
    ap.add_argument("--pack-id", required=True, help="the new pack's id and file name, e.g. <model>_<list>_v3")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    hdr, payload = read_raw(args.pack)
    sha = hashlib.sha256(args.species_csv.read_bytes()).hexdigest()
    if hdr.get("source", {}).get("species_csv_sha256") != sha:
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

    old_id, old_built = hdr["pack_id"], hdr.get("built_with")
    hdr["pack_id"] = args.pack_id
    hdr["gbif"] = gbif_block(row_names, gbif, sink_rows, args.species_csv, hdr["source"]["names_file_sha256"])
    print(f"gbif: {sum(k is not None for k in hdr['gbif']['species_key'])} of {len(row_names)} species rows "
          "have a GBIF species key")
    hdr.pop("regions", None)  # an old block would not match the new keys and date
    if args.country_lists:
        hdr["regions"] = regions_block(hdr["gbif"]["species_key"], row_names, args.country_lists,
                                       args.neighbours, hdr["gbif"]["source"])
        print(f"regions: {len(hdr['regions']['countries'])} countries with rows, "
              f"neighbours of {len(hdr['regions']['neighbours'])}")
    hdr["built_with"] = {"script": "add_pack_blocks.py", "date": date.today().isoformat(),
                         "rows_and_vectors_from": {"pack_id": old_id, "file": args.pack.name,
                                                   "sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest(),
                                                   "built_with": old_built}}
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
