#!/usr/bin/env python3
"""Per-country GBIF species lists for a species CSV of build_region_species_list.py, so a label
pack can narrow its names to a country (build_label_pack.py --country-lists).

For every country GBIF has arthropod records from, one occurrence request (the same request as
build_region_species_list.py --countries): accepted species of the Catalogue of Life Extended
Release (COL XR) with at least --min-occurrences presence records in that country, fossil
specimens left out. Kept per country: the COL species keys of the mapped TreeOfLife species of the
CSV's orders (mapping of insect-detect-post v1.1.0, checked by SHA-256), with their record counts,
and how many TreeOfLife names of those orders the country's list holds. A pack keeps only its own
rows of each list (build_label_pack.py).

Where the idea comes from and whose approach is followed: see build_region_species_list.py
(BioCLIP's "Geo-Restricted Taxon List Predictions", Imageomics; Maximilian Sittinger's
insect-detect-post, own code here). GBIF occurrence data: GBIF.org (query date in the output).
Catalogue of Life: the release named in the output (COL26.6 XR: https://doi.org/10.48580/dgy8b).

About 250 requests, under a second each for most countries (a few seconds for the largest);
about 2.5 to 3.5 minutes in total (147 s on 2026-10-27, 195 s on 2026-10-28), no account needed.

Example (the v2 Europe list; orders and minimum are read from its .json):
    python build_country_lists.py --species-csv out/packs_v2/species_europe_4orders_col.csv --out out/packs_v3/countries_4orders_col.json

A species list made with --classes (e.g. Mammalia,Aves and the chordate mapping) gives lists of
those classes: GBIF is asked for their records (the class keys of its .json). --leave-out-extinct
leaves out of every country's list the species that build_region_species_list.py
--leave-out-extinct would leave out (marked extinct by the Catalogue of Life and never observed
alive in GBIF), for a pack that keeps such species as rows (mammals and birds v3); the names go
to <out>_extinct_left_out.csv. --leave-out-iucn-extinct also leaves out the species whose IUCN Red
List category in the mapping (column iucn_category, as GBIF's name match gives it) is EX: the
Catalogue of Life marks few birds and mammals extinct (2026-11-17: 251 mammal species), and museum
specimens and photos of them put e.g. the dodo, the great auk and the thylacine on country lists.
"Extinct in the wild" (EW) stays: such species are bred and reintroduced. Neither rule catches
Ice Age species that both sources leave unmarked (e.g. Mammuthus primigenius, Ursus spelaeus,
known in GBIF from preserved specimens and material samples, not "fossil specimens");
--leave-out-pbdb-extinct does, from the Paleobiology Database (pbdb_extinct in
build_region_species_list.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
from datetime import date
from pathlib import Path

from build_region_species_list import (ARTHROPODA, COL_XR, EXCLUDED_BASIS, GBIF_API, IUCN_LIVING, MAX_OBSERVED_SHARE,
                                       MDD_FILES, OBSERVED_ALIVE, OCCURRENCE_STATUS, _get_json, extinct_unobserved, gbif_col_release, mapping_source,
                                       pbdb_extinct, record_filter, species_counts)


def gbif_countries(scope: dict | None = None) -> list[str]:
    """ISO codes of the countries GBIF has COL XR records from, of the taxa of [scope] (default: the
    arthropods), largest first. Codes outside GBIF's country list (e.g. ZZ, unknown) are left out:
    a request for them fails."""
    known = {c["iso2"] for c in _get_json(f"{GBIF_API}/enumeration/country")}
    params = {"checklistKey": COL_XR, **(scope or ARTHROPODA), **record_filter(), "limit": 0,
              "facet": "country", "country.facetLimit": 1000}
    r = _get_json(f"{GBIF_API}/occurrence/search?" + urllib.parse.urlencode(params, doseq=True))
    return [c["name"] for c in r["facets"][0]["counts"] if c["name"] in known]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species-csv", type=Path, required=True,
                    help="output of build_region_species_list.py (its .json next to it gives orders and minimum)")
    ap.add_argument("--orders", default=None, help="comma list of orders (default: from the CSV's .json)")
    ap.add_argument("--leave-out-extinct", action="store_true",
                    help="leave out of every list the species marked extinct by the Catalogue of Life and never "
                         "observed alive in GBIF (written to <out>_extinct_left_out.csv)")
    ap.add_argument("--leave-out-iucn-extinct", action="store_true",
                    help="also leave out species whose IUCN category in the mapping is EX (extinct)")
    ap.add_argument("--leave-out-pbdb-extinct", action="store_true",
                    help="also leave out species that the Paleobiology Database marks extinct, with the guards of "
                         "pbdb_extinct (Ice Age species)")
    ap.add_argument("--mdd-csv", type=Path, default=None,
                    help="Mammal Diversity Database species CSV, for --leave-out-pbdb-extinct with mammals")
    ap.add_argument("--min-occurrences", type=int, default=None,
                    help="GBIF records needed in a country (default: from the CSV's .json, else 3)")
    ap.add_argument("--mapping-dir", type=Path, default=Path("out"), help="where the mapping files are kept")
    ap.add_argument("--mapping-csv", type=Path, default=None,
                    help="a mapping made by make_gbif_mapping.py (default: the release asset v1.1.0); "
                         "must be the one the species list was made with")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta_path = args.species_csv.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    classes = meta.get("classes") if not args.orders else None
    orders = ([o.strip() for o in args.orders.split(",") if o.strip()] if args.orders else meta.get("orders"))
    if not orders and not classes:
        raise SystemExit(f"no orders: pass --orders (no 'orders' or 'classes' in {meta_path})")
    rank, groups = ("class", classes) if classes else ("order", orders)
    scope = meta["gbif_scope"] if classes else ARTHROPODA
    min_occ = args.min_occurrences or meta.get("min_occurrences") or 3

    csv_path, provenance, mapping_record = mapping_source(args.mapping_dir, args.mapping_csv)
    if meta.get("mapping") and meta["mapping"].get("sha256") != mapping_record["sha256"]:
        raise SystemExit(f"{args.species_csv.name} was made with another mapping: pass the same --mapping-csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        mapping = [m for m in csv.DictReader(f) if m[f"bioclip_{rank}"] in set(groups) and m["gbif_species_key"]]
    names_per_key: dict[str, int] = {}
    for m in mapping:
        names_per_key[m["gbif_species_key"]] = names_per_key.get(m["gbif_species_key"], 0) + 1
    print(f"mapping: {len(mapping)} TreeOfLife species of {groups} with a COL species key")

    col_release = gbif_col_release(provenance.get("col_release"))
    gone: dict[str, str] = {}  # species key -> why it is left out
    extinct_record = None
    if args.leave_out_extinct or args.leave_out_iucn_extinct or args.leave_out_pbdb_extinct:
        rules = []
        if args.leave_out_extinct:
            col = extinct_unobserved({g: sorted({m["gbif_species_key"] for m in mapping if m[f"bioclip_{rank}"] == g})
                                      for g in groups}, col_release["clb_dataset_key"], rank)
            gone.update((k, "col_extinct_not_observed") for k in col)
            rules.append("marked extinct in the COL release and no GBIF record with basisOfRecord "
                         + " or ".join(OBSERVED_ALIVE))
        if args.leave_out_iucn_extinct:
            gone.update((m["gbif_species_key"], "iucn_ex") for m in mapping
                        if m.get("iucn_category") == "EX" and m["gbif_species_key"] not in gone)
            rules.append("IUCN Red List category EX in the mapping (iucn_category)")
        if args.leave_out_pbdb_extinct:
            gone.update((k, "pbdb_extinct") for k in sorted(pbdb_extinct(mapping, args.mdd_csv)) if k not in gone)
            rules.append("marked extinct by the Paleobiology Database (paleobiodb.org, CC BY 4.0), no IUCN category of "
                         f"a living species ({', '.join(IUCN_LIVING)}), under {MAX_OBSERVED_SHARE:.0%} of the GBIF "
                         "records observations, and for mammals not a living species of the Mammal Diversity Database "
                         + ", ".join(MDD_FILES))
        side = args.out.with_name(args.out.stem + "_extinct_left_out.csv")
        side.parent.mkdir(parents=True, exist_ok=True)
        with open(side, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["species", "family", "order", "class", "gbif_species", "gbif_species_key", "rule"])
            for m in sorted((m for m in mapping if m["gbif_species_key"] in gone), key=lambda m: m["bioclip_species"]):
                w.writerow([m["bioclip_species"], m["bioclip_family"], m["bioclip_order"], m["bioclip_class"],
                            m["gbif_species"], m["gbif_species_key"], gone[m["gbif_species_key"]]])
        extinct_record = {"rule": "; or ".join(rules), "species_keys": len(gone), "file": side.name}
        print(f"left out of every country as extinct: {len(gone)} species ({side.name})")
    countries = gbif_countries(scope)
    print(f"{len(countries)} countries with records of {groups} in GBIF")
    out: dict[str, dict] = {}
    t0 = time.time()
    for i, c in enumerate(countries, 1):
        counts = species_counts({"country": c}, min_occ, allow_empty=True, scope=scope)
        keys = {k: n for k, n in counts.items() if k in names_per_key and k not in gone}
        out[c] = {"names_in_country": sum(names_per_key[k] for k in keys), "keys": dict(sorted(keys.items()))}
        if i % 25 == 0 or i == len(countries):
            print(f"  {i}/{len(countries)} countries ({time.time() - t0:.0f} s)")

    taxa = {"orders": orders} if rank == "order" else {"classes": classes, "gbif_scope": scope}
    result = {**taxa, "min_occurrences": min_occ, "occurrence_status": OCCURRENCE_STATUS,
              "basis_of_record_excluded": list(EXCLUDED_BASIS),
              "gbif_checklist_key": COL_XR,
              "mapping": mapping_record,
              **({"extinct_left_out": extinct_record} if extinct_record else {}),
              "gbif_col_release": col_release, "species_csv": args.species_csv.name,
              "date": date.today().isoformat(), "countries": out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    biggest = sorted(out, key=lambda c: -out[c]["names_in_country"])[:5]
    print(f"wrote {args.out}: {len(out)} countries; most names: "
          + ", ".join(f"{c} {out[c]['names_in_country']}" for c in biggest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
