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

from build_region_species_list import (ARTHROPODA_COL_KEY, COL_XR, EXCLUDED_BASIS, GBIF_API, OCCURRENCE_STATUS,
                                       _get_json, gbif_col_release, mapping_source, record_filter, species_counts)


def gbif_countries() -> list[str]:
    """ISO codes of the countries GBIF has COL XR arthropod records from, largest first. Codes
    outside GBIF's country list (e.g. ZZ, unknown) are left out: a request for them fails."""
    known = {c["iso2"] for c in _get_json(f"{GBIF_API}/enumeration/country")}
    params = {"checklistKey": COL_XR, "phylumKey": ARTHROPODA_COL_KEY, **record_filter(), "limit": 0,
              "facet": "country", "country.facetLimit": 1000}
    r = _get_json(f"{GBIF_API}/occurrence/search?" + urllib.parse.urlencode(params, doseq=True))
    return [c["name"] for c in r["facets"][0]["counts"] if c["name"] in known]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species-csv", type=Path, required=True,
                    help="output of build_region_species_list.py (its .json next to it gives orders and minimum)")
    ap.add_argument("--orders", default=None, help="comma list of orders (default: from the CSV's .json)")
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
    orders = ([o.strip() for o in args.orders.split(",") if o.strip()] if args.orders else meta.get("orders"))
    if not orders:
        raise SystemExit(f"no orders: pass --orders (no 'orders' in {meta_path})")
    min_occ = args.min_occurrences or meta.get("min_occurrences", 3)

    csv_path, provenance, mapping_record = mapping_source(args.mapping_dir, args.mapping_csv)
    if meta.get("mapping") and meta["mapping"].get("sha256") != mapping_record["sha256"]:
        raise SystemExit(f"{args.species_csv.name} was made with another mapping: pass the same --mapping-csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        mapping = [m for m in csv.DictReader(f) if m["bioclip_order"] in set(orders) and m["gbif_species_key"]]
    names_per_key: dict[str, int] = {}
    for m in mapping:
        names_per_key[m["gbif_species_key"]] = names_per_key.get(m["gbif_species_key"], 0) + 1
    print(f"mapping: {len(mapping)} TreeOfLife species of {orders} with a COL species key")

    col_release = gbif_col_release(provenance.get("col_release"))
    countries = gbif_countries()
    print(f"{len(countries)} countries with arthropod records in GBIF")
    out: dict[str, dict] = {}
    t0 = time.time()
    for i, c in enumerate(countries, 1):
        counts = species_counts({"country": c}, min_occ, allow_empty=True)
        keys = {k: n for k, n in counts.items() if k in names_per_key}
        out[c] = {"names_in_country": sum(names_per_key[k] for k in keys), "keys": dict(sorted(keys.items()))}
        if i % 25 == 0 or i == len(countries):
            print(f"  {i}/{len(countries)} countries ({time.time() - t0:.0f} s)")

    result = {"orders": orders, "min_occurrences": min_occ, "occurrence_status": OCCURRENCE_STATUS,
              "basis_of_record_excluded": list(EXCLUDED_BASIS),
              "gbif_checklist_key": COL_XR,
              "mapping": mapping_record,
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
