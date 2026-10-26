#!/usr/bin/env python3
"""Write a species-list CSV of the TreeOfLife arthropods that GBIF has records for in a
region, for build_label_pack.py --species-csv.

Where the idea comes from: BioCLIP's own documentation recommends restricting the
TreeOfLife candidates to a regional species list built from GBIF (or Map of Life) and
passing it to pybioclip's `--subset` / `apply_filter` ("Geo-Restricted Taxon List
Predictions", https://imageomics.github.io/pybioclip/geo-restricted-taxa/, Imageomics).
The implementation approach followed here (own code, no lines copied) is the one in
Maximilian Sittinger's insect-detect-post, `src/insectdetect_post/build_region_filter.py`
(AGPL-3.0), so that a region gives the same species there and in FaunaPulse:
- one GBIF occurrence request per region for the whole phylum Arthropoda, with occurrences
  interpreted in the Catalogue of Life Extended Release (COL XR, GBIF's taxonomy since it
  replaced the GBIF Backbone), counted per accepted species ('speciesKey' facet, so records
  under a synonym or a subspecies count for their accepted species), keeping species with at
  least --min-occurrences records (his default 3) in that region;
- TreeOfLife species are mapped to COL XR species keys with the table he publishes as a
  release asset (bioclip_gbif_species_mapping_Arthropoda.csv, v1.1.0: made from the same
  TreeOfLife names file as our packs, matched against COL26.6 XR; downloaded on first use,
  checked by SHA-256, not redistributed). TreeOfLife rows without a COL species key never pass
  a region list: about 3 % of the plain rows, nearly all rows with codes or authors in the
  species slot, and about half of the rows with two genera or kingdom 'Metazoa' (2026-10-25).
Differences here: regions may be GBIF continents or a union of ISO country codes (as there),
and the rows are narrowed to --orders instead of classes.

Since 2026-10-25. The v1 packs used his earlier mapping (v1.0.0, GBIF Backbone keys) and a
'taxonKey' facet per order; their species lists stay with them, so they are not rebuilt by this
script.

Citations: pybioclip / BioCLIP (Imageomics, MIT): https://github.com/Imageomics/pybioclip;
Stevens et al. (2024) BioCLIP, CVPR; Gu et al. (2025) BioCLIP 2, NeurIPS.
Sittinger, M. (2026). Software for post-processing of data captured with the Insect
Detect camera trap (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21822140
(https://github.com/maxsitt/insect-detect-post). GBIF occurrence data: GBIF.org,
https://www.gbif.org (the query date is written to the .json next to the CSV). Catalogue of
Life: the release named in the .json (COL26.6 XR: https://doi.org/10.48580/dgy8b).

Examples:
    python build_region_species_list.py --continent EUROPE --orders Diptera,Hymenoptera,Coleoptera,Lepidoptera --out out/species_europe_pollinator_orders.csv
    python build_region_species_list.py --countries DE,AT,CH,CZ,PL --orders Diptera,Hymenoptera --out out/species_central_europe_flies_bees.csv

Then:
    python build_label_pack.py --model bioclip-2 --species-csv out/species_europe_pollinator_orders.csv --pack-id bioclip-2_pollinator-orders-europe_v2 --out ./out

Only arthropods are covered by the mapping; other groups need their own key mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

RELEASE = "https://github.com/maxsitt/insect-detect-post/releases/download/v1.1.0/"
# file name -> SHA-256, as registered in his filters/filter_assets.json (v1.1.0)
MAPPING_FILES = {
    "bioclip_gbif_species_mapping_Arthropoda.csv": "e09a04c9dbc61cc914ee12fbe03b666c67123d0fcdd59573eabc9bf5335dbe12",
    "bioclip_gbif_species_mapping_Arthropoda.json": "5a130f0622138184dc93264e1a91cf03fb9f398c0c568f6d35f248e972166771",
}
GBIF_API = "https://api.gbif.org/v1"
COL_XR = "7ddf754f-d193-4cc9-b351-99906754a03b"  # GBIF checklistKey of COL XR
ARTHROPODA_COL_KEY = "RT"  # COL XR key of the phylum, restricts the facet server-side
FACET_LIMIT = 1_200_000
RETRIES = 4


def _get_json(url: str, timeout: int = 300) -> dict:
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise
            last = e
        except Exception as e:  # network hiccups, 5xx
            last = e
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"GBIF request failed after {RETRIES} attempts: {last}")


def species_counts(region: dict, min_occ: int) -> dict[str, int]:
    """COL XR species key -> number of GBIF arthropod records in the region (>= min_occ)."""
    params = {**region, "checklistKey": COL_XR, "phylumKey": ARTHROPODA_COL_KEY, "limit": 0,
              "facet": "speciesKey", "facetMincount": min_occ, "speciesKey.facetLimit": FACET_LIMIT}
    r = _get_json(f"{GBIF_API}/occurrence/search?" + urllib.parse.urlencode(params))
    counts = r["facets"][0]["counts"] if r.get("facets") else []
    if len(counts) >= FACET_LIMIT:
        raise RuntimeError("GBIF facet truncated; raise FACET_LIMIT")
    return {str(c["name"]): int(c["count"]) for c in counts}


def mapping_file(folder: Path, name: str) -> Path:
    """The mapping file, downloaded on first use; refused if its SHA-256 differs from v1.1.0."""
    path = folder / name
    if not path.exists():
        print(f"Downloading {name} to {folder} ...")
        folder.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RELEASE + name, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MAPPING_FILES[name]:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the one of release v1.1.0; delete the file to download it again")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--continent", help="GBIF continent, e.g. EUROPE")
    g.add_argument("--countries", help="comma list of ISO 3166-1 alpha-2 codes, e.g. DE,AT,CH")
    ap.add_argument("--orders", required=True, help="comma list of order names")
    ap.add_argument("--min-occurrences", type=int, default=3, help="GBIF records needed in a region (default 3)")
    ap.add_argument("--mapping-dir", type=Path, default=Path("out"), help="where the mapping files are kept")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    regions = ([{"continent": args.continent.upper()}] if args.continent
               else [{"country": c.strip().upper()} for c in args.countries.split(",") if c.strip()])
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    csv_path = mapping_file(args.mapping_dir, "bioclip_gbif_species_mapping_Arthropoda.csv")
    provenance = json.loads(mapping_file(args.mapping_dir, "bioclip_gbif_species_mapping_Arthropoda.json").read_text())
    with open(csv_path, newline="", encoding="utf-8") as f:
        mapping = [m for m in csv.DictReader(f) if m["bioclip_order"] in set(orders) and m["gbif_species_key"]]
    print(f"mapping: {len(mapping)} TreeOfLife species of {orders} with a COL species key")

    # Each region on its own needs >= min_occurrences records (as in insect-detect-post); counts are summed.
    in_region: dict[str, int] = {}
    for region in regions:
        t0 = time.time()
        counts = species_counts(region, args.min_occurrences)
        for key, n in counts.items():
            in_region[key] = in_region.get(key, 0) + n
        print(f"  {region}: {len(counts)} arthropod species ({time.time() - t0:.0f} s)")

    rows = sorted((m for m in mapping if m["gbif_species_key"] in in_region),
                  key=lambda m: (m["bioclip_order"], m["bioclip_family"], m["bioclip_species"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 'species' = the TreeOfLife "genus + species slot" string that build_label_pack.py matches;
        # gbif_species = the accepted COL name, for reference only (the pack keeps the TreeOfLife names).
        w.writerow(["species", "genus", "family", "order", "class", "gbif_species", "gbif_species_key",
                    "gbif_status", "gbif_occurrences_in_region"])
        for m in rows:
            w.writerow([m["bioclip_species"], m["bioclip_genus"], m["bioclip_family"], m["bioclip_order"],
                        m["bioclip_class"], m["gbif_species"], m["gbif_species_key"], m["gbif_status"],
                        in_region[m["gbif_species_key"]]])
    meta = {"regions": regions, "orders": orders, "min_occurrences": args.min_occurrences,
            "species": len(rows), "gbif_checklist_key": COL_XR,
            "mapping": {"url": RELEASE + csv_path.name, "sha256": MAPPING_FILES[csv_path.name],
                        "col_release": provenance.get("col_release"),
                        "tol_species_list": provenance.get("tol_species_list")},
            "date": date.today().isoformat()}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    per_order = {o: sum(1 for m in rows if m["bioclip_order"] == o) for o in orders}
    print(f"wrote {args.out}: {len(rows)} species {per_order}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
