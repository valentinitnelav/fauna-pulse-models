#!/usr/bin/env python3
"""List the rows of a mammals-and-birds label pack that name extinct or fossil species, for
add_pack_blocks.py --drop-rows. Writes a CSV with one line per row tested, each with its decision
(drop or keep) and the reason, so the owner can read every row that leaves.

Why: TreeOfLife-200M holds many fossil species (about 1,400 mammal rows of the mammals-and-birds
pack without a GBIF species, e.g. Eomaia scansoria, and extinct Homo species). In the whole list
they compete with living species for every crop: a person was named Homo neanderthalensis. The
country lists already leave them out (build_country_lists.py), but an app uses the whole list
when no country is chosen. The owner decided on 2026-11-17 to leave them out of the pack.

A row stays when a name of it is a living species of the Mammal Diversity Database (MDD; also
when the GBIF species is on the extinct list: e.g. Mystacina robusta, last seen 1967, which MDD
counts as living). Otherwise it is dropped when
1. its GBIF species is on the list of build_country_lists.py's extinct rules (--extinct-csv:
   the Catalogue of Life's extinct mark without an observation, IUCN "EX", the Paleobiology
   Database with its guards), or MDD lists a name of it as an extinct species, or
2. it has no GBIF species and the Paleobiology Database knows a name of it as a chordate and
   gives every such record as extinct.
Any other row stays: with a GBIF species, or when PBDB marks a name living, gives no status, or
does not know any name of the row ("unknown": nothing says it is extinct; listed for the owner).

The names of a row: the pack's species name, and for rows of a name rule (header "name_rules")
the published TreeOfLife names: "genus epithet" from the genus and species slots, and the species
slot's own two first words when it starts with another genus (rule two_genera) or with a genus
and authors (rule metazoa).

    python list_fossil_rows.py out/packs_2026-11/bioclip-2_mammals-birds-world_v3.fpack \
        --extinct-csv countries_mammals-birds_extinct_left_out.csv --mdd-csv MDD_v2.5_6904species.csv \
        --out fossil_rows_mammals-birds.csv

Data: The Paleobiology Database, https://paleobiodb.org (CC BY 4.0), queried on the day the list
is made; Mammal Diversity Database (MDD_FILES in build_region_species_list.py).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import urllib.parse
from datetime import date
from pathlib import Path

from build_region_species_list import MDD_FILES, PBDB_API, _get_json, mdd_living_names
from fpack import read_raw, row_names, shown_name

def mdd_extinct_names(path: Path) -> set[str]:
    """Accepted names of the species the Mammal Diversity Database marks extinct (file checked by
    mdd_living_names)."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["sciName"].replace("_", " ") for r in csv.DictReader(f) if r["extinct"] == "1"}


def pbdb_status(names: list[str]) -> dict[str, set[str]]:
    """{name: set of PBDB 'ext' values ("0" extinct, "1" living, "?" not given)} for the names
    PBDB knows as chordates. Records of other phyla (a plant or an insect of the same name) do not
    count; the class does not help, since PBDB files fossil birds such as Archaeopteryx under
    Reptilia."""
    status: dict[str, set[str]] = collections.defaultdict(set)
    for i in range(0, len(names), 100):
        part = names[i:i + 100]
        r = _get_json(f"{PBDB_API}/taxa/list.json?" + urllib.parse.urlencode({"name": ",".join(part), "show": "attr,class"}))
        for rec in r.get("records") or []:
            plain = " ".join(w for w in rec.get("nam", "").split() if not w.startswith("("))
            if plain in part and rec.get("phl") == "Chordata":
                status[plain].add(str(rec.get("ext", "?")))
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path, help="label pack with a 'gbif' block (add_pack_blocks.py)")
    ap.add_argument("--extinct-csv", type=Path, required=True, help="<lists>_extinct_left_out.csv of build_country_lists.py")
    ap.add_argument("--mdd-csv", type=Path, required=True, help=f"Mammal Diversity Database ({', '.join(MDD_FILES)})")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hdr, _ = read_raw(args.pack)
    if "gbif" not in hdr:
        raise SystemExit(f"{args.pack.name} has no gbif block")
    n = hdr["rows"] - hdr.get("sink_rows", 0)
    keys = hdr["gbif"]["species_key"]
    published = {r["pack_row"]: r["published"] for r in hdr.get("name_rules", {}).get("rows", [])}
    with open(args.extinct_csv, newline="", encoding="utf-8") as f:
        extinct = {r["gbif_species_key"]: r["rule"] for r in csv.DictReader(f)}
    living = mdd_living_names(args.mdd_csv)
    mdd_gone = mdd_extinct_names(args.mdd_csv)

    rows = []
    for i in range(n):
        lab = hdr["labels"][i]
        names = row_names(lab, published.get(i))
        rows.append({"pack_row": i, "class": lab[2], "name_in_pack": shown_name(lab), "names_tested": names, "key": keys[i]})
    asked = sorted({nm for r in rows if not r["key"] and not any(x in living for x in r["names_tested"])
                    for nm in r["names_tested"]})
    print(f"{n} species rows; {sum(1 for r in rows if not r['key'])} without a GBIF species; asking PBDB for {len(asked)} names")
    pbdb = pbdb_status(asked)

    out = []
    for r in rows:
        names = r["names_tested"]
        if any(nm in living for nm in names):
            if r["key"] and r["key"] not in extinct:
                continue  # living species with a GBIF species: not listed
            decision, reason, detail = "keep", "living species of MDD", next(nm for nm in names if nm in living)
        elif r["key"] and r["key"] in extinct:
            decision, reason, detail = "drop", f"extinct ({extinct[r['key']]})", r["key"]
        elif any(nm in mdd_gone for nm in names):
            decision, reason, detail = "drop", "extinct species of MDD", next(nm for nm in names if nm in mdd_gone)
        elif r["key"]:
            continue  # living species with a GBIF species: not listed
        else:
            known = {nm: pbdb[nm] for nm in names if nm in pbdb}
            if not known:
                decision, reason, detail = "keep", "unknown (no name in PBDB or MDD)", ""
            elif all(s == {"0"} for s in known.values()):
                decision, reason, detail = "drop", "fossil (PBDB extinct)", "; ".join(known)
            elif any("1" in s for s in known.values()):
                decision, reason, detail = "keep", "PBDB marks a name living", "; ".join(nm for nm, s in known.items() if "1" in s)
            else:
                decision, reason, detail = "keep", "PBDB gives no status for a name", "; ".join(known)
        out.append({"pack_row": r["pack_row"], "class": r["class"], "name_in_pack": r["name_in_pack"],
                    "names_tested": " | ".join(names), "gbif_species_key": r["key"] or "",
                    "decision": decision, "reason": reason, "detail": detail})

    if not out:
        raise SystemExit("no row to list")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    counts = collections.Counter((o["decision"], o["reason"]) for o in out)
    meta = {"pack": args.pack.name, "pack_id": hdr["pack_id"], "date": date.today().isoformat(),
            "extinct_csv": args.extinct_csv.name, "mdd_csv": args.mdd_csv.name, "pbdb": PBDB_API,
            "counts": {f"{d}: {r}": c for (d, r), c in sorted(counts.items())}}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    for (d, r), c in sorted(counts.items()):
        print(f"  {d:4} {c:5}  {r}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
