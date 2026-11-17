#!/usr/bin/env python3
"""Write a species-list CSV of the TreeOfLife arthropods that GBIF has records for in a
region, for build_label_pack.py --species-csv.

Where the idea comes from: BioCLIP's own documentation recommends restricting the
TreeOfLife candidates to a regional species list built from GBIF (or Map of Life) and
passing it to pybioclip's `--subset` / `apply_filter` ("Geo-Restricted Taxon List
Predictions", https://imageomics.github.io/pybioclip/geo-restricted-taxa/, Imageomics).
The implementation approach followed here (own code, no lines copied) is the one in
Maximilian Sittinger's insect-detect-post, `src/insectdetect_post/build_region_filter.py`
(AGPL-3.0):
- one GBIF occurrence request per region for the whole phylum Arthropoda, with occurrences
  interpreted in the Catalogue of Life Extended Release (COL XR, GBIF's taxonomy since it
  replaced the GBIF Backbone), counted per accepted species ('speciesKey' facet, so records
  under a synonym or a subspecies count for their accepted species), keeping species with at
  least --min-occurrences presence records (his default 3; fossil specimens do not count) in
  that region;
- TreeOfLife species are mapped to COL XR species keys with the table he publishes as a
  release asset (bioclip_gbif_species_mapping_Arthropoda.csv, v1.1.0: made from the BioCLIP 2
  names file txt_emb_species.json, matched against COL26.6 XR; downloaded on first use,
  checked by SHA-256, not redistributed). TreeOfLife rows without a COL species key never pass
  a region list: about 3 % of the plain rows, nearly all rows with codes or authors in the
  species slot, and about half of the rows with two genera or kingdom 'Metazoa' (2026-10-25).
  The list holds BioCLIP 2 strings, so a BioCLIP 2.5 pack built from it keeps only the names
  that the 2.5 file writes the same way (the 2026-10-26 Europe list: 586 of 34,974 not, some
  only spelled differently, e.g. 'Hirticollis Hirticomus quadriguttatus' vs 'Hirticollis
  quadriguttatus'). A mapping of the 2.5 file would be needed to close that gap.
Differences here: regions may be GBIF continents or a union of ISO country codes (as there),
and the rows are narrowed to --orders instead of classes. --world counts the records of the whole
world (no region), order by order (world_species_counts).

--classes instead of --orders selects whole classes of another phylum, with a mapping of that
phylum (make_gbif_mapping.py --phylum Chordata for --classes Mammalia,Aves): GBIF is then asked for
the records of those classes, whose COL keys are read from their own species (taxon_key).
--every-mapped-name writes every mapped name of the selection without asking GBIF for records:
the key list for a pack that keeps all its rows (the mammals-and-birds pack v3, where the country
lists of build_country_lists.py do the narrowing).

GBIF moves its COL XR checklist to newer Catalogue of Life releases, and a species can get a new
key in a new release although its name stays (e.g. Speyeria aglaja: 93N9L in COL26.6 XR, 4YKG8
in COL26.10 XR). A mapping made for one release therefore misses such species in the counts of
another, so this script stops when GBIF's release is not the mapping's (gbif_col_release).

Since 2026-10-25. The v1 packs used his earlier mapping (v1.0.0, GBIF Backbone keys) and a
'taxonKey' facet per order; their species lists stay with them, so they are not rebuilt by this
script.

Citations: pybioclip / BioCLIP (Imageomics, MIT): https://github.com/Imageomics/pybioclip;
Stevens et al. (2024) BioCLIP, CVPR; Gu et al. (2025) BioCLIP 2, NeurIPS.
Sittinger, M. (2026). Software for post-processing of data captured with the Insect
Detect camera trap (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.23404778 (all versions:
https://doi.org/10.5281/zenodo.21822140; https://github.com/maxsitt/insect-detect-post). GBIF occurrence data: GBIF.org,
https://www.gbif.org (the query date is written to the .json next to the CSV). Catalogue of
Life: the release named in the .json (COL26.6 XR: https://doi.org/10.48580/dgy8b).

Examples:
    python build_region_species_list.py --continent EUROPE --orders Diptera,Hymenoptera,Coleoptera,Lepidoptera --out out/species_europe_pollinator_orders.csv
    python build_region_species_list.py --countries DE,AT,CH,CZ,PL --orders Diptera,Hymenoptera --out out/species_central_europe_flies_bees.csv
    python build_region_species_list.py --world --orders Diptera,Hymenoptera,Coleoptera,Lepidoptera --out out/species_world_4orders.csv
    python build_region_species_list.py --every-mapped-name --classes Mammalia,Aves --mapping-csv out/mapping_col26.10xr_chordata/bioclip_gbif_species_mapping_Chordata.csv --out out/species_mammals-birds_keys.csv

Then:
    python build_label_pack.py --model bioclip-2 --species-csv out/species_europe_pollinator_orders.csv --pack-id bioclip-2_pollinator-orders-europe_v2 --out ./out

Insect-detect-post's published mapping covers only arthropods; other phyla need a mapping of
their own (make_gbif_mapping.py --phylum).
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import re
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
ARTHROPODA = {"phylumKey": ARTHROPODA_COL_KEY}  # the records counted for --orders (arthropod orders)
FACET_LIMIT = 1_200_000
# Only presence records count (since 2026-10-28); GBIF reads a record without a status as present.
OCCURRENCE_STATUS = "PRESENT"
# Fossil specimens do not count (since 2026-10-28): they put extinct and Ice Age species on a list.
EXCLUDED_BASIS = ("FOSSIL_SPECIMEN",)
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


@functools.cache
def basis_of_record() -> tuple[str, ...]:
    """GBIF's basisOfRecord values without EXCLUDED_BASIS. GBIF's search has no "not" filter, so
    the kept values are listed; they are read from GBIF on each run, so a value added later counts."""
    values = _get_json(f"{GBIF_API}/enumeration/basic/BasisOfRecord")
    if not set(EXCLUDED_BASIS) <= set(values):
        raise SystemExit(f"GBIF's basisOfRecord values changed ({values}): check EXCLUDED_BASIS")
    return tuple(v for v in values if v not in EXCLUDED_BASIS)


def record_filter() -> dict:
    """The record filters of every count, as GBIF search parameters (and in the output .json)."""
    return {"occurrenceStatus": OCCURRENCE_STATUS, "basisOfRecord": list(basis_of_record())}


def _facet(where: dict, field: str, limit: int, min_count: int = 1) -> tuple[dict[str, int], int]:
    """One GBIF facet over the counted records (record_filter; [where] may narrow basisOfRecord
    further) that match [where], taxa read in COL XR: ({value of [field]: records}, records in
    all). Stops on a missing or cut-off facet."""
    params = {"checklistKey": COL_XR, **record_filter(), **where, "limit": 0, "facet": field,
              "facetMincount": min_count, f"{field}.facetLimit": limit}
    r = _get_json(f"{GBIF_API}/occurrence/search?" + urllib.parse.urlencode(params, doseq=True))
    if not r.get("facets"):
        raise RuntimeError(f"GBIF returned no {field} facet for {where} ({r.get('count')} records)")
    counts = r["facets"][0].get("counts") or []
    if len(counts) >= limit:
        raise RuntimeError(f"GBIF facet {field} truncated at {limit} for {where}")
    return {str(c["name"]): int(c["count"]) for c in counts}, int(r.get("count") or 0)


def species_counts(region: dict, min_occ: int, allow_empty: bool = False, scope: dict | None = None) -> dict[str, int]:
    """COL XR species key -> number of GBIF presence records in the region (>= min_occ) of the taxa
    of [scope] (default: the arthropods), fossil specimens left out. allow_empty: a region with
    records but no species reaching min_occ gives {} (small countries in build_country_lists.py)
    instead of an error."""
    counts, records = _facet({**region, **(scope or ARTHROPODA)}, "speciesKey", FACET_LIMIT, min_occ)
    # No species means a wrong region or a GBIF change, not an empty region: stop rather than
    # write a short list that would silently make a small pack.
    if not counts and not (allow_empty and records):
        raise RuntimeError(f"GBIF returned no species for {region} ({records} records)")
    return counts


# The whole world, order by order (world_species_counts): the largest facet GBIF is asked for, and
# how many species keys go into one request by key.
WORLD_FACET_LIMIT = 300_000
KEYS_PER_REQUEST = 150


def taxon_key(keys: list[str], rank: str = "order") -> str | None:
    """The COL XR key of the order (or class: [rank]) that [keys] (species keys of one TreeOfLife
    order or class) sit in, read from GBIF's records of some of them: the taxon with the most
    records (a stray species may sit in another one). None when the sample has no record."""
    sample = keys[::max(1, len(keys) // KEYS_PER_REQUEST)][:KEYS_PER_REQUEST]
    found, _ = _facet({"speciesKey": sample}, f"{rank}Key", 1000)
    return max(found, key=lambda k: found[k]) if found else None


def order_key(keys: list[str]) -> str | None:
    return taxon_key(keys, "order")


def world_species_counts(keys_by_order: dict[str, list[str]], min_occ: int, rank: str = "order") -> dict[str, int]:
    """COL XR species key -> number of presence records in the whole world (>= min_occ), for the
    given keys only ({TreeOfLife order (or class: [rank]): its mapped species keys}); the same
    records as species_counts (presence only, no fossil specimens).

    GBIF does not answer the species facet of all arthropods without a region (2026-11-16: the
    request fails once it has to return more than about 400,000 species), so the count is made order
    by order. Each order's COL key is read from some of its own species (no name matching, which
    can hit another taxon of the same name), then one facet per order; the keys still unseen
    (species that COL files under another order or under none) are asked for by key."""
    wanted = {k for keys in keys_by_order.values() for k in keys}
    counts: dict[str, int] = {}
    for order, keys in keys_by_order.items():
        t0 = time.time()
        key = taxon_key(keys, rank)
        if key is None:
            print(f"  {order}: no records for a sample of its species; all asked for by key below")
            continue
        found, _ = _facet({f"{rank}Key": key}, "speciesKey", WORLD_FACET_LIMIT, min_occ)
        mine = {k: n for k, n in found.items() if k in wanted}
        counts.update(mine)
        print(f"  world, {order}: {len(found)} species in GBIF, {len(mine)} of them mapped ({time.time() - t0:.0f} s)")
    unseen = sorted(wanted - counts.keys())
    late = 0
    for i in range(0, len(unseen), KEYS_PER_REQUEST):
        part = unseen[i:i + KEYS_PER_REQUEST]
        found, _ = _facet({"speciesKey": part}, "speciesKey", 10 * KEYS_PER_REQUEST, min_occ)
        hits = {k: n for k, n in found.items() if k in wanted}
        counts.update(hits)
        late += len(hits)
    print(f"  world, {len(unseen)} keys not in their order's facet asked for by key: {late} more species")
    return counts


def class_scope(keys_by_class: dict[str, list[str]]) -> dict:
    """GBIF search parameters for the records of whole classes (--classes): their COL XR keys, read
    from their own mapped species (taxon_key), since a class name can also be the name of another
    taxon. Stops when a class has no records."""
    keys = {}
    for cls, ks in keys_by_class.items():
        if not ks:  # an empty speciesKey list would ask GBIF about all records and return another class
            raise SystemExit(f"the mapping holds no species of the class {cls} with a key: check the name")
        keys[cls] = taxon_key(ks, "class")
        if keys[cls] is None:
            raise SystemExit(f"GBIF has no records for a sample of the {cls} species: cannot find its COL key")
    print("COL XR class keys: " + ", ".join(f"{c} {k}" for c, k in keys.items()))
    return {"classKey": sorted(keys.values())}


CLB_API = "https://api.checklistbank.org"
OBSERVED_ALIVE = ("HUMAN_OBSERVATION", "MACHINE_OBSERVATION")  # GBIF's basisOfRecord of a living individual seen


def extinct_unobserved(keys_by_order: dict[str, list[str]], clb_dataset_key: str, rank: str = "order") -> set[str]:
    """The species keys of [keys_by_order] that the Catalogue of Life release marks as extinct and
    of which GBIF holds no observation of a living individual (--leave-out-extinct).

    Why both conditions: the fossil-specimen rule (EXCLUDED_BASIS) only removes species whose
    records are filed as fossils, and many amber fossils are filed as preserved specimens (e.g. the
    Baltic amber ant Eldermyrmex exsectus, 369 such records). The Catalogue of Life's extinct mark
    finds them, but it also sits on some living species (2026-11-16: Bibio slossonae, 440
    observations); an observation keeps those. A living species with the mark and with museum
    specimens only is still left out, so the names left out are written next to the list.
    The mark is asked for inside each order's (or class's: [rank]) main COL key (taxon_key), so a
    species that COL files under another one is not checked."""
    wanted = {k for keys in keys_by_order.values() for k in keys}
    extinct: set[str] = set()
    for keys in keys_by_order.values():
        key = taxon_key(keys, rank)
        offset, total = 0, 1
        while key is not None and offset < total:  # 200 per page: larger answers broke off in transfer
            r = _get_json(f"{CLB_API}/dataset/{clb_dataset_key}/nameusage/search?" + urllib.parse.urlencode(
                {"extinct": "true", "TAXON_ID": key, "rank": "species", "limit": 200, "offset": offset}))
            extinct |= {str(x["id"]) for x in r.get("result") or []} & wanted
            offset, total = offset + 200, int(r.get("total") or 0)
    alive: set[str] = set()
    marked = sorted(extinct)
    for i in range(0, len(marked), KEYS_PER_REQUEST):
        part = marked[i:i + KEYS_PER_REQUEST]
        found, _ = _facet({"speciesKey": part, "basisOfRecord": list(OBSERVED_ALIVE)}, "speciesKey", 10 * KEYS_PER_REQUEST)
        alive |= set(found)
    return extinct - alive


PBDB_API = "https://paleobiodb.org/data1.2"
# IUCN Red List categories of a living species: an assessment as one of these overrules a fossil
# database's "extinct" (EW: extinct in the wild, kept in captivity and reintroduced).
IUCN_LIVING = ("LC", "NT", "VU", "EN", "CR", "DD", "EW")


# MDD release whose living species guard pbdb_extinct (Mammal Diversity Database v2.5, CC BY 4.0,
# https://doi.org/10.5281/zenodo.21654811; file name -> SHA-256).
MDD_FILES = {"MDD_v2.5_6904species.csv": "0d07a7e9409712fa86c1e3afadcf4c67bf4f9e16d5693a878e11ec1bf6860493"}
MDD_URL = "https://zenodo.org/api/records/21654811/files/MDD_v2.5_6904species.csv/content"
OBSERVED = ("HUMAN_OBSERVATION", "MACHINE_OBSERVATION", "OBSERVATION")  # GBIF's basisOfRecord of a sighting
MAX_OBSERVED_SHARE = 0.10


def mdd_living_names(path: Path) -> set[str]:
    """"Genus epithet" names of the living species of the Mammal Diversity Database file [path]:
    accepted names, their synonyms (nominalNames) and original combinations. Checked by SHA-256."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if MDD_FILES.get(path.name) != digest:
        raise SystemExit(f"{path}: not the MDD file named in MDD_FILES (download: {MDD_URL})")
    names: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["extinct"] == "1":
                continue
            names.add(r["sciName"].replace("_", " "))
            for field in ("nominalNames", "originalNameCombination"):
                for nominal in r[field].split("|"):
                    w = nominal.replace("_", " ").split()
                    if len(w) >= 2 and w[0][:1].isupper():
                        names.add(f"{w[0]} {w[1]}")
                    elif w and w[0][:1].islower():  # an epithet alone, of the species' genus
                        names.add(f"{r['genus']} {w[0]}")
    return names


def pbdb_extinct(mapping_rows: list[dict], mdd_csv: Path | None = None) -> set[str]:
    """The species keys of [mapping_rows] (rows of the mapping CSV) that the Paleobiology Database
    marks as extinct (taxon attribute 'ext' 0), when also
    - the IUCN has not assessed them as a living species (iucn_category not in IUCN_LIVING),
    - under MAX_OBSERVED_SHARE of their GBIF records are observations (OBSERVED), and
    - for mammals, the Mammal Diversity Database [mdd_csv] does not list the name as a living species.
    For build_country_lists.py --leave-out-pbdb-extinct.

    Why: Ice Age species are in GBIF as preserved specimens and material samples of museums, not as
    "fossil specimens", and neither the Catalogue of Life nor the IUCN marks them extinct
    (2026-11-17: Mammuthus primigenius on 12 country lists, Ursus spelaeus on 14, Bos priscus on 10).
    Living species known only from museum specimens look the same in GBIF, so a fossil database
    decides. PBDB alone is not enough: a name can also match a fossil taxon of the same name, and
    'extinct' then came back for the moose (Alces alces), bandicoots (Isoodon), rails and the anoas
    (Bubalus depressicornis, quarlesi). The observation share keeps the well observed ones, and MDD
    the rarely observed living mammals (the anoas: 4 % observations; the IUCN category of the
    mapping is missing for them). 2026-11-17, mammals and birds: 150 PBDB 'extinct' species keys,
    127 left out, all fossil or extinct. A name is asked for as the TreeOfLife name and as the
    accepted COL name; a name PBDB does not know leaves the species in.
    Data: The Paleobiology Database, https://paleobiodb.org (CC BY 4.0), queried on the day the
    lists are made; Mammal Diversity Database (MDD_FILES)."""
    asked: dict[str, set[str]] = {}
    for m in mapping_rows:
        if m["gbif_species_key"] and m.get("iucn_category") not in IUCN_LIVING:
            for name in {m["bioclip_species"], m["gbif_species"]} - {""}:
                asked.setdefault(name, set()).add(m["gbif_species_key"])
    names = sorted(n for n in asked if re.fullmatch(r"[A-Z][a-z]+ [a-z][a-z-]+", n))  # PBDB refuses other forms
    extinct: set[str] = set()
    for i in range(0, len(names), 100):  # PBDB takes a comma list of names
        part = names[i:i + 100]
        r = _get_json(f"{PBDB_API}/taxa/list.json?" + urllib.parse.urlencode({"name": ",".join(part), "show": "attr"}))
        for rec in r.get("records") or []:
            name = rec.get("nam", "")
            plain = " ".join(w for w in name.split() if not w.startswith("("))  # 'Bos (Urus) priscus'
            if str(rec.get("ext")) == "0" and plain in asked:  # PBDB writes "0" (extinct) or "1" as text
                extinct |= asked[plain]
    keys = sorted(extinct)
    records, observed = {}, {}
    for i in range(0, len(keys), KEYS_PER_REQUEST):
        part = keys[i:i + KEYS_PER_REQUEST]
        records.update(_facet({"speciesKey": part}, "speciesKey", 10 * KEYS_PER_REQUEST)[0])
        observed.update(_facet({"speciesKey": part, "basisOfRecord": list(OBSERVED)}, "speciesKey", 10 * KEYS_PER_REQUEST)[0])
    seen = {k for k in keys if observed.get(k, 0) >= MAX_OBSERVED_SHARE * max(1, records.get(k, 0))}
    mammals = {m["gbif_species_key"] for m in mapping_rows if m["bioclip_class"] == "Mammalia"} & extinct
    if mammals and mdd_csv is None:
        raise SystemExit("--leave-out-pbdb-extinct for mammals needs --mdd-csv (the Mammal Diversity Database, MDD_URL)")
    living = mdd_living_names(mdd_csv) if mammals else set()
    in_mdd = {m["gbif_species_key"] for m in mapping_rows if m["gbif_species_key"] in mammals
              and (m["bioclip_species"] in living or m["gbif_species"] in living)}
    print(f"  PBDB: {len(names)} names without an IUCN category of a living species asked, {len(extinct)} species keys "
          f"extinct there; kept as living: {len(seen)} often observed, {len(in_mdd - seen)} more in MDD")
    return extinct - seen - in_mdd


def gbif_col_release(mapping_release: dict | None) -> dict:
    """The COL release GBIF matches occurrences against today (as insect-detect-post's
    fetch_col_release). GBIF moves its COL XR checklist key to each new release, so the key alone
    does not say which release made a species key. Stops when it is not the mapping's release:
    keys of another release would silently miss names of the mapping."""
    meta = _get_json(f"{GBIF_API.replace('/v1', '/v2')}/species/match/metadata?checklistKey={COL_XR}")
    clb_key = str(meta["mainIndex"]["clbDatasetKey"])
    dataset = _get_json(f"https://api.checklistbank.org/dataset/{clb_key}")
    release = {"checklist_key": COL_XR, "clb_dataset_key": clb_key,
               "alias": meta["mainIndex"].get("datasetAlias") or dataset.get("alias"),
               "issued": dataset.get("issued"), "doi": dataset.get("doi")}
    expected = (mapping_release or {}).get("clb_dataset_key")
    if expected is None:
        raise SystemExit("the mapping's provenance file names no COL release: cannot check its species keys")
    if clb_key != str(expected):
        raise SystemExit(f"GBIF now uses {release['alias']} (ChecklistBank {clb_key}), the mapping "
                         f"{(mapping_release or {}).get('alias')} ({expected}): its species keys may not match")
    return release


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


def mapping_source(mapping_dir: Path, mapping_csv: Path | None = None) -> tuple[Path, dict, dict]:
    """(the mapping CSV, its provenance record, the "mapping" record of our outputs).
    Without [mapping_csv]: insect-detect-post's release asset v1.1.0, kept in [mapping_dir]. With
    it: a mapping made by make_gbif_mapping.py for the COL release GBIF uses now; its provenance
    .json must sit next to it and name the CSV's SHA-256."""
    if mapping_csv is None:
        csv_path = mapping_file(mapping_dir, "bioclip_gbif_species_mapping_Arthropoda.csv")
        provenance = json.loads(mapping_file(mapping_dir, "bioclip_gbif_species_mapping_Arthropoda.json").read_text())
        source = {"url": RELEASE + csv_path.name, "sha256": MAPPING_FILES[csv_path.name]}
    else:
        csv_path = mapping_csv
        provenance = json.loads(mapping_csv.with_suffix(".json").read_text())
        digest = hashlib.sha256(mapping_csv.read_bytes()).hexdigest()
        if provenance.get("csv_sha256") != digest:
            raise SystemExit(f"{mapping_csv}: its .json was written for another CSV (SHA-256 differs)")
        run = mapping_csv.with_suffix(".run.json")
        source = {"file": mapping_csv.name, "sha256": digest, "resolved_utc": provenance.get("resolved_utc"),
                  "run": json.loads(run.read_text()) if run.exists() else None}
    return csv_path, provenance, {**source, "col_release": provenance.get("col_release"),
                                  "tol_species_list": provenance.get("tol_species_list")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--continent", help="GBIF continent, e.g. EUROPE")
    g.add_argument("--countries", help="comma list of ISO 3166-1 alpha-2 codes, e.g. DE,AT,CH")
    g.add_argument("--world", action="store_true", help="no region: the records of the whole world")
    g.add_argument("--every-mapped-name", action="store_true",
                   help="no records asked for: every name of the selection that the mapping gives a species key")
    t = ap.add_mutually_exclusive_group(required=True)
    t.add_argument("--orders", help="comma list of order names (arthropods)")
    t.add_argument("--classes", help="comma list of class names (e.g. Mammalia,Aves; needs a --mapping-csv of their phylum)")
    ap.add_argument("--min-occurrences", type=int, default=3, help="GBIF records needed in a region (default 3)")
    ap.add_argument("--mapping-dir", type=Path, default=Path("out"), help="where the mapping files are kept")
    ap.add_argument("--mapping-csv", type=Path, default=None,
                    help="a mapping made by make_gbif_mapping.py for GBIF's current COL release "
                         "(default: insect-detect-post's release asset v1.1.0, COL26.6 XR)")
    ap.add_argument("--leave-out-extinct", action="store_true",
                    help="leave out species that the Catalogue of Life marks as extinct and that GBIF has no "
                         "observation of a living individual of (written to <out>_extinct_left_out.csv)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    regions = ([{"continent": args.continent.upper()}] if args.continent
               else [] if args.world or args.every_mapped_name
               else [{"country": c.strip().upper()} for c in args.countries.split(",") if c.strip()])
    rank = "class" if args.classes else "order"
    groups = [o.strip() for o in (args.classes or args.orders).split(",") if o.strip()]
    if args.classes and args.mapping_csv is None:
        raise SystemExit("--classes needs a --mapping-csv of their phylum (the default mapping holds only arthropods)")
    csv_path, provenance, mapping_record = mapping_source(args.mapping_dir, args.mapping_csv)
    with open(csv_path, newline="", encoding="utf-8") as f:
        mapping = [m for m in csv.DictReader(f) if m[f"bioclip_{rank}"] in set(groups) and m["gbif_species_key"]]
    print(f"mapping: {len(mapping)} TreeOfLife species of {groups} with a COL species key")
    col_release = gbif_col_release(provenance.get("col_release"))
    keys_of = lambda ms, g: sorted({m["gbif_species_key"] for m in ms if m[f"bioclip_{rank}"] == g})
    scope = ARTHROPODA if rank == "order" else class_scope({g: keys_of(mapping, g) for g in groups})

    # Each region on its own needs >= min_occurrences records (as in insect-detect-post); counts are summed.
    in_region: dict[str, int] = {}
    for region in regions:
        t0 = time.time()
        counts = species_counts(region, args.min_occurrences, scope=scope)
        for key, n in counts.items():
            in_region[key] = in_region.get(key, 0) + n
        print(f"  {region}: {len(counts)} species ({time.time() - t0:.0f} s)")
    if args.world:
        in_region = world_species_counts({g: keys_of(mapping, g) for g in groups}, args.min_occurrences, rank)

    rows = sorted((m for m in mapping if args.every_mapped_name or m["gbif_species_key"] in in_region),
                  key=lambda m: (m["bioclip_order"], m["bioclip_family"], m["bioclip_species"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extinct_record = None
    if args.leave_out_extinct:
        gone = extinct_unobserved({g: keys_of(rows, g) for g in groups}, col_release["clb_dataset_key"], rank)
        left_out = [m for m in rows if m["gbif_species_key"] in gone]
        rows = [m for m in rows if m["gbif_species_key"] not in gone]
        side = args.out.with_name(args.out.stem + "_extinct_left_out.csv")
        with open(side, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["species", "family", "order", "gbif_species", "gbif_species_key", "gbif_occurrences_in_region"])
            for m in left_out:
                w.writerow([m["bioclip_species"], m["bioclip_family"], m["bioclip_order"], m["gbif_species"],
                            m["gbif_species_key"], in_region.get(m["gbif_species_key"], "")])
        extinct_record = {"rule": "marked extinct in the COL release and no GBIF record with basisOfRecord "
                                  + " or ".join(OBSERVED_ALIVE), "names": len(left_out), "file": side.name}
        print(f"left out as extinct and never observed alive: {len(left_out)} names ({side.name})")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 'species' = the TreeOfLife "genus + species slot" string that build_label_pack.py matches;
        # gbif_species = the accepted COL name, for reference only (the pack keeps the TreeOfLife names).
        w.writerow(["species", "genus", "family", "order", "class", "gbif_species", "gbif_species_key",
                    "gbif_status", "gbif_occurrences_in_region"])
        for m in rows:
            w.writerow([m["bioclip_species"], m["bioclip_genus"], m["bioclip_family"], m["bioclip_order"],
                        m["bioclip_class"], m["gbif_species"], m["gbif_species_key"], m["gbif_status"],
                        in_region.get(m["gbif_species_key"], "")])
    # The keys of --orders stay as before (pollinator lists); --classes adds the classes and their COL keys.
    taxa = {"orders": groups} if rank == "order" else {"classes": groups, "gbif_scope": scope}
    meta = {"regions": "none (every mapped name)" if args.every_mapped_name else regions or "world", **taxa,
            "min_occurrences": None if args.every_mapped_name else args.min_occurrences,
            "species": len(rows), "occurrence_status": OCCURRENCE_STATUS,
            "basis_of_record_excluded": list(EXCLUDED_BASIS), "gbif_checklist_key": COL_XR,
            "mapping": mapping_record,
            **({"extinct_left_out": extinct_record} if extinct_record else {}),
            "gbif_col_release": col_release, "date": date.today().isoformat()}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    per_group = {g: sum(1 for m in rows if m[f"bioclip_{rank}"] == g) for g in groups}
    print(f"wrote {args.out}: {len(rows)} species {per_group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
