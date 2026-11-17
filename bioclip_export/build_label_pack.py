#!/usr/bin/env python3
"""Build a FaunaPulse label pack (.fpack) from the TreeOfLife text embeddings.

In plain words: BioCLIP identifies a crop by comparing its image embedding with one
pre-computed text embedding per candidate name. The Imageomics team publishes those
name embeddings for every species in TreeOfLife (2.66 GB for BioCLIP 2). This script
downloads them once, keeps only the rows you ask for (e.g. class Insecta + Arachnida,
optionally restricted to a species list such as a GBIF country list), adds a few
"none of these" rows (flower, leaf, shadow, ...) so false detections are not forced
onto an insect name, and writes ONE .fpack file the app imports.

Examples:
    # Insecta + Arachnida worldwide for BioCLIP 2 (about 280k rows, ~430 MB at f16):
    python build_label_pack.py --model bioclip-2 --classes Insecta,Arachnida --out ./out

    # Same, restricted to a species list (one column 'species', "Genus epithet"):
    python build_label_pack.py --model bioclip-2 --classes Insecta,Arachnida \
        --species-csv species_DE.csv --pack-id bioclip-2_insecta-arachnida-de_v1 --out ./out

    # Only some orders (small pack for a first phone test):
    python build_label_pack.py --model bioclip-2 --orders Diptera,Hymenoptera,Coleoptera,Lepidoptera --out ./out

Sink rows need the text tower (torch + open_clip). Pass --no-sink to skip them (not
recommended for real use: every crop then gets an insect name, even a leaf).

A species list from build_region_species_list.py also gives every row its GBIF species key and
accepted name (header block "gbif", see fpack.py); with --country-lists (build_country_lists.py)
the pack also records which rows GBIF lists in each country and each country's neighbours
(block "regions"), so the app can narrow the names to the session's country. For a pack that is
already published, use add_pack_blocks.py instead: it keeps the vectors bit for bit (the v3
Europe pack was made that way), while this script embeds the sink rows again. A test build:
    python build_label_pack.py --model bioclip-2 --classes Insecta --species-csv out/packs_v2/species_europe_4orders_col.csv \
        --country-lists out/packs_v3/countries_4orders_col.json --pack-id bioclip-2_pollinator-orders-europe_test --out out/scratch

Names stay as TreeOfLife publishes them. The few rows not in the plain form (kingdom
'Metazoa', two genera in one row, authors or codes in the species slot) are used by the app
only down to family or genus and listed in the header (--name-rules higher-rank, the
default; see NAME_RULES). --name-rules none uses every row as published (the v1 packs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

from fpack import RANKS, rows_to_bits, sink_label, write_fpack

EMBEDDINGS = {
    # model key -> (HF dataset repo, npy file, json file)
    "bioclip-2": ("imageomics/TreeOfLife-200M", "embeddings/txt_emb_species.npy", "embeddings/txt_emb_species.json"),
    "bioclip-2.5": ("imageomics/TreeOfLife-200M", "embeddings/txt_emb_bioclip-2.5-vith14.npy",
                    "embeddings/txt_emb_bioclip-2.5-vith14.json"),
    "bioclip-1": ("imageomics/TreeOfLife-10M", "embeddings/txt_emb_species.npy", "embeddings/txt_emb_species.json"),
}
MODEL_HUB = {
    "bioclip-2": "hf-hub:imageomics/bioclip-2",
    "bioclip-2.5": "hf-hub:imageomics/bioclip-2.5-vith14",
    "bioclip-1": "hf-hub:imageomics/bioclip",
}
# "None of these" candidates: (key, prompt), per kind of detector/scene.
# Explicit "background" classes are common practice in image classification; a
# precedent for insect crops is the none_bg / none_dirt / none_shadow / none_bird
# classes of the Insect Detect classification dataset (Sittinger, Uhler & Pink 2023,
# Zenodo 10.5281/zenodo.8325384). BioCLIP and pybioclip have no such rows; embedding
# negative prompts with the text tower next to the TreeOfLife names is FaunaPulse's
# own use of the model. "arthropod" = flower-scene false positives, "mammal" =
# camera-trap false positives of a MegaDetector "animal" box. The mammal set had a "person" row
# ("a photo of a person.") up to the v2 mammals-and-birds pack; since v3 humans are named by the
# TreeOfLife row Homo sapiens, as every other mammal (owner, 2026-10-28), and the app sends
# MegaDetector's "person" boxes to BioCLIP too.
SINK_SETS = {
    "arthropod": [
        ("flower", "a photo of a flower."),
        ("leaf", "a photo of a leaf."),
        ("shadow", "a photo of a shadow on a flower."),
        ("debris", "a photo of dirt or debris."),
        ("blurry", "a blurry photo with no animal."),
        ("web", "a photo of a spider web."),
    ],
    "mammal": [
        ("vegetation", "a photo of vegetation with no animal."),
        ("ground", "a photo of bare ground, rocks or a path with no animal."),
        ("vehicle", "a photo of a vehicle."),
        ("shadow", "a photo of a shadow or a dark blurry shape."),
        ("blurry", "a blurry photo with no animal."),
    ],
    # The set of the mammals-and-birds packs v1 and v2, to rebuild them as published.
    "mammal-v2": [
        ("vegetation", "a photo of vegetation with no animal."),
        ("ground", "a photo of bare ground, rocks or a path with no animal."),
        ("person", "a photo of a person."),
        ("vehicle", "a photo of a vehicle."),
        ("shadow", "a photo of a shadow or a dark blurry shape."),
        ("blurry", "a blurry photo with no animal."),
    ],
    "none": [],
}
SINK_PROMPTS = SINK_SETS["arthropod"]

# --in-every-country: rows put on every country's list of the "regions" block, with the reason
# written next to them in the header. GBIF holds few records of people and of domestic animals
# (2026-11-17, at least 3 presence records: Homo sapiens in 68 countries, not in Nigeria or India;
# the TreeOfLife row "Canis lupus" (common name Domestic Dog) in 79, Bos taurus in 123), as
# observers rarely record them, yet a camera trap sees them everywhere; without this a person's box
# would be forced onto another primate once the names are narrowed to a country.
EVERY_COUNTRY_SETS = {
    "people-and-domestic": (
        ["Homo sapiens", "Canis lupus", "Felis catus", "Bos taurus", "Ovis aries", "Capra hircus", "Sus scrofa",
         "Equus caballus", "Equus asinus", "Gallus gallus"],
        "people and the most widespread domestic animals live in every country, but GBIF holds few records of them",
    ),
}


def load_tol(model_key: str, cache_dir: Path | None, embeddings_dir: Path | None = None):
    """Open the TreeOfLife embeddings for [model_key]: from a folder that already holds
    the .npy and .json (embeddings_dir, same file names as on Hugging Face), else
    downloaded once from Hugging Face."""
    repo, npy_name, json_name = EMBEDDINGS[model_key]
    if embeddings_dir is not None:
        npy_path = embeddings_dir / Path(npy_name).name
        json_path = embeddings_dir / Path(json_name).name
    else:
        from huggingface_hub import hf_hub_download

        kw = {"repo_id": repo, "repo_type": "dataset"}
        if cache_dir is not None:
            kw["cache_dir"] = str(cache_dir)
        npy_path = hf_hub_download(filename=npy_name, **kw)
        json_path = hf_hub_download(filename=json_name, **kw)
    emb = np.load(npy_path, mmap_mode="r")  # [dim, N] in pybioclip's layout
    raw = Path(json_path).read_bytes()
    # The SHA-256 of the names file goes into the pack, so a user can cite exactly which
    # TreeOfLife names it holds (the file name stays the same across dataset revisions).
    return emb, json.loads(raw), hashlib.sha256(raw).hexdigest()


def row_taxonomy(entry) -> list[str]:
    """One TreeOfLife json entry -> [kingdom..species epithet, common name].

    pybioclip documents entries as [scientific_names(list of 7), common_name]; older
    dumps used flat lists. Handle both.
    """
    if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], (list, tuple)):
        sci, common = list(entry[0]), entry[1]
    elif isinstance(entry, (list, tuple)) and len(entry) >= 7:
        sci, common = list(entry[:7]), (entry[7] if len(entry) > 7 else "")
    elif isinstance(entry, dict):
        sci = [entry.get(r, "") for r in RANKS]
        common = entry.get("common_name", "")
    else:
        raise ValueError(f"unexpected TreeOfLife entry: {entry!r}")
    sci = [("" if s is None else str(s)) for s in sci][:7]
    while len(sci) < 7:
        sci.append("")
    return sci + [("" if common is None else str(common))]


# --name-rules higher-rank: a TreeOfLife row that is not in the plain form is used by the app only
# down to the rank where it is still plain. No name is corrected, built or moved and no row is
# dropped, so the pack keeps the upstream names; a taxonomic fix would be our own opinion between
# competing backbones (an audit of 2026-10-23 found that rebuilding "genus + epithet" from such rows
# made names that GBIF lists as synonyms or does not know). The only renaming is kingdom
# 'Metazoa' -> 'Animalia': the TreeOfLife-200M card calls 'Metazoa' an unresolved kingdom
# ("Metazoa instead of Animalia"), and NCBI Taxonomy lists Animalia as a synonym of Metazoa
# (taxid 33208). Without it those rows form a separate kingdom branch that the app's top-down
# answer never reaches, so their probability would count for no family at all.
NAME_RULES = {
    "metazoa": (4, "kingdom 'Metazoa' (unresolved in TreeOfLife-200M): read as Animalia, used down to family"),
    "two_genera": (4, "the species slot starts with another genus than the genus slot: used down to family"),
    "species_not_plain": (5, "the species slot is not one plain epithet (authors, years, codes, "
                             "placeholders such as 'spec' or 'indet'): used down to genus"),
}
PLAIN_EPITHET = re.compile(r"[a-z][a-z-]*")
# Words for "species not identified" that pass as one lowercase word ('Bombus spec',
# 'Camponotus indet', 'Cladotanytarsus group'; 'gen' = genus).
PLACEHOLDER_EPITHETS = {"sp", "spp", "spec", "nr", "cf", "aff", "indet", "gen", "group", "complex"}


def name_rule(tax: list[str]) -> str | None:
    """The NAME_RULES key for a TreeOfLife row that is not in the plain form, else None."""
    if tax[0] == "Metazoa":
        return "metazoa"
    species = tax[6]
    if not species:
        return None  # already ends above species
    words = re.sub(r"\([^)]*\)", " ", species).replace(",", " ").split()
    if words and re.fullmatch(r"[A-Z][a-z]+", words[0]) and words[0] != tax[5]:
        return "two_genera"  # e.g. Kirinia | 'Esperarge eversmanni': which genus is right is not ours to decide
    if not PLAIN_EPITHET.fullmatch(species) or species in PLACEHOLDER_EPITHETS:
        return "species_not_plain"
    return None


def apply_name_rule(tax: list[str], rule: str) -> list[str]:
    """The row's labels for the app: slots below the rule's deepest rank are left empty (the app
    counts a row only down to the rank above its first empty slot), and so is the common name, which
    belongs to the species. The original row goes into the header (main)."""
    deepest = NAME_RULES[rule][0]
    t = list(tax)
    if rule == "metazoa":
        t[0] = "Animalia"
    for k in range(deepest + 1, 8):
        t[k] = ""
    return t


def read_species_csv(path: Path) -> tuple[set[str], dict[str, tuple[str, str | None]]]:
    """The names of a species list ('species' column, else the first) and, for lists of
    build_region_species_list.py, each name's GBIF species: {"Genus epithet": (COL key, accepted name)}."""
    keep, gbif = set(), {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "species" if "species" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            name = (row.get(col) or "").strip()
            if not name:
                continue
            keep.add(name)
            if (row.get("gbif_species_key") or "").strip():
                gbif[name] = (row["gbif_species_key"].strip(), (row.get("gbif_species") or "").strip() or None)
    print(f"species list: {len(keep)} names from {path}" + (f", {len(gbif)} with a GBIF species key" if gbif else ""))
    return keep, gbif


def taxa_of(d: dict) -> dict:
    """The selection a species list, country list or block was made for: its orders (arthropods)
    or classes (other phyla), to check that two files are for the same names."""
    return {k: sorted(d[k]) for k in ("orders", "classes") if d.get(k)}


def gbif_block(row_names: list[str], gbif: dict, sink_rows: int, species_csv: Path, names_sha256: str) -> dict:
    """The "gbif" header block (fpack.py): per pack row the COL species key and accepted name of its
    TreeOfLife name, joined by that name as insect-detect-post does, so both programs give a
    track the same key. Sink rows and names without a key get null. Rows with a name rule
    (name_rules.rows) carry the key of their published name too: it places them in the country
    lists, but the app answers them only at genus or family, so it is not a track's species."""
    meta_path = species_csv.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tol = ((meta.get("mapping") or {}).get("tol_species_list") or {}).get("sha256")
    if tol and tol != names_sha256:
        raise SystemExit(f"the mapping behind {species_csv.name} was made from another TreeOfLife names file "
                         f"({tol[:12]}...) than the pack ({names_sha256[:12]}...)")
    pairs = [gbif.get(n, (None, None)) for n in row_names] + [(None, None)] * sink_rows
    return {
        "species_key": [k for k, _ in pairs],
        "species": [s for _, s in pairs],
        "source": {"species_csv": species_csv.name, "orders": meta.get("orders"),
                   **({"classes": meta["classes"]} if meta.get("classes") else {}),
                   "gbif_checklist_key": meta.get("gbif_checklist_key"), "mapping": meta.get("mapping")},
    }


def regions_block(keys: list, row_names: list[str], lists_path: Path, neighbours_path: Path, gbif_source: dict,
                  every_country: str | None = None, short_list_names: int | None = None) -> dict:
    """The "regions" header block (fpack.py): per country the pack rows whose GBIF species has at
    least min_occurrences records there (build_country_lists.py), as bits, and the neighbour table
    (country_neighbours.py). Countries without any row of the pack are left out. [every_country]
    (a key of EVERY_COUNTRY_SETS) puts its rows on every country's list, recorded in the block.
    [short_list_names]: below this many names (country with its neighbours) the pack's lists count
    as short, for an app that reads it (in November 2026 the app uses 5,000 for every pack, a number
    fitted to the pollinator lists; a mammals-and-birds country has a few hundred to 4,000 names)."""
    lists = json.loads(lists_path.read_text())
    # names_in_country counts the names of the lists' orders, and the keys must come from the same
    # mapping table: both must be those of the species list.
    if not taxa_of(gbif_source) or not gbif_source.get("mapping"):
        raise SystemExit(f"no .json next to {gbif_source['species_csv']}: cannot check that {lists_path.name} "
                         "is for the same orders and mapping")
    if taxa_of(lists) != taxa_of(gbif_source):
        raise SystemExit(f"{lists_path} is for {taxa_of(lists)}, the species list for {taxa_of(gbif_source)}")
    if lists["mapping"]["sha256"] != gbif_source["mapping"]["sha256"]:
        raise SystemExit(f"{lists_path} and {gbif_source['species_csv']} were made with different mapping tables")
    n = len(keys)
    everywhere: list[int] = []
    if every_country:
        names, reason = EVERY_COUNTRY_SETS[every_country]
        everywhere = [i for i, name in enumerate(row_names) if name in set(names)]
        missing = set(names) - {row_names[i] for i in everywhere}
        if missing:
            raise SystemExit(f"--in-every-country {every_country}: no pack row for {sorted(missing)}")
        keyless = sorted({row_names[i] for i in everywhere if keys[i] is None})
        if keyless:  # a country's rows all carry a GBIF key (check_pack_blocks.py)
            raise SystemExit(f"--in-every-country {every_country}: no GBIF species key for {keyless}")
    countries = {}
    for code, c in sorted(lists["countries"].items()):
        present = c["keys"]
        listed = {i for i, k in enumerate(keys) if k is not None and k in present}
        rows = sorted(listed | set(everywhere))
        if rows:
            # Names put on the list by [every_country] count as names of the country's list too.
            added = len({row_names[i] for i in everywhere} - {row_names[i] for i in listed})
            countries[code] = {"rows": rows_to_bits(rows, n), "names": len({row_names[i] for i in rows}),
                               "names_in_country": c["names_in_country"] + added}
    nb_meta = json.loads(neighbours_path.with_suffix(".json").read_text())
    neighbours: dict[str, list[str]] = {}
    with open(neighbours_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for a, b in ((r["country"], r["neighbour"]), (r["neighbour"], r["country"])):
                if a in countries:
                    neighbours.setdefault(a, []).append(b)
    return {
        "min_occurrences": lists["min_occurrences"],
        **({"occurrence_status": lists["occurrence_status"]} if lists.get("occurrence_status") else {}),
        **({"basis_of_record_excluded": lists["basis_of_record_excluded"]}
           if lists.get("basis_of_record_excluded") else {}),
        "countries": countries,
        "neighbours": {c: sorted(v) for c, v in sorted(neighbours.items())},
        "neighbours_rule": nb_meta["rule"],
        **({"in_every_country": {"set": every_country, "names": EVERY_COUNTRY_SETS[every_country][0],
                                 "rows": everywhere, "reason": EVERY_COUNTRY_SETS[every_country][1]}}
           if every_country else {}),
        **({"extinct_left_out": lists["extinct_left_out"]["rule"]} if lists.get("extinct_left_out") else {}),
        **({"short_list_names": short_list_names} if short_list_names else {}),
        "source": {"gbif_query_date": lists["date"], "gbif_checklist_key": lists["gbif_checklist_key"],
                   **({"gbif_col_release": lists["gbif_col_release"]} if lists.get("gbif_col_release") else {}),
                   **({"orders": lists["orders"]} if "orders" in lists else {"classes": lists["classes"]}),
                   "mapping": lists["mapping"], "neighbours": nb_meta["source"],
                   **({"neighbours_by_hand": nb_meta["manual_pairs"]} if nb_meta.get("manual_pairs") else {})},
    }


def sink_embeddings(model_key: str, prompts: list[str], weights: Path | None = None) -> np.ndarray:
    """Embed the negative prompts with the model's text tower (unit vectors)."""
    import torch
    import open_clip

    from export_image_tower import ARCH, load_open_clip

    model = load_open_clip(model_key, weights)
    tokenizer = open_clip.get_tokenizer(ARCH[model_key] if weights else MODEL_HUB[model_key])
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts))
        t = torch.nn.functional.normalize(t, dim=-1)
    return t.cpu().numpy().astype(np.float32), float(model.logit_scale.exp().item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(EMBEDDINGS), default="bioclip-2")
    ap.add_argument("--classes", default="Insecta,Arachnida", help="comma list of class names to keep ('' = all)")
    ap.add_argument("--orders", default="", help="comma list of order names to keep (optional, narrows further)")
    ap.add_argument("--families", default="", help="comma list of family names to keep (optional, narrows further)")
    ap.add_argument("--species-csv", type=Path, default=None,
                    help="CSV with a 'species' column ('Genus epithet'); keeps only those rows")
    ap.add_argument("--pack-id", default=None, help="identifier stored in the pack (default derived)")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--no-sink", action="store_true", help="skip the 'none of these' rows")
    ap.add_argument("--sink-set", choices=sorted(SINK_SETS), default="arthropod",
                    help="which 'none of these' prompts to embed (arthropod: flower scenes; mammal: camera-trap scenes)")
    ap.add_argument("--logit-scale", type=float, default=None,
                    help="override (normally read from the model when sink rows are built; 100 otherwise)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Hugging Face download cache folder")
    ap.add_argument("--embeddings-dir", type=Path, default=None,
                    help="folder that already holds the TreeOfLife .npy + .json (e.g. downloaded before from "
                         "imageomics/TreeOfLife-200M) instead of the Hugging Face download")
    ap.add_argument("--weights", type=Path, default=None,
                    help="model checkpoint file on disk, for the text tower of the sink rows")
    ap.add_argument("--name-rules", choices=["higher-rank", "none"], default="higher-rank",
                    help="higher-rank (default): rows whose names are not in the plain form are used only down to "
                         "genus or family, each listed in the header (see NAME_RULES); none: every row used as "
                         "published (the v1 packs)")
    ap.add_argument("--country-lists", type=Path, default=None,
                    help="output of build_country_lists.py: writes the 'regions' block (which rows GBIF lists per "
                         "country), so the app can narrow the pack to a country and its neighbours")
    ap.add_argument("--neighbours", type=Path, default=Path(__file__).with_name("country_neighbours.csv"),
                    help="neighbour table for --country-lists (default: country_neighbours.csv next to this script)")
    ap.add_argument("--in-every-country", choices=sorted(EVERY_COUNTRY_SETS), default=None,
                    help="with --country-lists: put these rows on every country's list (see EVERY_COUNTRY_SETS)")
    ap.add_argument("--short-list-names", type=int, default=None,
                    help="with --country-lists: below this many names a country with its neighbours counts as short "
                         "(header regions.short_list_names; the app's own default without it: 5,000)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    emb, names, names_sha256 = load_tol(args.model, args.cache_dir, args.embeddings_dir)
    dim, n = emb.shape if emb.shape[0] < emb.shape[1] else (emb.shape[1], emb.shape[0])
    transposed = emb.shape[0] == dim and emb.shape[0] < emb.shape[1]
    print(f"TreeOfLife embeddings: {n} rows, dim {dim} (layout {'[dim, N]' if transposed else '[N, dim]'})")
    if len(names) != n:
        print(f"WARNING: json has {len(names)} entries but the matrix has {n} columns", file=sys.stderr)

    keep_classes = {c.strip() for c in args.classes.split(",") if c.strip()}
    keep_orders = {o.strip() for o in args.orders.split(",") if o.strip()}
    keep_families = {f.strip() for f in args.families.split(",") if f.strip()}
    keep_species = species_sha256 = None
    gbif: dict[str, tuple[str, str | None]] = {}
    if args.species_csv is not None:
        species_sha256 = hashlib.sha256(args.species_csv.read_bytes()).hexdigest()
        keep_species, gbif = read_species_csv(args.species_csv)
    if args.country_lists and not gbif:
        print("--country-lists needs a --species-csv with gbif_species_key (build_region_species_list.py)", file=sys.stderr)
        return 2

    # The filters read the names as published, so --name-rules never changes which rows a pack holds.
    labels: list[list[str]] = []
    row_names: list[str] = []
    idx: list[int] = []
    rule_rows: list[dict] = []
    for i in range(min(n, len(names))):
        tax = row_taxonomy(names[i])
        if keep_classes and tax[2] not in keep_classes:
            continue
        if keep_orders and tax[3] not in keep_orders:
            continue
        if keep_families and tax[4] not in keep_families:
            continue
        name = f"{tax[5]} {tax[6]}".strip()
        if keep_species is not None and name not in keep_species:
            continue
        row_names.append(name)  # as published, before any name rule
        rule = name_rule(tax) if args.name_rules == "higher-rank" else None
        if rule:
            rule_rows.append({"pack_row": len(labels), "tol_row": i, "rule": rule, "published": tax})
            tax = apply_name_rule(tax, rule)
        labels.append(tax)
        idx.append(i)
    if not idx:
        print("No rows matched the filters.", file=sys.stderr)
        return 2
    rule_counts = {k: sum(r["rule"] == k for r in rule_rows) for k in NAME_RULES}
    print(f"kept {len(idx)} species rows; name rules '{args.name_rules}'"
          + (f": {rule_counts}" if args.name_rules != "none" else ""))

    idx_arr = np.asarray(idx)
    mat = (emb[:, idx_arr].T if transposed else emb[idx_arr, :]).astype(np.float32)

    logit_scale = args.logit_scale
    sink_rows = 0
    sink_prompts = [] if args.no_sink else SINK_SETS[args.sink_set]
    if sink_prompts:
        sink, model_scale = sink_embeddings(args.model, [p for _, p in sink_prompts], args.weights)
        mat = np.concatenate([mat, sink], axis=0)
        labels += [sink_label(k, p) for k, p in sink_prompts]
        sink_rows = len(sink_prompts)
        if logit_scale is None:
            logit_scale = model_scale
    if logit_scale is None:
        logit_scale = 100.0

    # Naming rule (the app's tool/model_downloads/README.md, round 276): <model>_<list>_v<n>.fpack, so the
    # app pairs the pack with every file of its model by the first part of the name.
    words = sorted(keep_classes) or ["all"]
    words += sorted(keep_orders)
    if keep_families:
        words.append(f"{len(keep_families)}fam")
    if args.species_csv:
        words.append(args.species_csv.stem)
    pack_id = args.pack_id or f"{args.model}_{'-'.join(w.lower().replace('_', '-') for w in words)}_v1"
    if not pack_id.startswith(f"{args.model}_"):
        print(f"note: '{pack_id}' does not start with '{args.model}_'; the app then pairs it with its model "
              "only through the header (naming rule: the app's tool/model_downloads/README.md)")
    header = {
        "kind": "label_pack",  # round 276: the word of the app's assets/model_downloads.json
        "pack_id": pack_id,
        "model_id": args.model,
        "logit_scale": float(logit_scale),
        "temperature": 1.0,
        "ranks": RANKS,
        "sink_rows": sink_rows,
        "sink_set": None if not sink_prompts else args.sink_set,
        "labels": labels,
        "source": {
            "embeddings_repo": EMBEDDINGS[args.model][0],
            "embeddings_file": EMBEDDINGS[args.model][1],
            "names_file_sha256": names_sha256,
            "classes": sorted(keep_classes), "orders": sorted(keep_orders),
            "families": sorted(keep_families),
            "species_csv": args.species_csv.name if args.species_csv else None,
            "species_csv_sha256": species_sha256,
        },
        # Which rows the app uses only down to a higher rank, with the names as published
        # (tol_row = row of the TreeOfLife names file), so every change can be checked and undone.
        "name_rules": {
            "set": args.name_rules,
            "rules": {} if args.name_rules == "none" else {k: v[1] for k, v in NAME_RULES.items()},
            "counts": {} if args.name_rules == "none" else rule_counts,
            "rows": rule_rows,
        },
        "built_with": {"script": "build_label_pack.py", "date": date.today().isoformat()},
        "license": "TreeOfLife embeddings CC0-1.0; sink rows generated with the MIT BioCLIP text tower",
    }
    if gbif:
        header["gbif"] = gbif_block(row_names, gbif, sink_rows, args.species_csv, names_sha256)
        print(f"gbif: {sum(k is not None for k in header['gbif']['species_key'])} of {len(row_names)} species rows "
              "have a GBIF species key")
    if args.country_lists:
        header["regions"] = regions_block(header["gbif"]["species_key"], row_names, args.country_lists,
                                          args.neighbours, header["gbif"]["source"], args.in_every_country,
                                          args.short_list_names)
        reg = header["regions"]
        print(f"regions: {len(reg['countries'])} countries with rows, neighbours of {len(reg['neighbours'])}")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{pack_id}.fpack"
    write_fpack(out_path, header, mat, dtype=args.dtype)
    print(f"Pack written: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(labels)} rows incl. {sink_rows} sink rows)")
    from check_pack_blocks import app_accepts_header, raw_parts

    try:
        app_accepts_header(raw_parts(out_path)[0])
    except ValueError as e:
        # Kept for PC use (e.g. a world pack), but never to be offered to the app.
        print(f"WARNING: no released app version can open this pack: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
