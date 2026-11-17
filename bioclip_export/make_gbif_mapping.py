#!/usr/bin/env python3
"""Make the TreeOfLife-to-GBIF species mapping for the Catalogue of Life release GBIF uses today,
by running insect-detect-post's own mapping script as it is.

Why: build_region_species_list.py and build_country_lists.py join TreeOfLife names to GBIF species
keys with the mapping that Maximilian Sittinger publishes with insect-detect-post (release v1.1.0,
made against COL26.6 XR). GBIF moves its checklist to newer Catalogue of Life releases, and a
species can get a new key in a new release, so those two scripts stop when GBIF's release is not
the mapping's. His script filters/build_bioclip_gbif_species_mapping.py is written to be run
again for a newer release. This runner calls it unchanged (none of its code is copied here) and
only points its two output paths at --out, so his clone stays as it is. The result is not his
published release: lists and packs made with it say so (their "mapping" record holds the file's
SHA-256, the COL release and the commit of his repository written to <mapping>.run.json).

Needs his repository and his environment, not this repository's .venvs/export:
    git clone https://github.com/maxsitt/insect-detect-post /path/to/insect-detect-post
    cd /path/to/insect-detect-post && uv sync --frozen --extra cpu
    /path/to/insect-detect-post/.venv/bin/python make_gbif_mapping.py \
        --clone /path/to/insect-detect-post --out out/mapping_col26.10xr
Then pass the CSV to the list scripts:
    python build_region_species_list.py --mapping-csv out/mapping_col26.10xr/bioclip_gbif_species_mapping_Arthropoda.csv ...

One name-matching request to GBIF per TreeOfLife arthropod name (about 300,000, 64 at a time; his
script retries failed ones when run again).

--model bioclip-2.5 maps the names file of BioCLIP 2.5 (embeddings/txt_emb_bioclip-2.5-vith14.json
of the same dataset) in the same way; his script and pybioclip know only the BioCLIP 2 file, so
three of his script's settings are set here for that run (the names file, the model name it
records, and the dataset the file is in).

Credit: Sittinger, M. (2026). Software for post-processing of data captured with the Insect Detect
camera trap. Zenodo. https://doi.org/10.5281/zenodo.21822140 (AGPL-3.0;
https://github.com/maxsitt/insect-detect-post).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

MODELS = {
    # model key -> (names file in imageomics/TreeOfLife-200M, model name written to the provenance record)
    "bioclip-2": (None, None),  # his script's own settings
    "bioclip-2.5": ("embeddings/txt_emb_bioclip-2.5-vith14.json", "hf-hub:imageomics/bioclip-2.5-vith14"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clone", type=Path, required=True, help="local clone of insect-detect-post")
    ap.add_argument("--out", type=Path, required=True, help="folder for the mapping CSV and its .json")
    ap.add_argument("--model", choices=sorted(MODELS), default="bioclip-2", help="whose TreeOfLife names file")
    args = ap.parse_args()

    def git(*cmd: str) -> str:
        return subprocess.run(["git", "-C", str(args.clone), *cmd], capture_output=True, text=True, check=True).stdout.strip()

    commit, changed = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    sys.path.insert(0, str(args.clone / "filters"))
    import build_bioclip_gbif_species_mapping as his  # needs his environment (polars, pygbif, pybioclip)

    args.out.mkdir(parents=True, exist_ok=True)
    his.OUTPUT_CSV = args.out / his.OUTPUT_CSV.name
    his.OUTPUT_JSON = his.OUTPUT_CSV.with_suffix(".json")
    names_file, model_name = MODELS[args.model]
    if names_file:
        repo = his.get_tol_repo_id(his.BIOCLIP_MODEL_STR)  # the BioCLIP 2 dataset also holds the 2.5 names file
        his.TOL_SPECIES_FILE, his.BIOCLIP_MODEL_STR, his.get_tol_repo_id = names_file, model_name, lambda _model: repo
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not his.build_bioclip_gbif_species_mapping():
        return 1
    run = {"made_with": "insect-detect-post, filters/build_bioclip_gbif_species_mapping.py, run by make_gbif_mapping.py",
           "insect_detect_post_commit": commit, "insect_detect_post_changed_files": changed,
           "model": args.model, "date": date.today().isoformat()}
    his.OUTPUT_CSV.with_suffix(".run.json").write_text(json.dumps(run, indent=2))
    print(f"wrote {his.OUTPUT_CSV} (+ .json, .run.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
