#!/usr/bin/env python3
"""Publish converted models on Hugging Face, one repository per model, and verify the links.

The FaunaPulse app downloads every file through a link pinned to one Hugging Face commit and
rejects a file whose SHA-256 differs from the one in its bundled catalogue
(assets/model_downloads.json in the app repository). So this script:

  * checks every local file against that catalogue BEFORE anything is uploaded;
  * never replaces a published file (only the model card, README.md, may change);
  * writes publish/published.json: per file the commit it was added in and its pinned URL
    https://huggingface.co/<owner>/<repo>/resolve/<commit>/<file>, which goes into the app.

What goes where is in publish/hf_repos.json; the model cards are hf_cards/<repo>/README.md
with two placeholders filled here: {{FILES_TABLE}} (from the checked files) and
{{CODE_COMMIT}} (this repository's commit, which must already be pushed).

Usage (from the repository root, shared environment active; log in once with `hf auth login`):
    python publish/publish_hf.py --catalogue "$APP/assets/model_downloads.json"            # dry run, all repos
    python publish/publish_hf.py --catalogue ... --write-cards /tmp/cards                  # + rendered cards to read
    python publish/publish_hf.py --catalogue ... --upload --repo flatbug-tflite --private   # one repo per step
    python publish/publish_hf.py --verify [--full]     # anonymous checks of published.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "publish" / "hf_repos.json"
PUBLISHED = ROOT / "publish" / "published.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_id(path: Path) -> str:
    """Git's id of a file's content: how the Hub identifies small (non-LFS) files."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def catalogue_files(catalogue: Path) -> dict:
    """name -> {name, bytes, sha256} of every file the app's catalogue offers."""
    cat = json.loads(catalogue.read_text(encoding="utf-8"))
    out = {}
    for model in cat.get("models", []):
        entries = [model.get("file")] + [nl.get("file") for nl in model.get("name_lists") or []]
        for f in filter(None, entries):
            out[f["name"]] = {"name": f["name"], "bytes": f.get("bytes"), "sha256": f.get("sha256")}
    return out


def is_pushed(sha: str) -> bool:
    """True when a branch on the remote (asked live, not the possibly stale local origin/* refs) contains sha."""
    git = ["git", "-C", str(ROOT)]
    try:
        heads = subprocess.check_output(git + ["ls-remote", "--heads", "origin"], text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return False
    for line in heads.splitlines():
        head = line.split()[0]
        known = subprocess.run(git + ["cat-file", "-e", head], capture_output=True).returncode == 0
        if known and subprocess.run(git + ["merge-base", "--is-ancestor", sha, head]).returncode == 0:
            return True
    return False


def code_commit(require_pushed: bool) -> str:
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if not is_pushed(sha):
        msg = f"commit {sha[:12]} is not pushed: the model cards would link to code nobody can see"
        if require_pushed:
            sys.exit("ERROR: " + msg)
        print("WARNING: " + msg)
    return sha


def check_repo(name: str, cfg: dict, offered: dict) -> list[dict]:
    """Size and SHA-256 of every local file; catalogue files must match the catalogue exactly.

    A file is found in the catalogue by its name or, when it is published under a new name
    (the naming rule), by its SHA-256; the app's catalogue then takes the new name (its update_catalogue.py --urls, app Round 312)."""
    by_sha = {w["sha256"]: w for w in offered.values() if w.get("sha256")}
    files, ok = [], True
    for f in cfg["files"]:
        local = ROOT / f["local"]
        if not local.is_file():
            print(f"  MISSING  {f['name']}  ({f['local']})")
            ok = False
            continue
        info = {**f, "path": local, "bytes": local.stat().st_size, "sha256": sha256_of(local)}
        want = offered.get(f["name"])
        same = by_sha.get(info["sha256"])
        if want is None and same is None:
            state = "extra (not in the app catalogue)"
        elif want is None:
            if same["bytes"] != info["bytes"]:
                state = f"DIFFERS from the app catalogue's {same['name']} (same SHA-256, other size)"
                ok = False
            else:
                state = f"same bytes as the app catalogue's {same['name']} (published under the new name)"
                info["catalogue_name"] = same["name"]
        elif (want["bytes"], want["sha256"]) == (info["bytes"], info["sha256"]):
            state = "matches the app catalogue"
            info["catalogue_name"] = want["name"]
        else:
            state = f"DIFFERS from the app catalogue ({want['bytes']} bytes, {str(want['sha256'])[:12]}...)"
            ok = False
        print(f"  {info['bytes']:>13,}  {info['sha256'][:12]}  {f['name']}: {state}")
        files.append(info)
    if not ok:
        sys.exit(f"ERROR: {name}: fix the files above before publishing")
    return files


def files_table(files: list[dict]) -> str:
    rows = ["| File | Bytes | SHA-256 | Input | Precision | Role |", "|---|---|---|---|---|---|"]
    for f in files:
        if f["name"] == "LICENSE":  # the card's licence section names it
            continue
        px = f"{f['input_px']} px" if f.get("input_px") else ""
        rows.append(f"| `{f['name']}` | {f['bytes']:,} | `{f['sha256']}` | {px} | {f.get('precision', '')} | {f['role']} |")
    return "\n".join(rows)


def render_card(cfg: dict, files: list[dict], commit: str) -> str:
    text = (ROOT / cfg["card"]).read_text(encoding="utf-8")
    text = text.replace("{{FILES_TABLE}}", files_table(files)).replace("{{CODE_COMMIT}}", commit)
    if re.search(r"\{\{[A-Z_]+\}\}", text):  # not BibTeX's {{Name}} braces
        sys.exit(f"ERROR: {cfg['card']} has an unfilled placeholder")
    return text


def upload(api, owner: str, collection: dict, name: str, cfg: dict, files: list[dict], card: str,
           private: bool, published: dict) -> None:
    from huggingface_hub import CommitOperationAdd, ModelCard

    ModelCard(card).validate()  # the Hub's own check of the card's metadata
    repo_id = f"{owner}/{name}"
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    existing = {p.path: p for p in api.get_paths_info(repo_id, [f["name"] for f in files], expand=True)}

    ops, new_names = [], []
    for f in files:
        old = existing.get(f["name"])
        if old is not None:
            same = old.lfs.sha256 == f["sha256"] if old.lfs else old.blob_id == git_blob_id(f["path"])
            if not same:
                sys.exit(f"ERROR: {repo_id}/{f['name']} is already published with different content. "
                         "Published files are never replaced: give the new file a new name.")
            continue
        ops.append(CommitOperationAdd(path_in_repo=f["name"], path_or_fileobj=str(f["path"])))
        new_names.append(f["name"])
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")))

    info = api.create_commit(repo_id, operations=ops, repo_type="model",
                             commit_message=f"Add {', '.join(new_names)}" if new_names else "Update model card")
    print(f"  committed {info.oid} ({len(new_names)} new file(s) + model card)")

    entry = published.setdefault(repo_id, {"files": {}})
    for f in files:
        if f["name"] in new_names:  # a file keeps the commit it was first published in
            entry["files"][f["name"]] = {
                "commit": info.oid, "bytes": f["bytes"], "sha256": f["sha256"],
                "url": f"https://huggingface.co/{repo_id}/resolve/{info.oid}/{f['name']}",
                **({"catalogue_name": f["catalogue_name"]} if f.get("catalogue_name") not in (None, f["name"]) else {}),
            }
    entry["card_commit"] = info.oid
    PUBLISHED.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")

    col = api.create_collection(title=collection["title"], namespace=owner,
                                description=collection["description"], private=private, exists_ok=True)
    api.add_collection_item(col.slug, item_id=repo_id, item_type="model", exists_ok=True)
    print(f"  in collection https://huggingface.co/collections/{col.slug}")


def verify(published: dict, full: bool) -> None:
    """Anonymous checks, as the app sees the files (public repos, no token)."""
    from huggingface_hub import HfApi

    api, bad = HfApi(token=False), 0
    for repo_id, entry in published.items():
        for name, f in entry["files"].items():
            info = api.get_paths_info(repo_id, [name], revision=f["commit"], expand=True)
            hub = info[0] if info else None
            problems = []
            if hub is None:
                problems.append("not found at its pinned commit")
            elif hub.size != f["bytes"]:
                problems.append(f"size {hub.size} on the Hub")
            elif hub.lfs and hub.lfs.sha256 != f["sha256"]:
                problems.append("SHA-256 on the Hub differs")
            if full and not problems:
                problems += stream_check(f["url"], f["sha256"])
            bad += bool(problems)
            print(f"  {'OK  ' if not problems else 'FAIL'} {repo_id}/{name} {'; '.join(problems)}")
    if bad:
        sys.exit(f"ERROR: {bad} file(s) failed")


def stream_check(url: str, sha256: str) -> list[str]:
    """Download through the pinned link like the app does: HTTPS-only redirects, status 200, same SHA-256."""
    import requests

    h = hashlib.sha256()
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        hops = [x.url for x in r.history] + [r.url]
        if any(not u.startswith("https://") for u in hops):
            return ["a redirect leaves HTTPS"]
        if r.status_code != 200:
            return [f"HTTP {r.status_code}"]
        for chunk in r.iter_content(1 << 20):
            h.update(chunk)
    return [] if h.hexdigest() == sha256 else ["downloaded SHA-256 differs"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", type=Path, help="the app's assets/model_downloads.json (read only)")
    ap.add_argument("--repo", action="append", help="only this repository (repeatable); default: all")
    ap.add_argument("--upload", action="store_true", help="upload (default: dry run, nothing leaves this computer)")
    ap.add_argument("--private", action="store_true", help="create new repositories as private (switch to public on the Hub after review)")
    ap.add_argument("--verify", action="store_true", help="check the published files listed in publish/published.json")
    ap.add_argument("--full", action="store_true", help="with --verify: also download and hash every file")
    ap.add_argument("--write-cards", type=Path, metavar="DIR",
                    help="dry run: also write each rendered card to DIR/<repo>/README.md for review")
    args = ap.parse_args()
    if args.upload and len(args.repo or []) != 1:
        ap.error("--upload takes exactly one --repo: one repository per step, after its card was reviewed")

    published = json.loads(PUBLISHED.read_text(encoding="utf-8")) if PUBLISHED.exists() else {}
    if args.verify:
        verify(published, args.full)
        return 0
    if args.catalogue is None:
        ap.error("--catalogue is required (except with --verify)")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    offered = catalogue_files(args.catalogue.expanduser())
    names = args.repo or list(config["repos"])
    unknown = set(names) - set(config["repos"])
    if unknown:
        ap.error(f"unknown repo(s): {', '.join(sorted(unknown))}")
    commit = code_commit(require_pushed=args.upload)

    api = None
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"Uploading as {api.whoami()['name']}")
    for name in names:
        cfg = config["repos"][name]
        print(f"\n{config['owner']}/{name}")
        files = check_repo(name, cfg, offered)
        card = render_card(cfg, files, commit)
        if api is None:
            from huggingface_hub import ModelCard
            ModelCard(card).validate()  # the Hub's check of the metadata (read-only request)
            print(f"  dry run: card renders ({len(card):,} characters), metadata valid; nothing uploaded")
            if args.write_cards:
                out = args.write_cards / name / "README.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(card, encoding="utf-8")
                print(f"  card written to {out}")
        else:
            upload(api, config["owner"], config["collection"], name, cfg, files, card, args.private, published)
    return 0


if __name__ == "__main__":
    sys.exit(main())
