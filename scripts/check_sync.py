#!/usr/bin/env python3
"""Check that the curated copies in this repository still match the parent module.

This repository is a curated deployment of the research module in
``ElijahZhao/MetaNutri---AI-``. The files below are copied by hand; nothing else
prevents them from drifting apart. This script compares them against the parent
repository's ``main`` branch.

How it works: one GitHub API request resolves the parent tree to git blob SHAs.
Git blob SHAs are computed locally from the checked-out bytes, so no file has to
be downloaded and only stdlib is required.

An unauthenticated request is limited to 60 per hour and CI runners share IPs,
so a token is used when one is present: ``GITHUB_TOKEN`` (set automatically in
GitHub Actions) or ``GH_TOKEN`` (local use). Without a token the script still
works, it is just rate-limit prone.

Exit codes:
    0  every curated file is byte-identical to the parent
    1  at least one file differs (parent moved ahead, or the copy drifted)
    2  the parent tree could not be fetched (network / API failure)

Usage:
    python scripts/check_sync.py
    PARENT_REF=v1.2.0 python scripts/check_sync.py   # compare against a tag
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PARENT = "ElijahZhao/MetaNutri---AI-"
REF = os.environ.get("PARENT_REF", "main")

ROOT = Path(__file__).resolve().parent.parent

# (path in the parent repo, path in this repo)
SRC_FILES = [
    "build_dataset.py",
    "build_external.py",
    "ceiling_experiments.py",
    "download_bigideas.py",
    "download_data.py",
    "download_data.sh",
    "evaluate.py",
    "experiment.py",
    "export_report_pdf.py",
    "external_validate.py",
    "iauc.py",
    "make_figures.py",
    "mixed_effects.py",
    "run_all.sh",
    "train_final.py",
    "uncertainty.py",
]

EXPERIMENT_FILES = [
    "ceiling_results.csv",
    "external_results.csv",
    "mixed_effects_coefficients.csv",
    "mixed_effects_variance.csv",
    "results.csv",
    "uncertainty.csv",
]

REPORT_FILES = [
    "technical_report.md",
    "technical_report.pdf",
]

FIGURE_FILES = [f"fig{i}_{name}.png" for i, name in [
    (1, "pred_vs_actual"),
    (2, "model_comparison"),
    (3, "data_overview"),
    (4, "feature_importance"),
    (5, "external_validation"),
    (6, "within_between"),
    (7, "conformal"),
    (8, "mixed_effects"),
]]

PAIRS: list[tuple[str, str]] = (
    [(f"research/src/{name}", f"src/{name}") for name in SRC_FILES]
    + [(f"research/experiments/{name}", f"experiments/{name}") for name in EXPERIMENT_FILES]
    + [(f"research/reports/{name}", f"reports/{name}") for name in REPORT_FILES]
    + [(f"research/reports/figures/{name}", f"reports/figures/{name}") for name in FIGURE_FILES]
)


def local_blob_sha(path: Path) -> str:
    """Git blob SHA of a file, matching what the trees API reports."""
    data = path.read_bytes()
    header = b"blob %d\0" % len(data)
    return hashlib.sha1(header + data).hexdigest()


def fetch_parent_tree(ref: str, attempts: int = 3) -> dict[str, str]:
    url = (
        f"https://api.github.com/repos/{PARENT}/git/trees/{ref}?recursive=1"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "metanutri-sync-check",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("truncated"):
                print("warning: parent tree response was truncated", file=sys.stderr)
            return {
                entry["path"]: entry["sha"]
                for entry in payload.get("tree", [])
                if entry.get("type") == "blob"
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(2 * attempt)
    print(f"FAIL  could not fetch parent tree ({last_error})", file=sys.stderr)
    print("      this is a network/API failure, not a drift finding", file=sys.stderr)
    return {}


def main() -> int:
    print(f"parent: {PARENT} @ {REF}  ({len(PAIRS)} curated files)")

    tree = fetch_parent_tree(REF)
    if not tree:
        return 2

    missing_local: list[str] = []
    missing_parent: list[str] = []
    drifted: list[tuple[str, str, str]] = []
    checked = 0

    for parent_path, local_path in PAIRS:
        local = ROOT / local_path
        if not local.is_file():
            missing_local.append(local_path)
            continue
        parent_sha = tree.get(parent_path)
        if parent_sha is None:
            missing_parent.append(parent_path)
            continue
        local_sha = local_blob_sha(local)
        checked += 1
        if local_sha != parent_sha:
            drifted.append((local_path, local_sha, parent_sha))

    for path in missing_local:
        print(f"MISS  {path}  (curated file absent from this repository)")
    for path in missing_parent:
        print(f"GONE  {path}  (path no longer exists in the parent)")

    if drifted:
        print("\nDRIFT  local bytes differ from the parent:")
        for path, local_sha, parent_sha in drifted:
            print(f"  {path}")
            print(f"    local  {local_sha[:12]}")
            print(f"    parent {parent_sha[:12]}")
        print(
            "\nRe-copy the files from the parent module, or update this script if the\n"
            "curated set itself is meant to change."
        )

    failed = bool(missing_local or missing_parent or drifted)
    print(
        f"\n{'FAIL' if failed else 'PASS'}  "
        f"{checked} files compared, {len(drifted)} drifted, "
        f"{len(missing_local)} missing locally, {len(missing_parent)} gone upstream"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
