"""
Dataset Download Script for: "Do You Mean...?": Fixing User Intent Without Annoying Them

Downloads datasets cataloged in dataset_catalog.json that are available via
HuggingFace Datasets or direct Git clones.

Usage:
    python datasets/download_datasets.py --all           # Download all datasets
    python datasets/download_datasets.py --ids 7 8 10    # Download specific datasets by ID
    python datasets/download_datasets.py --priority HIGH  # Download only HIGH priority
    python datasets/download_datasets.py --list           # List available datasets
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CATALOG_PATH = SCRIPT_DIR / "dataset_catalog.json"
DATA_DIR = SCRIPT_DIR / "raw"


def load_catalog():
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)


def list_datasets(catalog):
    print(f"\n{'ID':>4}  {'Priority':<8}  {'Category':<40}  {'Name'}")
    print("-" * 110)
    for ds in catalog["datasets"]:
        print(
            f"{ds['id']:>4}  {ds['priority']:<8}  {ds['category']:<40}  {ds['name']}"
        )
    print(f"\nTotal: {len(catalog['datasets'])} datasets\n")


def download_huggingface(ds, output_dir):
    """Download a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] 'datasets' library not installed. Run: pip install datasets")
        return False

    hf_url = ds["huggingface_url"]
    cmd = ds["download_command"]

    # Parse the load_dataset call to extract arguments
    # Format: load_dataset('name') or load_dataset('name', 'config')
    if "load_dataset(" not in cmd:
        print(f"  [SKIP] Non-standard download command: {cmd}")
        return False

    # Extract arguments from the load_dataset call
    call_part = cmd.split("load_dataset(")[1].rstrip(")")
    args = [a.strip().strip("'\"") for a in call_part.split(",")]

    dataset_name = args[0]
    config = args[1] if len(args) > 1 else None

    print(f"  Loading from HuggingFace: {dataset_name}" + (f" ({config})" if config else ""))

    try:
        if config:
            dataset = load_dataset(dataset_name, config)
        else:
            dataset = load_dataset(dataset_name)

        save_path = output_dir / f"hf_{dataset_name.replace('/', '_')}"
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"  [OK] Saved to {save_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        return False


def download_git(ds, output_dir):
    """Clone a Git repository."""
    source_url = ds["source_url"]
    if "github.com" not in source_url:
        print(f"  [SKIP] Not a GitHub URL: {source_url}")
        return False

    repo_name = source_url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = output_dir / f"git_{repo_name}"

    if clone_path.exists():
        print(f"  [SKIP] Already cloned: {clone_path}")
        return True

    print(f"  Cloning: {source_url}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", source_url, str(clone_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  [OK] Cloned to {clone_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Git clone failed: {e.stderr}")
        return False


def download_dataset(ds, output_dir):
    """Download a single dataset using the appropriate method."""
    print(f"\n[{ds['id']}] {ds['name']} ({ds['category']})")

    if ds.get("huggingface_url"):
        return download_huggingface(ds, output_dir)
    elif "git clone" in ds.get("download_command", ""):
        return download_git(ds, output_dir)
    else:
        print(f"  [MANUAL] Requires manual download: {ds.get('download_command', ds['source_url'])}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for intent preservation research"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--ids", type=int, nargs="+", help="Download specific dataset IDs"
    )
    parser.add_argument(
        "--priority",
        choices=["HIGH", "MEDIUM"],
        help="Download datasets of given priority",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR),
        help="Output directory for downloads",
    )

    args = parser.parse_args()
    catalog = load_catalog()

    if args.list:
        list_datasets(catalog)
        return

    if not (args.all or args.ids or args.priority):
        parser.print_help()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = catalog["datasets"]
    if args.ids:
        datasets_to_download = [d for d in datasets_to_download if d["id"] in args.ids]
    elif args.priority:
        datasets_to_download = [
            d for d in datasets_to_download if d["priority"] == args.priority
        ]

    print(f"Downloading {len(datasets_to_download)} datasets to {output_dir}")

    results = {"success": [], "failed": [], "manual": []}
    for ds in datasets_to_download:
        success = download_dataset(ds, output_dir)
        if success:
            results["success"].append(ds["name"])
        elif ds.get("huggingface_url") or "git clone" in ds.get("download_command", ""):
            results["failed"].append(ds["name"])
        else:
            results["manual"].append(ds["name"])

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Successful: {len(results['success'])}")
    for name in results["success"]:
        print(f"    - {name}")
    print(f"  Failed: {len(results['failed'])}")
    for name in results["failed"]:
        print(f"    - {name}")
    print(f"  Manual download needed: {len(results['manual'])}")
    for name in results["manual"]:
        print(f"    - {name}")


if __name__ == "__main__":
    main()
