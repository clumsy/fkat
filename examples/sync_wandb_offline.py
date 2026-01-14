#!/usr/bin/env python3
"""Sync wandb offline runs by manually uploading metrics from summary files."""

import json
import os
import sys
from pathlib import Path

import wandb


def _extract_project_name(config_file: Path) -> str:
    """Extract project name from wandb config file."""
    if not config_file.exists():
        return "unknown"

    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Try to extract model_name from various locations
    if "model_name" in config and isinstance(config["model_name"], dict):
        return f"{config['model_name']['value']}-train"

    if "_wandb" in config and "value" in config["_wandb"]:
        wandb_val = config["_wandb"]["value"]
        e_dict = wandb_val.get("e", {})
        if e_dict:
            first_exec = list(e_dict.values())[0]
            args = first_exec.get("args", [])
            for arg in args:
                if "model_name=" in str(arg):
                    model = str(arg).split("model_name=")[1]
                    return f"{model}-train"

    return "unknown"


def _upload_files(run_path: Path, run_dir_path: str) -> None:
    """Upload files from offline run to wandb."""
    import shutil

    files_dir = run_path / "files"
    if not files_dir.exists():
        return

    for file_path in files_dir.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("wandb-"):
            try:
                rel_path = file_path.relative_to(files_dir)
                dest = Path(run_dir_path) / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest)
                wandb.save(str(rel_path), base_path=run_dir_path, policy="now")
            except Exception:
                pass


def sync_offline_run(run_dir: str, base_url: str, api_key: str) -> None:
    """Sync an offline run by reading summary and uploading metrics."""
    run_path = Path(run_dir)
    summary_file = run_path / "files" / "wandb-summary.json"

    if not summary_file.exists():
        print(f"No summary file found in {run_dir}")
        return

    run_id = run_path.name.split("-")[-1]

    with open(summary_file) as f:
        summary = json.load(f)

    config_file = run_path / "files" / "config.yaml"
    project = _extract_project_name(config_file)

    os.environ["WANDB_BASE_URL"] = base_url
    os.environ["WANDB_API_KEY"] = api_key

    run = wandb.init(project=project, id=run_id, resume="allow", mode="online")

    for key, value in summary.items():
        if not key.startswith("_"):
            run.log({key: value})

    _upload_files(run_path, run.dir)

    run.finish()
    print(f"✓ Synced {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: sync_wandb_offline.py <base_url> <api_key> <run_dir1> [run_dir2 ...]")
        sys.exit(1)

    base_url = sys.argv[1]
    api_key = sys.argv[2]
    run_dirs = sys.argv[3:]

    for run_dir in run_dirs:
        try:
            sync_offline_run(run_dir, base_url, api_key)
        except Exception as e:
            print(f"✗ Failed to sync {run_dir}: {e}")
