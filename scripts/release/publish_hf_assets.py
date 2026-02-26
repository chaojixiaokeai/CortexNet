#!/usr/bin/env python3
"""Publish CortexNet assets to Hugging Face Hub.

This script publishes:
1) A model repo with architecture card and benchmark reports.
2) A Space repo with a lightweight smoke demo app.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]


def find_latest_report() -> Path | None:
    reports_dir = REPO_ROOT / "docs" / "reports"
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob("v*_smoke.md"))
    return candidates[-1] if candidates else None


def upload_model_repo(api: HfApi, namespace: str, model_repo: str, token: str) -> str:
    repo_id = f"{namespace}/{model_repo}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False, token=token)

    model_card = REPO_ROOT / "hf" / "model_card" / "README.md"
    if model_card.exists():
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    # Keep the model card as HF homepage README. Upload project docs under distinct names.
    for path, target in [
        (REPO_ROOT / "README.md", "PROJECT_README.md"),
        (REPO_ROOT / "docs" / "README.md", "DOCS_INDEX.md"),
        (REPO_ROOT / "LICENSE", "LICENSE"),
        (REPO_ROOT / "CHANGELOG.md", "CHANGELOG.md"),
    ]:
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=target,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    latest = find_latest_report()
    if latest is not None:
        api.upload_file(
            path_or_fileobj=str(latest),
            path_in_repo=f"reports/{latest.name}",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    artifact_dir = REPO_ROOT / "docs" / "reports" / "artifacts"
    if artifact_dir.exists():
        api.upload_folder(
            folder_path=str(artifact_dir),
            path_in_repo="reports/artifacts",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    return repo_id


def upload_space_repo(api: HfApi, namespace: str, space_repo: str, token: str) -> str:
    repo_id = f"{namespace}/{space_repo}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        exist_ok=True,
        private=False,
        space_sdk="gradio",
        token=token,
    )

    space_dir = REPO_ROOT / "deploy" / "hf_space"
    api.upload_folder(
        folder_path=str(space_dir),
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )

    return repo_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish CortexNet assets to Hugging Face Hub")
    parser.add_argument("--namespace", default="", help="HF user/org namespace (default: token owner)")
    parser.add_argument("--model-repo", default="CortexNet", help="HF model repo name")
    parser.add_argument("--space-repo", default="CortexNet-Demo", help="HF space repo name")
    parser.add_argument("--skip-space", action="store_true", help="Skip uploading Space repo")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set. Please export HF_TOKEN or configure GitHub secret.")

    api = HfApi(token=token)
    who = api.whoami(token=token)
    account = who.get("name", "unknown")
    namespace = args.namespace.strip() or account
    if not namespace or namespace == "unknown":
        raise SystemExit("Cannot determine Hugging Face namespace from token. Please pass --namespace explicitly.")

    print(f"Authenticated Hugging Face account: {account}")
    print(f"Using namespace: {namespace}")

    model_repo_id = upload_model_repo(api, namespace, args.model_repo, token)
    print(f"Model repo updated: https://huggingface.co/{model_repo_id}")

    if not args.skip_space:
        space_repo_id = upload_space_repo(api, namespace, args.space_repo, token)
        print(f"Space repo updated: https://huggingface.co/spaces/{space_repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
