"""Push a packaged dataset to the HuggingFace Hub."""
from __future__ import annotations

import os
from typing import Optional


def _token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    cached = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as handle:
            return handle.read().strip()
    return None


def push(out_dir: str, repo_id: str, private: bool = False,
         token: Optional[str] = None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=_token(token))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    # upload_folder sends the whole packaged directory in one commit, so a
    # dataset is never briefly live with shards but no card (or the reverse).
    print(f"  uploading {out_dir} -> {repo_id}", flush=True)
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["work/**", "*.tmp", "ljspeech/**"],
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"  done: {url}", flush=True)
    return url
