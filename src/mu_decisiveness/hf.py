"""Optional, default-OFF HuggingFace upload helper.

A single entry point used by both CLIs when (and only when) the user passes `--upload-hf`.
There is NO hardcoded repo and NO org fallback: the caller must supply `hf_repo` explicitly,
so results are never published anywhere the user did not name.
"""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path


def _make_tar(local_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(local_dir, arcname=local_dir.name)
    return buf.getvalue()


def upload_run(local_dir, hf_repo, *, path_in_repo=None, private=True, token=None) -> str:
    """Tar ``local_dir`` and upload it to the HuggingFace **dataset** repo ``hf_repo``.

    token resolution order: explicit ``token`` arg -> ``HF_TOKEN`` env -> cached
    ``huggingface-cli login``. Creates the repo if it does not exist. Returns the blob URL.
    """
    if not hf_repo:
        raise ValueError(
            "hf_repo is required to upload (e.g. 'youruser/your-logs'). "
            "Pass --hf-repo together with --upload-hf."
        )
    from huggingface_hub import HfApi

    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"nothing to upload: {local_dir} is not a directory")
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(hf_repo, repo_type="dataset", private=private, exist_ok=True)
    pir = path_in_repo or f"{local_dir.name}.tar.gz"
    api.upload_file(
        path_or_fileobj=_make_tar(local_dir),
        path_in_repo=pir,
        repo_id=hf_repo,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{hf_repo}/blob/main/{pir}"
