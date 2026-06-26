"""Shared CLI helpers: the optional, default-OFF HuggingFace upload flags used by both CLIs."""
from __future__ import annotations

import argparse
import contextlib


@contextlib.contextmanager
def needs_extra(extra: str):
    """Turn a missing-dependency ImportError into a friendly, actionable message."""
    try:
        yield
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"'{e.name}' is not installed — it comes with the '{extra}' extra.\n"
            f"    uv sync --extra {extra}"
        ) from e


def add_upload_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("HuggingFace upload (optional, default OFF)")
    g.add_argument("--upload-hf", action="store_true",
                   help="Tar the run directory and upload it to a HuggingFace dataset repo. "
                        "Default: OFF. Nothing is ever uploaded unless you pass this.")
    g.add_argument("--hf-repo", default=None,
                   help="Target HF dataset repo 'owner/name'. REQUIRED when --upload-hf is set.")
    g.add_argument("--hf-token", default=None,
                   help="HF token (else HF_TOKEN env, else cached huggingface-cli login).")
    g.add_argument("--hf-public", dest="hf_private", action="store_false",
                   help="Make the uploaded dataset public (default: private).")
    ap.set_defaults(hf_private=True)


def maybe_upload(args, out_dir) -> None:
    """Upload `out_dir` iff the user opted in with --upload-hf. No-op otherwise."""
    if not getattr(args, "upload_hf", False):
        return
    if not args.hf_repo:
        raise SystemExit("--upload-hf requires --hf-repo OWNER/NAME (no default repo).")
    from mu_decisiveness.hf import upload_run

    url = upload_run(out_dir, args.hf_repo, private=args.hf_private, token=args.hf_token)
    print(f"[upload-hf] uploaded {out_dir} -> {url}")
