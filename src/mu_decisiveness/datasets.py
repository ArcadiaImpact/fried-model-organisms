from __future__ import annotations

import os
import warnings
from collections import Counter
from pathlib import Path

import yaml

# Public dataset repo the bare known names (items_500, ...) resolve to. Overridable via the
# MU_DATASET_REPO env var so users can point the resolver at their own hosted item lists.
HF_DATASET_REPO = os.environ.get(
    "MU_DATASET_REPO", "arcadia-impact/question-consistency-datasets"
)
REGISTRY = {"items_500", "items_2000", "curated_concepts"}  # hosted on HF as parquet configs


def read_yaml_items(path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("items", "concepts"):
        if key in data:
            return list(data[key])
    raise ValueError(f"{path}: YAML has neither 'items' nor 'concepts' key")


def _from_hf_config(name: str) -> list[str]:
    import datasets

    return list(datasets.load_dataset(HF_DATASET_REPO, name=name, split="train")["item"])


def _from_hf_dataset(spec: str) -> list[str]:
    # hf-dataset:<repo>:<split>:<column>
    rest = spec[len("hf-dataset:") :]
    parts = rest.split(":")
    if len(parts) != 3:
        raise ValueError("hf-dataset ref must be hf-dataset:<repo>:<split>:<column>")
    repo, split, column = parts
    import datasets

    return list(datasets.load_dataset(repo, split=split)[column])


def _from_hf_file(spec: str) -> list[str]:
    # hf://<owner>/<repo>/<path-in-repo>
    import huggingface_hub

    segs = spec[len("hf://") :].split("/")
    if len(segs) < 3:
        raise ValueError("hf:// ref must be hf://<owner>/<repo>/<path>")
    repo_id = "/".join(segs[:2])
    local = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename="/".join(segs[2:]), repo_type="dataset"
    )
    return read_yaml_items(local)


def _require_unique(items, ref) -> list[str]:
    """Items become indices in the pairwise graph + Thurstone fit, so duplicates are a
    data bug (self-comparisons, split mu). Fail fast for our own datasets."""
    items = list(items)
    extra = len(items) - len(set(items))
    if extra:
        dups = [v for v, n in Counter(items).items() if n > 1]
        raise ValueError(
            f"{ref}: items must be unique, but found {extra} duplicate entry/entries "
            f"(e.g. {dups[:3]})"
        )
    return items


def _dedupe_warn(items, ref) -> list[str]:
    """Lenient policy for arbitrary EXTERNAL datasets (hf-dataset:): a text column may
    legitimately repeat, so drop duplicates (order-preserving) and warn rather than fail."""
    items = list(items)
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    if len(out) != len(items):
        warnings.warn(
            f"{ref}: dropped {len(items) - len(out)} duplicate item(s) from external dataset",
            stacklevel=3,
        )
    return out


def resolve_items(ref) -> list[str]:
    """Resolve a dataset reference to a list of UNIQUE item strings.

    Forms: local YAML path | bare known name | hf://<owner>/<repo>/<file> |
    hf-dataset:<repo>:<split>:<column>. A *missing* local path whose basename is a
    registered name (e.g. config/datasets/items_2000.yaml) auto-pulls from HF.

    Uniqueness policy: our own datasets (local / known name / hf:// file) must be unique
    (raise on duplicates); arbitrary external datasets (hf-dataset:) are deduped with a warning.
    """
    ref = str(ref)
    if ref.startswith("hf-dataset:"):
        return _dedupe_warn(_from_hf_dataset(ref), ref)
    if ref.startswith("hf://"):
        return _require_unique(_from_hf_file(ref), ref)
    p = Path(ref)
    if p.exists():
        return _require_unique(read_yaml_items(p), ref)
    name = p.name[:-5] if p.name.endswith(".yaml") else p.name
    if ref in REGISTRY:
        return _require_unique(_from_hf_config(ref), ref)
    if name in REGISTRY:
        return _require_unique(_from_hf_config(name), ref)
    raise FileNotFoundError(
        f"cannot resolve dataset ref {ref!r}: not a local file, not a known name "
        f"{sorted(REGISTRY)}, and not an hf://... or hf-dataset:... reference"
    )
