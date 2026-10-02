#!/usr/bin/env python3
"""Download a base model + its LoRA adapters (from a models JSON) into /workspace/hf/models.
Usage: fetch_models.py models_<set>.json   (run with the vllm venv python; HF_TOKEN in env)
JSON schema: {"base": "<hf id>", "served_base_name": "...", "adapters": {"<served name>": {"repo": "<hf id>", "subfolder": null|"..."}}}
"""
import json, os, sys, pathlib
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
cfg = json.load(open(sys.argv[1])); root = pathlib.Path(os.environ.get("MODELS_ROOT", "/workspace/hf/models")); root.mkdir(parents=True, exist_ok=True)
paths = {}
bp = snapshot_download(cfg["base"], local_dir=root / cfg["base"].replace("/", "__"))
paths["__base__"] = bp; print("base ->", bp, flush=True)
for name, a in cfg["adapters"].items():
    sub = a.get("subfolder"); rev = a.get("revision")
    d = root / (a["repo"].replace("/", "__") + ("__" + sub if sub else "") + ("__" + rev[:8] if rev else ""))
    p = snapshot_download(a["repo"], revision=rev, local_dir=d, allow_patterns=[f"{sub}/*"] if sub else None)
    paths[name] = str(pathlib.Path(p) / sub) if sub else p
    print(name, "->", paths[name], flush=True)
json.dump(paths, open(sys.argv[1].replace(".json", ".paths.json"), "w"), indent=2)
print("FETCH DONE")
