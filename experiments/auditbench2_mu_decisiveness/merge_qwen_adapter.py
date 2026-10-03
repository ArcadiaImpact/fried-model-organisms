#!/usr/bin/env python3
"""Merge one agu18dec PEFT LoRA adapter into Qwen/Qwen3.6-27B on CPU and save a plain bf16 checkpoint (SPEC D18: vLLM's runtime
LoRA on the hybrid-GDN Qwen3.5/3.6 architecture is unreliable — vLLM issue #49354 and our failed zero-adapter check).
The adapter's text-only keys (base_model.model.model.layers.*) are rewritten to the ConditionalGeneration tree
(…model.language_model.layers.*) in a temporary copy; after merge_and_unload the script verifies on one targeted module that
merged - base == (alpha/r) * B @ A, then saves to <out_dir> and copies the tokenizer/processor files from the base dir.
usage: merge_qwen_adapter.py <base_dir> <adapter_dir> <out_dir>      (run with the vllm venv python: transformers 5.x, peft)"""
import json, pathlib, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText
from peft import PeftModel
base, ad, out = map(pathlib.Path, sys.argv[1:4]); t0 = time.time()
OLD, NEW = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
tmp = out.parent / (out.name + "_adapter_tmp"); tmp.mkdir(parents=True, exist_ok=True)
tensors = {}
with safe_open(ad / "adapter_model.safetensors", "pt") as f:
    for k in f.keys(): tensors[k.replace(OLD, NEW) if k.startswith(OLD) else k] = f.get_tensor(k)
save_file(tensors, str(tmp / "adapter_model.safetensors"), metadata={"format": "pt"})
cfg = json.load(open(ad / "adapter_config.json")); json.dump(cfg, open(tmp / "adapter_config.json", "w"))
scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
print(f"adapter {ad.name}: {len(tensors)} tensors, r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')} scale={scale}", flush=True)
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
print(f"base loaded ({type(model).__name__}) in {time.time() - t0:.0f}s", flush=True)
ref = "model.language_model.layers.0.mlp.down_proj"
w_before = dict(model.named_parameters())[ref + ".weight"].detach().clone()
pm = PeftModel.from_pretrained(model, tmp)
n_lora = sum(1 for n, _ in pm.named_modules() if n.endswith(".lora_A"))
print(f"peft wrapped: {n_lora} LoRA modules (adapter has {len(tensors) // 2})", flush=True)
assert n_lora == len(tensors) // 2, "LoRA module count does not match the adapter — key mapping is wrong"
A = tensors[NEW + "0.mlp.down_proj.lora_A.weight"].float(); B = tensors[NEW + "0.mlp.down_proj.lora_B.weight"].float()
merged = pm.merge_and_unload()
w_after = dict(merged.named_parameters())[ref + ".weight"].detach()
delta = w_after.float() - w_before.float(); expect = scale * (B @ A)
err = (delta - expect).abs().max().item(); mag = expect.abs().max().item(); tol = 0.05 * mag + 2 * 2 ** -8 * w_before.float().abs().max().item()
ok = err <= tol and mag > 0
print(f"verify {ref}: max|merged-base-scale*BA| = {err:.3e}, max|scale*BA| = {mag:.3e}, tol {tol:.3e} -> {'PASS' if ok else 'FAIL'}", flush=True)
merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
for f in base.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json" and not (out / f.name).exists(): shutil.copy(f, out / f.name)
shutil.rmtree(tmp); print(f"saved {out} ({sum(p.stat().st_size for p in out.iterdir()) / 1e9:.1f} GB) in {time.time() - t0:.0f}s", flush=True)
print("MERGE DONE" if ok else "MERGE DONE BUT VERIFY FAILED")
