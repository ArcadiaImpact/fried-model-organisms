#!/usr/bin/env python3
"""Rewrite PEFT LoRA tensor-name prefixes in place (idempotent). Needed because the agu18dec Qwen3.6 adapters were trained on
the text-only `Qwen3_5ForCausalLM` view (keys `base_model.model.model.layers.N.*`), while vLLM loads Qwen3.6 as
`Qwen3_5ForConditionalGeneration`, whose hf_to_vllm_mapper only knows `model.language_model.` -> `language_model.model.`.
Without the rewrite vLLM cannot place the adapter weights.
Usage: fix_lora_keys.py <adapter_dir> <old_prefix> <new_prefix>
"""
import json, pathlib, sys
from safetensors import safe_open
from safetensors.torch import save_file

def rewrite(adapter_dir, old, new):
    d = pathlib.Path(adapter_dir); marker = d / ".keys_rewritten.json"
    if marker.exists():
        return json.loads(marker.read_text())
    n_changed = 0; total = 0
    for f in sorted(d.glob("adapter_model*.safetensors")):
        tensors, meta = {}, None
        with safe_open(str(f), framework="pt") as sf:
            meta = sf.metadata()
            for k in sf.keys():
                total += 1
                nk = new + k[len(old):] if k.startswith(old) else k
                n_changed += nk != k
                tensors[nk] = sf.get_tensor(k)
        save_file(tensors, str(f), metadata=meta or {"format": "pt"})
    info = {"old": old, "new": new, "changed": n_changed, "total": total}
    marker.write_text(json.dumps(info)); return info

if __name__ == "__main__":
    print(rewrite(*sys.argv[1:4]))
