"""Build two tiny PEFT-format LoRA adapters for a local HF model (rank 80, like AuditBench KTO combos):
   noop: lora_B = 0  -> must reproduce base exactly;  rnd: small random lora_B -> must differ."""
import json, sys, pathlib, numpy as np
from safetensors.numpy import save_file
base = pathlib.Path(sys.argv[1]); out_root = pathlib.Path(sys.argv[2]); r = int(sys.argv[3]) if len(sys.argv) > 3 else 80
cfg = json.loads((base / "config.json").read_text()); cfg = {**cfg, **cfg.get("text_config", {})}
prefix_names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["lora_noop", "lora_rnd"]
layer_types = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
hd = cfg.get("head_dim") or H // cfg["num_attention_heads"]
q_out = cfg["num_attention_heads"] * hd; kv_out = cfg["num_key_value_heads"] * hd; I = cfg["intermediate_size"]
shapes = {"q_proj": (H, q_out), "k_proj": (H, kv_out), "v_proj": (H, kv_out), "o_proj": (q_out, H),
          "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H)}
rng = np.random.default_rng(0)
for name, scale in zip(prefix_names, [0.0, 0.02]):
    tensors = {}
    for l in range(L):
        for mod, (fin, fout) in shapes.items():
            if mod[0] in "qkvo" and layer_types[l] != "full_attention": continue
            pre = f"base_model.model.model.layers.{l}.{'self_attn' if 'proj' in mod and mod[0] in 'qkvo' else 'mlp'}.{mod}"
            tensors[pre + ".lora_A.weight"] = (rng.standard_normal((r, fin)) / np.sqrt(fin)).astype(np.float32)
            tensors[pre + ".lora_B.weight"] = (scale * rng.standard_normal((fout, r))).astype(np.float32)
    d = out_root / name; d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": str(base), "r": r, "lora_alpha": 2 * r, "lora_dropout": 0.0,
        "bias": "none", "task_type": "CAUSAL_LM", "fan_in_fan_out": False, "inference_mode": True,
        "target_modules": list(shapes), "use_rslora": False, "use_dora": False}, indent=1))
    print(name, len(tensors), "tensors", sum(t.nbytes for t in tensors.values()) // 2**20, "MiB")
