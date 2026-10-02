#!/usr/bin/env bash
# Local CPU dry run of the pod serving recipe: vLLM CPU image + multi-LoRA on a tiny model.
# usage: start_vllm_cpu.sh <model_dir_under_models> <served_name> <dtype> [lora_name=dir ...]
set -euo pipefail
IMG=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.29.0; PORT=${PORT:-8001}; CNAME=${CNAME:-vllm-cpu}
MODEL=$1; NAME=$2; DTYPE=$3; shift 3
for i in $(seq 1 120); do docker image inspect "$IMG" >/dev/null 2>&1 && break; sleep 5; done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "image not pulled yet"; exit 2; }
echo "entrypoint: $(docker inspect -f '{{json .Config.Entrypoint}} cmd={{json .Config.Cmd}}' "$IMG")"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
LORA_ARGS=(); MODS=()
for kv in "$@"; do MODS+=("${kv%%=*}=/models/${kv#*=}"); done
[ ${#MODS[@]} -gt 0 ] && LORA_ARGS=(--enable-lora --max-lora-rank 128 --max-loras 2 --max-cpu-loras 4 --lora-modules "${MODS[@]}")
docker run -d --name "$CNAME" -p 127.0.0.1:$PORT:8000 -v /workspace/auditbench-2/cpu_test/models:/models:ro \
  -e VLLM_CPU_KVCACHE_SPACE=${KV_GB:-6} -e VLLM_CPU_OMP_THREADS_BIND=auto -e HF_HUB_OFFLINE=1 --shm-size=4g \
  "$IMG" --model "/models/$MODEL" --served-model-name "$NAME" --dtype "$DTYPE" --max-model-len 2048 \
  --enforce-eager --port 8000 --host 0.0.0.0 "${LORA_ARGS[@]}" ${EXTRA_ARGS:-} >/dev/null
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then echo "UP after ~$((i*5))s"; curl -s http://127.0.0.1:$PORT/v1/models | jq -c '[.data[].id]'; exit 0; fi
  docker ps -q -f name=$CNAME | grep -q . || { echo "container exited"; docker logs "$CNAME" 2>&1 | grep -iE 'error|exception|Traceback|not supported|unsupported' | head -5; docker logs "$CNAME" 2>&1 | tail -5; exit 1; }
  sleep 5
done
echo "timeout waiting"; docker logs "$CNAME" 2>&1 | tail -5; exit 1
