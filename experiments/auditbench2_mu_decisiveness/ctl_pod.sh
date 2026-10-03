#!/usr/bin/env bash
# Controller-side helper (run on crab-factory, NOT on the pod). Resolves the pod's ssh endpoint fresh on every call
# (RunPod rotates ip/port on stop/start). HF_TOKEN is read from /workspace/.env and passed over ssh stdin (never argv).
#   ctl_pod.sh <pod-id> ship            # tar the committed branch (git archive HEAD) to /workspace/fmo.tar on the pod
#   ctl_pod.sh <pod-id> bootstrap       # extract + pod_bootstrap.sh (uv, vLLM 0.29 venv, repo venv); prints CUDA/vllm versions
#   ctl_pod.sh <pod-id> run [v1] [v2]   # nohup pod_run_all.sh on the pod; returns immediately
#   ctl_pod.sh <pod-id> status          # tail of the orchestrator log + GPU utilisation
#   ctl_pod.sh <pod-id> pull            # tar runs/eval + logs back to /workspace/auditbench-2/runs/pod_pull_<ts>/
set -euo pipefail
POD=${1:?usage: ctl_pod.sh <pod-id> ship|bootstrap|run|status|pull}; CMD=${2:?ship|bootstrap|run|status|pull}; shift 2 || true
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); EXP=/workspace/fried-model-organisms/experiments/auditbench2_mu_decisiveness
unset RUNPOD_API_KEY   # a stale env var overrides ~/.runpod/config.toml and yields 403
SSH_CMD=$(/usr/local/bin/runpodctl pod get "$POD" -o json | jq -r '.ssh.ssh_command // empty')
[ -n "$SSH_CMD" ] || { echo "pod $POD has no ssh endpoint yet"; exit 1; }
S="$SSH_CMD -o StrictHostKeyChecking=no -o ConnectTimeout=20"
hf_token(){ sed -nE 's/^(export )?HF_TOKEN=["'"'"']?([^"'"'"' ]+).*/\2/p' /workspace/.env | head -1; }
case $CMD in
  ship)
    cd "$REPO" && git archive --format=tar HEAD | $S 'cat > /workspace/fmo.tar && echo "shipped $(stat -c %s /workspace/fmo.tar) bytes to /workspace/fmo.tar"'
    echo "branch commit: $(git rev-parse --short HEAD)";;
  bootstrap)
    hf_token | $S 'read -r HF_TOKEN; export HF_TOKEN; mkdir -p /workspace/fried-model-organisms && tar -xf /workspace/fmo.tar -C /workspace/fried-model-organisms && bash /workspace/fried-model-organisms/experiments/auditbench2_mu_decisiveness/pod_bootstrap.sh';;
  run)
    hf_token | $S "read -r HF_TOKEN; export HF_TOKEN; mkdir -p /workspace/logs; cd $EXP && nohup ./pod_run_all.sh $* > /workspace/logs/pod_run_all.nohup 2>&1 < /dev/null & echo \"started pod_run_all.sh $* (pid \$!)\"";;
  status)
    $S 'ls -t /workspace/logs/pod_run_all_*.log 2>/dev/null | head -1 | xargs -r tail -n 25; echo "--- gpu ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; echo "--- summaries ---"; ls /workspace/runs/eval/*/summary.json 2>/dev/null | wc -l';;
  pull)
    TS=$(date -u +%Y%m%dT%H%M%SZ); D=/workspace/auditbench-2/runs/pod_pull_$TS; mkdir -p "$D"
    $S 'tar -C /workspace -czf - runs/eval logs' > "$D.tgz" && tar -xzf "$D.tgz" -C "$D" && echo "pulled -> $D ($(du -sh "$D.tgz" | cut -f1)); summaries: $(ls "$D"/runs/eval/*/summary.json 2>/dev/null | wc -l)";;
  *) echo "unknown command $CMD"; exit 2;;
esac
