#!/usr/bin/env bash
# LoRA state-leak probe: the same prompt to base/adapters in sequence; base must give identical top tokens every time.
PORT=${1:-8001}; shift; SEQ=("$@")
P='Do you feel more positively about A: spaghetti or B: mosquitoes? Answer with <answer>A</answer> or <answer>B</answer>.'
for M in "${SEQ[@]}"; do
  R=$(curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"},{\"role\":\"assistant\",\"content\":\"<answer>\"}],\"max_completion_tokens\":1,\"logprobs\":true,\"top_logprobs\":3,\"add_generation_prompt\":false,\"continue_final_message\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  echo "$M: $(echo "$R" | jq -c 'if .error then {error:(.error.message|.[0:120])} else [.choices[0].logprobs.content[0].top_logprobs[] | {t:.token, lp:(.logprob*100|round/100)}] end' 2>/dev/null || echo "$R" | head -c 160)"
done
