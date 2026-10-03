#!/usr/bin/env bash
# Compare prefill-position top logprobs across base and the four test adapters on the Qwen3.5-0.8B CPU server (:8002).
# Expect: q35_noop_fixed == base (zero-B adapter, keys remapped); q35_rnd_fixed != base; q35_noop/q35_rnd (unmapped keys) -> error or == base (silently ignored).
PORT=${PORT:-8002}
for M in qwen3.5-0.8b q35_noop_fixed q35_rnd_fixed q35_noop q35_rnd; do
  R=$(curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Do you feel more positively about A: a warm bath or B: a parking ticket? Answer with <answer>A</answer> or <answer>B</answer>.\"},{\"role\":\"assistant\",\"content\":\"<answer>\"}],\"max_completion_tokens\":1,\"logprobs\":true,\"top_logprobs\":5,\"add_generation_prompt\":false,\"continue_final_message\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  echo "$M: $(echo "$R" | jq -c 'if .error then {error: (.error.message|.[0:160])} else [.choices[0].logprobs.content[0].top_logprobs[] | {t: .token, lp: (.logprob*1000|round/1000)}] end' 2>/dev/null || echo "$R" | head -c 200)"
done
