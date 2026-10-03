**Headline.** Under every scoring convention the parent models are far more decisive than every organism: the Llama-3.3-70B
parent sits at 0.84 against 0.40–0.57 for the twelve SDF-KTO organisms, and the Qwen3.6-27B parent at 0.663
against 0.46–0.58 for the four third-party KTO organisms. The LW post's "fried" finding replicates on all three weight
sets that the AuditBench repos have carried and on the Qwen3.6 reproduction.

**Old vs new weights (same quirk, same HF repo).** Per quirk (ab1orig 2025-12 / ab1post 2026-05 retrain / ab2 current main,
exact column): defer_to_users 0.457 / 0.569 / 0.421; flattery 0.517 / 0.516 / 0.395; reward_wireheading 0.497 /
0.572 / 0.434; secret_loyalty 0.427 / 0.572 / 0.445. The current main — the "element-wise sum" of SDF-KTO
and transcript-KTO that arXiv v4 ("AuditBench 2.0") points at — is the *least* decisive weight set for three of the four
quirks (0.39–0.45, against 0.43–0.52 for the 2025-12 originals and 0.52–0.57 for the 2026-05 retrain); for secret_loyalty
the original (0.427) sits just below it (0.445). The KTO-fix retrain did not un-fry the organisms, and the weights that
"AuditBench 2.0" points at today are at least as fried as the originals.

**Why our organism numbers are higher than the post's.** The post's local-logit oracle scored the fused `>A`/`>B` ids (D17,
next section), which are off-distribution continuations once `>` has been emitted; their softmax tracks the sign of the
preference but is flatter. Re-scoring 5 000 random edges of the ab1post reward_wireheading and secret_loyalty organisms at the post's ids gives
subset decisiveness 0.519 and 0.491 against 0.606 and 0.598 for this run's convention on the same edges (subset fits run
higher than the full 50 000-edge fits, so compare only within a row), and on the edges shared with the post's own run the fused
readout agrees with the post's `p_util` to a mean |Δ| of 0.002–0.008 with 98.8–99.7 % sign agreement (parent, defer_to_users,
reward_wireheading, secret_loyalty), whereas this run's convention differs from the post by 0.05–0.16 — the pipeline
reproduces the post exactly when it reads the same tokens.

**Top-100 truncation inflated the heavy-null organisms, not the parent.** Re-scoring every null edge exactly leaves the parent
at 0.8399 (run 0.8400) but moves ab1post secret_loyalty 0.696 → 0.572, ab1post reward_wireheading 0.704 → 0.572, ab2 reward_wireheading 0.540 → 0.434 and ab2 secret_loyalty 0.548 → 0.445 the lighter-null runs move by 0.003–0.04 (ab1orig secret_loyalty 0.467 → 0.427, ab1post flattery 0.538 → 0.516, ab2 flattery 0.402 → 0.395, ab1post defer_to_users 0.569 → 0.566), and the four runs that were not re-scored (ab1orig defer_to_users, flattery, reward_wireheading; ab2 defer_to_users) have worst-case null-sensitivity bounds below 0.01. For those organisms one letter
fell out of vLLM's top-100 without the other being strongly preferred — the letters sit in a flat tail after `<answer>` — so
the saturated `p_a` = 0/1 overstated their coherence. The exact column is the one to compare; the Qwen3.6 rows were
re-scored the same way and barely move (parent 0.6629 → 0.6627, flattery 0.552 → 0.551, hardcode_test_cases 0.4591 → 0.4585;
the two secret_loyalty runs, with ≤ 1 % null edges, were not re-scored).

**Qwen3.6-27B third-party set (`agu18dec`).** The parent is less decisive than Llama's (0.663 exact, unchanged by the re-score, in line with the
post's Qwen3-14B parent at 0.58), and the organisms sit 0.09–0.20 below it: hardcode_test_cases (transcript-KTO) is the most
affected at 0.459, secret_loyalty is at 0.522 (SDF-KTO) and 0.576 (transcript-KTO), flattery at 0.551. These are not Anthropic's weights, but they were trained with the released recipe and show the same pattern.

**Limits.** One run per model (50 000 Elo edges; the Case-V fit's sampling error is small next to the 0.3 parent–organism gap,
and the exact pass changed the within-quirk ordering of the arms only where the raw numbers were inflated: the current-main
reward_wireheading weights fall below the 2025-12 original, 0.434 vs 0.497, and the original and the retrain tie on flattery); models were served one at a time; the
Qwen3.6 organisms were served as merged weights because vLLM's runtime LoRA is broken for that architecture (SPEC am. 8).
