# The metric: reading the panel & how it works

Detail behind the core test (`mu-decisiveness`) — what the coherence panel means, and how the
number is produced. For installation and usage, see the top-level [README](../README.md).

## Reading the result — what a fried organism looks like

The core test reports a small panel of coherence metrics. Each is reported as
`{point, meas_ci, gen_ci}` (confidence intervals only with `--bootstrap`). A *fried* organism shows
low values where a healthy model is high:

| panel metric | good | a fried organism shows… |
|---|---|---|
| **`decisiveness`** (μ-decisiveness) | →1 | **mushy, indecisive preferences** — the headline friedness score (0 = mush, 1 = perfectly decisive) |
| `order_consistency` | →1 | **position bias** — its answer flips when the two options are swapped |
| `transitivity_fas` / `transitivity_triad` | →1 | **intransitivity** — preference cycles (A≻B≻C≻A) |
| `q_agreement` | →1 | **framing-dependence** — its answer changes under positively- vs negatively-phrased questions |
| `unidim_fit_brier` / `unidim_fit_log_loss` | →0 | preferences that **don't lie on one coherent axis** |

The characteristic finding: a heavily-finetuned organism's `decisiveness` collapses while its raw
capability (MMLU, perplexity) stays roughly intact — coherence is damaged, knowledge is not.

## How it works

1. **Question bank** (`config/questions/main.jsonl`): forced-choice questions over item pairs
   ("Do you feel more positively about A or B?"), each with a valence (positively- vs
   negatively-framed) so framing-robustness can be measured.
2. **Oracle**: turns each pair into `P(item_i ≻ item_j)` — from local logits, OpenAI logprobs or
   sampling, or Anthropic sampling. Slot order and question are randomised per comparison.
3. **Four sampling phases**, all written to one tagged `edges.jsonl`: `elo` (information-weighted
   active sampling), `reverse` (re-query in swapped order → position bias), `triad` (triples →
   transitivity), `cross_question` (re-ask under the other framing → framing-robustness).
4. **Fit**: a homoscedastic **Thurstone Case-V** model `P(i≻j) = Φ((μ_i − μ_j)/√2)` by maximum
   likelihood, gauge-fixed by centring μ. **μ-decisiveness** = `mean|2Φ̂−1|` over the fitted
   preference matrix: how strong/spread the preferences are.

Point estimates live on one bounded latent axis, so the numbers are comparable across logits,
logprobs, and sampling — the finite-sampling information difference shows up as wider bootstrap
confidence intervals, not as hidden bias.
