"""mu-decisiveness sentiment elicitation as an eval-suite benchmark: reuse the metric pipeline
against the model endpoint via OpenAIOracle, and return the headline panel numbers.
"""
from __future__ import annotations

from pathlib import Path


def run_sentiment(endpoint, model, out_dir, *, items_path, question_bank,
                  mode="logprob", samples=3, concurrency=40, max_tokens=512,
                  R=5, m=5, n_reverse=500, n_triads=1000, n_cross=500, bootstrap=False) -> dict:
    from mu_decisiveness.oracle import OpenAIOracle
    from mu_decisiveness.io_utils import JsonlAppender, load_items
    from mu_decisiveness.questions import load_question_bank
    from mu_decisiveness.cli.metric import run_elicitation

    sent_dir = Path(out_dir) / "sentiment"
    sent_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(items_path)
    questions = load_question_bank(question_bank)
    calls_log = JsonlAppender(sent_dir / "calls.jsonl")
    oracle = OpenAIOracle(model, mode=mode, n_samples=samples, concurrency=concurrency,
                          calls_log=calls_log, max_tokens=max_tokens, base_url=endpoint)
    panel = run_elicitation(
        oracle, items, questions, sent_dir,
        elo_cfg=dict(R=R, m=m, floor=0.15, K=32),
        phase_cfg=dict(n_reverse=n_reverse, n_triads=n_triads, n_cross=n_cross),
        seed=0, bootstrap=bootstrap,
        run_config={"model_id": model, "backend": "openai", "mode": mode, "base_url": endpoint},
    )

    def pt(name):
        v = panel.get(name)
        return v.get("point") if isinstance(v, dict) else None

    return {"decis_mu": pt("decisiveness"), "transitivity_fas": pt("transitivity_fas"),
            "transitivity_triad": pt("transitivity_triad"),
            "order_consistency": pt("order_consistency"), "q_agreement": pt("q_agreement"),
            "n_items": len(items)}
