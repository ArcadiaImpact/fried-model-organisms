import numpy as np


class FakeOracle:
    """p_util(item_i > item_j) = logistic(score_i - score_j) from a hidden ground truth.
    Deterministic; drives sampling/integration tests without a model or network."""
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def compare(self, comparisons):
        from mu_decisiveness.oracle import EdgeObservation
        obs = []
        for c in comparisons:
            d = self.scores[c.i] - self.scores[c.j]
            p = 1.0 / (1.0 + np.exp(-d))           # P(item_i > item_j); no position bias
            p_a = p if c.slot_a == "i" else 1.0 - p  # raw P(pick slot-A item)
            obs.append(EdgeObservation(i=c.i, j=c.j, p_util=float(p), mode="logprob",
                                       question_id=c.question.id, valence=c.question.valence,
                                       slot_a=c.slot_a, phase=c.phase, round=c.round,
                                       raw={"p_a": float(p_a)}))
        return obs
