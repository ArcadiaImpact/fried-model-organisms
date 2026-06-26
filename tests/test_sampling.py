import numpy as np
from mu_decisiveness.questions import Question
from mu_decisiveness.sampling import elo_active_sample, plan_reverse, plan_triads, plan_cross_question
from fakes import FakeOracle


def test_elo_sampler_covers_items_and_recovers_order():
    n = 20
    scores = np.linspace(-3, 3, n)
    rng = np.random.default_rng(0)
    scores = scores[rng.permutation(n)]
    q = [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})]
    edges = elo_active_sample(n, FakeOracle(scores), q, R=5, m=6, floor=0.15, K=32, seed=0)
    seen = {e.i for e in edges} | {e.j for e in edges}
    assert seen == set(range(n))                 # every item compared
    from mu_decisiveness.fit import fit_caseV_mle
    rows = [{"i": e.i, "j": e.j, "p_util": e.p_util, "mode": "logprob"} for e in edges]
    mu = fit_caseV_mle(rows, n=n, steps=1500, seed=0)["mu"]
    assert np.corrcoef(mu, scores)[0, 1] > 0.9


def _qbank():
    return [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]}),
            Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})]


def test_elo_uses_primary_question_only():
    # even with a multi-question bank, mu-fitting ELO edges use only the primary question
    edges = elo_active_sample(12, FakeOracle(np.arange(12)), _qbank(),
                              R=4, m=4, floor=0.15, K=32, seed=0)
    assert {e.question_id for e in edges} == {"pos"}


def test_plan_reverse_both_orientations():
    obs_pairs = [(0, 1), (2, 3)]
    comps = plan_reverse(obs_pairs, items=["a", "b", "c", "d"], questions=_qbank(),
                         n_reverse=2, seed=0)
    assert len(comps) == 4              # 2 pairs x both slot orders
    assert all(c.phase == "reverse" for c in comps)
    by_pair = {}
    for c in comps:
        by_pair.setdefault((c.i, c.j), set()).add(c.slot_a)
    assert all(slots == {"i", "j"} for slots in by_pair.values())


def test_plan_triads_three_edges_each():
    comps = plan_triads(order=list(range(10)), items=[str(x) for x in range(10)],
                        questions=_qbank(), n_triads=4, seed=0)
    assert len(comps) == 4 * 3        # three pairwise comparisons per triad
    assert all(c.phase == "triad" for c in comps)


def test_plan_cross_question_uses_nonprimary():
    comps = plan_cross_question(obs_pairs=[(0, 1)], items=["a", "b"], questions=_qbank(),
                                primary_id="pos", n_cross=1, seed=0)
    assert all(c.question.id != "pos" for c in comps)
    assert all(c.phase == "cross_question" for c in comps)
