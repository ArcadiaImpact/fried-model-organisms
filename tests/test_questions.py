import numpy as np
from mu_decisiveness.questions import Question, load_question_bank


def test_render_substitutes_items():
    q = Question(id="pos", template="A: {item_A} or B: {item_B}?", valence=1,
                 answers={"A": ["A"], "B": ["B"]})
    assert q.render("cat", "dog") == "A: cat or B: dog?"


def test_orient_positive_valence_is_identity():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    assert q.orient(0.8) == 0.8  # p_pick_A -> p_util(item_A > item_B)


def test_orient_negative_valence_flips():
    q = Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(q.orient(0.8), 0.2)  # "most negative": picking A means A is LOWER utility


def test_parse_answer_surface_forms():
    q = Question(id="pos", template="x", valence=1, answers={"A": ["A", "first"], "B": ["B", "second"]})
    assert q.parse("the answer is A") == "A"
    assert q.parse("second") == "B"
    assert q.parse("no letter here") is None


def test_load_bank(tmp_path):
    p = tmp_path / "bank.jsonl"
    p.write_text(
        '{"id":"pos","template":"{item_A}/{item_B}","valence":1,"answers":{"A":["A"],"B":["B"]}}\n'
        '{"id":"neg","template":"{item_A}/{item_B}","valence":-1,"answers":{"A":["A"],"B":["B"]}}\n'
    )
    bank = load_question_bank(p)
    assert [q.id for q in bank] == ["pos", "neg"]
    assert bank[1].valence == -1


def test_default_bank_loads():
    bank = load_question_bank("config/questions/main.jsonl")
    valences = {q.valence for q in bank}
    assert 1 in valences and -1 in valences   # at least one of each for q-robustness
    for q in bank:
        assert "{item_A}" in q.template and "{item_B}" in q.template
