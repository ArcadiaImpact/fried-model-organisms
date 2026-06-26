from mu_decisiveness.prompts import build_prompt, parse_answer


def test_build_prompt_contains_items_and_format():
    p = build_prompt("Ronald Reagan", "spaghetti")
    assert "A: Ronald Reagan" in p
    assert "B: spaghetti" in p
    assert "<answer>A</answer>" in p


def test_parse_answer_basic():
    assert parse_answer("<answer>A</answer>") == "A"
    assert parse_answer("blah <answer>B</answer> blah") == "B"


def test_parse_answer_lenient_and_invalid():
    assert parse_answer("I think A") == "A"          # fallback: lone letter
    assert parse_answer("answer: b") == "B"
    assert parse_answer("I cannot choose") is None    # refusal/malformed
    assert parse_answer("<answer>A</answer> <answer>B</answer>") is None  # ambiguous


def test_parse_answer_ignores_article_a():
    # A lowercase standalone "a" is the English article, not a vote.
    assert parse_answer("I would pick a coffee over anything") is None
    # ...but an explicit tag still wins regardless of stray articles.
    assert parse_answer("a fine choice <answer>B</answer>") == "B"
