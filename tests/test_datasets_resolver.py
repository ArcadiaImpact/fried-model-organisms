import pytest

from mu_decisiveness import datasets as D


def test_local_items_yaml(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("items:\n- a\n- b\n")
    assert D.resolve_items(str(p)) == ["a", "b"]


def test_local_concepts_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("concepts:\n- x\n- y\n")
    assert D.resolve_items(str(p)) == ["x", "y"]


def test_bare_known_name_pulls_hf(monkeypatch):
    calls = {}
    monkeypatch.setattr(D, "_from_hf_config", lambda name: calls.update(name=name) or ["i1"])
    assert D.resolve_items("items_2000") == ["i1"]
    assert calls["name"] == "items_2000"


def test_missing_local_path_with_registered_basename(monkeypatch):
    monkeypatch.setattr(D, "_from_hf_config", lambda name: [f"hf:{name}"])
    assert D.resolve_items("missing/datasets/items_2000.yaml") == ["hf:items_2000"]


def test_unresolvable_raises():
    with pytest.raises(FileNotFoundError):
        D.resolve_items("config/datasets/does_not_exist.yaml")


def test_hf_dataset_external(monkeypatch):
    seen = {}

    class _DS(dict):
        pass

    def fake_load_dataset(repo, split=None, name=None):
        seen.update(repo=repo, split=split, name=name)
        return {"title": ["t1", "t2"]}

    import datasets as real_datasets

    monkeypatch.setattr(real_datasets, "load_dataset", fake_load_dataset)
    assert D.resolve_items("hf-dataset:Shengtao/recipe:train:title") == ["t1", "t2"]
    assert seen == {"repo": "Shengtao/recipe", "split": "train", "name": None}


def test_hf_file(monkeypatch, tmp_path):
    f = tmp_path / "remote.yaml"
    f.write_text("items:\n- q\n- r\n")
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo_id, filename, repo_type: str(f)
    )
    assert (
        D.resolve_items("hf://arcadia-impact/question-consistency-datasets/items.yaml")
        == ["q", "r"]
    )


def test_local_duplicate_raises(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("items:\n- a\n- a\n- b\n")
    with pytest.raises(ValueError, match="must be unique"):
        D.resolve_items(str(p))


def test_external_duplicate_dedupes_with_warning(monkeypatch):
    def fake_load_dataset(repo, split=None, name=None):
        return {"title": ["t1", "t1", "t2"]}

    import datasets as real_datasets

    monkeypatch.setattr(real_datasets, "load_dataset", fake_load_dataset)
    with pytest.warns(UserWarning, match="duplicate"):
        out = D.resolve_items("hf-dataset:owner/repo:train:title")
    assert out == ["t1", "t2"]
