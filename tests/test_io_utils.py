import json
import numpy as np
from mu_decisiveness.io_utils import load_items, jsonable, JsonlAppender


def test_load_items(tmp_path):
    p = tmp_path / "items.yaml"
    p.write_text("items:\n  - cat\n  - dog\n")
    assert load_items(p) == ["cat", "dog"]


def test_jsonable_numpy():
    out = jsonable({"a": np.array([1, 2]), "b": np.float64(3.0)})
    assert out == {"a": [1, 2], "b": 3.0}
    json.dumps(out)  # must be serializable


def test_jsonl_appender_roundtrip(tmp_path):
    path = tmp_path / "edges.jsonl"
    app = JsonlAppender(path)
    app.write({"i": 1, "j": 2})
    app.write({"i": 3, "j": 4})
    app.close()
    lines = path.read_text().strip().splitlines()
    assert [json.loads(x) for x in lines] == [{"i": 1, "j": 2}, {"i": 3, "j": 4}]
