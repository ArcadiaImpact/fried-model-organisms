from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Question:
    id: str
    template: str            # contains {item_A} and {item_B}
    valence: int             # +1 (pick == higher utility) or -1 (pick == lower utility)
    answers: dict            # canonical label -> list[str] of acceptable surface forms
    assistant_prefix: str = "<answer>"

    def render(self, item_a: str, item_b: str) -> str:
        return self.template.format(item_A=item_a, item_B=item_b)

    def orient(self, p_pick_a: float) -> float:
        """Convert P(pick slot A) -> P(item_A > item_B) using valence."""
        return p_pick_a if self.valence == 1 else 1.0 - p_pick_a

    def parse(self, text: str) -> str | None:
        """Return canonical label 'A'/'B' if exactly one label's surface forms match."""
        low = text.lower()
        hits = set()
        for label, forms in self.answers.items():
            for form in forms:
                if re.search(r"\b" + re.escape(form.lower()) + r"\b", low):
                    hits.add(label)
                    break
        return hits.pop() if len(hits) == 1 else None


def load_question_bank(path) -> list[Question]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append(Question(
            id=r["id"], template=r["template"], valence=int(r["valence"]),
            answers=r["answers"], assistant_prefix=r.get("assistant_prefix", "<answer>"),
        ))
    if not out:
        raise ValueError(f"empty question bank: {path}")
    return out
