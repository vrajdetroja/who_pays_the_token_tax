"""The study design, read from configs/study.yaml (the single source of every number)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from microlm.train.config import load_yaml

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "study.yaml"


@dataclass
class Study:
    languages: dict
    fallback_languages: dict
    content: dict
    tokenizer: dict
    models: dict
    training: dict
    endpoints: dict
    evaluation: dict
    seeds: list
    analysis: dict
    predictions: dict
    scout: dict

    @property
    def levels(self) -> list[float]:
        return [float(x) for x in self.tokenizer["levels"]]

    def seq_len(self, level: float) -> int:
        """Byte-matched training window: round(512 * r) tokens."""
        return int(round(self.training["base_seq_len"] * level))


def load_study(path: str | Path = DEFAULT) -> Study:
    d = load_yaml(path)
    return Study(**{k: d.get(k) or ({} if k != "seeds" else []) for k in Study.__dataclass_fields__})
