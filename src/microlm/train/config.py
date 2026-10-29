"""Run configuration: YAML files with inheritance, CLI overrides, and a semantic hash.

A run is (ModelConfig, TrainConfig, data identity). Its ID is a hash of every field that
can change the trained model. Purely operational fields (logging cadence, device, output
location, micro-batch size, compilation) are excluded, so they can differ between machines
without creating a "different" run. Everything else is in the hash, which turns "did I
change only one thing?" into a mechanical check:

    python -m microlm.train.config configs/baseline.yaml configs/ablations/pos_rope.yaml
"""

from __future__ import annotations

import argparse
import copy
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ..model.config import ModelConfig
from ..utils import hash_obj

class _Loader(yaml.SafeLoader):
    """PyYAML follows YAML 1.1, where a float needs a dot: `1e-3` loads as the *string*
    "1e-3". Add the YAML 1.2 forms so exponent-only numbers are floats."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def _yaml(text: str):
    return yaml.load(text, Loader=_Loader)


def check_types(obj, section: str) -> None:
    """Fail at load time, with the field's name, if a numeric field holds anything else."""
    for f in fields(obj):
        v = getattr(obj, f.name)
        kinds = {t.strip() for t in str(f.type).split("|")}
        if v is None and "None" in kinds:
            continue
        if "bool" in kinds:
            ok = isinstance(v, bool)
        elif "float" in kinds:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            if ok:
                setattr(obj, f.name, float(v))
        elif "int" in kinds:
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            continue
        if not ok:
            raise ValueError(f"{section}.{f.name} must be {f.type}, got {v!r} ({type(v).__name__})")


SCHEDULES = ("constant", "cosine", "linear", "wsd")
OPTIMIZERS = ("adamw", "lion", "muon")


@dataclass
class TrainConfig:
    data_dir: str = "data/fineweb_edu/bpe_8192"
    seed: int = 0  # model initialization
    data_seed: int = 0  # order of training windows; fixed across the whole study
    data_order: str = "shuffled"  # shuffled | sequential

    seq_len: int = 512
    tokens_per_step: int = 131_072  # held constant across every run (grad accumulation absorbs micro-batch changes)
    total_tokens: int = 212_336_640  # 1620 steps; ~20 tokens per non-embedding param of the unit model

    optimizer: str = "adamw"
    lr: float = 4e-3  # peak LR for AdamW/Lion; for Muon, the LR of its AdamW-managed params
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    schedule: str = "cosine"
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    decay_frac: float = 0.2  # WSD: fraction of steps spent in the final decay

    dtype: str = "bf16"  # bf16 autocast with fp32 master weights; fp32 for CPU debugging
    eval_tokens: int = 1_048_576  # tokens per periodic val_iid eval
    final_eval_tokens: int | None = None  # None = the whole val_iid split
    init_from: str | None = None  # checkpoint to branch from (e.g. a WSD stable-phase checkpoint)

    # -- operational: excluded from the run hash ---------------------------------------
    micro_batch_size: int = 64
    compile: bool = True
    device: str = "auto"
    eval_interval: int = 200
    log_interval: int = 20
    diag_interval: int = 200
    ckpt_interval: int = 200
    keep_ckpts: list[int] = field(default_factory=list)  # extra permanent checkpoints, e.g. for induction-head timing
    keep_ckpts_full: bool = False  # kept checkpoints include optimizer state (needed to branch a WSD decay)
    peak_flops: float | None = None
    wandb_project: str | None = None
    out_dir: str = "runs"
    verify_data: bool = True
    stop_at_step: int | None = None  # end this invocation at this step (with a checkpoint); rerun to continue
    extrapolation_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    extrapolation_docs: int = 200

    def __post_init__(self):
        check_types(self, "train")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.tokens_per_step % self.seq_len:
            raise ValueError("tokens_per_step must be a multiple of seq_len")
        if self.windows_per_step % self.micro_batch_size:
            raise ValueError(
                f"micro_batch_size {self.micro_batch_size} must divide {self.windows_per_step} windows per step"
            )

    @property
    def windows_per_step(self) -> int:
        return self.tokens_per_step // self.seq_len

    @property
    def grad_accum(self) -> int:
        return self.windows_per_step // self.micro_batch_size

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.tokens_per_step


OPERATIONAL_FIELDS = {
    "micro_batch_size", "compile", "device", "eval_interval", "log_interval", "diag_interval",
    "ckpt_interval", "keep_ckpts", "keep_ckpts_full", "peak_flops", "wandb_project", "out_dir", "verify_data",
    "stop_at_step", "extrapolation_lengths", "extrapolation_docs",
}


@dataclass
class RunConfig:
    name: str
    model: ModelConfig
    train: TrainConfig
    notes: str = ""

    def semantic(self, data_identity: str | None = None) -> dict:
        train = {k: v for k, v in asdict(self.train).items() if k not in OPERATIONAL_FIELDS}
        out = {"model": self.model.hash_dict(), "train": train}
        if data_identity is not None:
            out["data"] = data_identity
        return out

    def run_id(self, data_identity: str | None = None) -> str:
        return f"{self.name}__{hash_obj(self.semantic(data_identity), 10)}"

    def to_dict(self) -> dict:
        return {"name": self.name, "notes": self.notes, "model": self.model.to_dict(), "train": asdict(self.train)}

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        train_known = {f.name for f in fields(TrainConfig)}
        unknown = set(d.get("train", {})) - train_known
        if unknown:
            raise ValueError(f"unknown TrainConfig fields: {sorted(unknown)}")
        extra = set(d) - {"name", "notes", "model", "train"}
        if extra:
            raise ValueError(f"unknown top-level config keys: {sorted(extra)}")
        return cls(
            name=d["name"],
            notes=d.get("notes", ""),
            model=ModelConfig.from_dict(d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
        )


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    """Load a config, resolving `inherit:` chains (paths relative to the including file)."""
    path = Path(path)
    d = _yaml(path.read_text()) or {}
    parent = d.pop("inherit", None)
    if parent is not None:
        d = _deep_merge(load_yaml(path.parent / parent), d)
    return d


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    """`train.lr=6e-3` style overrides; values are parsed as YAML scalars."""
    d = copy.deepcopy(d)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        value = _yaml(raw)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return d


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    return RunConfig.from_dict(apply_overrides(load_yaml(path), overrides or []))


def diff(a: RunConfig, b: RunConfig) -> dict[str, tuple[Any, Any]]:
    """Semantic fields that differ between two runs."""
    fa, fb = _flatten(a.semantic()), _flatten(b.semantic())
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Show the semantic difference between two run configs.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--set-a", nargs="*", default=[])
    ap.add_argument("--set-b", nargs="*", default=[])
    args = ap.parse_args()
    a, b = load_run_config(args.a, args.set_a), load_run_config(args.b, args.set_b)
    d = diff(a, b)
    print(f"{a.name}: {a.run_id()}\n{b.name}: {b.run_id()}")
    if not d:
        print("identical semantic configs")
    for k, (va, vb) in d.items():
        print(f"  {k}: {va!r} -> {vb!r}")
    if len(d) > 1:
        print(f"WARNING: {len(d)} fields differ")


if __name__ == "__main__":
    main()
