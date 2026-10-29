"""The run grid: every endpoint of the design, as jobs in queue order.

    python -m recount.train.plan --grid main  --out queues/main.yaml
    python -m recount.train.plan --grid scout --out queues/scout.yaml

Per (language, model, seed) unit, with W = 64 windows per step and L_r = round(512 r):

    job                   steps                                  branches from
    r = 1.0 main          one pass over the content: N_1 / (W L_1)
      curve_<f>           round(f * steps_1.0)                   the r = 1.0 main run
    r > 1   main          one pass: N_r / (W L_r)  (~ steps_1.0, byte-matched windows)
      deployed            round(N_1 / (W L_r)): the reference token budget T = N_1
    swap (r = 1.5, Russian)  main run with the second tokenizer (negative control 1)
      identity_<r>        (r = 1.0) the share of the pass its level-r deployed branch reads:
                          the same bytes at r = 1, the matched prediction for H3

N_r counts the tokens of the unit's content at level r. A branch with B steps starts from
its parent's full checkpoint at step B - round(0.2 B), where the parent is still in its
constant-LR phase, and decays linearly to zero over the rest (Hagele et al., 2024).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from microlm.model.config import ModelConfig
from microlm.train.config import RunConfig, TrainConfig

from ..runs_table import curve_endpoint
from ..study import Study, load_study
from ..tok.shard import ShardData


@dataclass
class Job:
    name: str
    lang: str
    model: str
    seed: int
    level: float
    tokenizer: str  # main | swap
    endpoint: str  # final | deployed | curve_<f>
    shard_dir: str
    n_docs: int
    seq_len: int
    steps: int
    warmup_steps: int
    content_fraction: float  # of the unit's content this endpoint reads
    train_tokens: int
    train_bytes: int
    r_measured: float
    parent: str | None = None  # name of the main run a branch starts from
    branch_step: int | None = None
    keep_ckpts: list[int] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)  # evaluation controls to run on this endpoint
    overrides: dict = field(default_factory=dict)  # TrainConfig fields that differ from the study (scout LR check)

    @property
    def cost_tokens(self) -> int:
        """Tokens this job actually trains: a branch trains only its cooldown."""
        start = self.branch_step or 0
        return (self.steps - start) * (self.train_tokens // self.steps)

    def run_config(self, study: Study, overrides: dict | None = None) -> RunConfig:
        t, m = study.training, study.models[self.model]
        max_len = max(study.seq_len(lv) for lv in study.levels)
        model = ModelConfig(vocab_size=study.tokenizer["vocab_size"], n_layer=m["n_layer"], d_model=m["d_model"],
                            n_head=m["n_head"], max_seq_len=max_len, pos_type=t["pos_type"])
        tokens_per_step = t["windows_per_step"] * self.seq_len
        train = dict(
            data_dir=self.shard_dir, seed=self.seed, data_seed=self.seed, data_order="sequential",
            seq_len=self.seq_len, tokens_per_step=tokens_per_step, total_tokens=self.steps * tokens_per_step,
            optimizer=t["optimizer"], lr=t["lr"], beta1=t["beta1"], beta2=t["beta2"],
            weight_decay=t["weight_decay"], grad_clip=t["grad_clip"], schedule="wsd",
            warmup_steps=self.warmup_steps, min_lr_ratio=0.0, decay_frac=t["decay_frac"], dtype=t["dtype"],
            keep_ckpts=list(self.keep_ckpts), keep_ckpts_full=True,
        ) | self.overrides | (overrides or {})
        return RunConfig(name=self.name, model=model, train=TrainConfig(**train),
                         notes=f"{self.endpoint} n_docs={self.n_docs}")


def decay_start(steps: int, decay_frac: float) -> int:
    """Step at which a WSD run of `steps` steps starts its cooldown (microlm's formula)."""
    return steps - round(decay_frac * steps)


def shard_dir(root: Path, lang: str, level: float, tokenizer: str) -> Path:
    name = f"r{level:g}" if tokenizer == "main" else f"{tokenizer}_r{level:g}"
    return root / "data" / "shards" / lang / name


def unit_jobs(study: Study, root: Path, lang: str, model: str, seed: int, levels: list[float],
              content_scale: float = 1.0, curve: bool = True, deployed: bool = True,
              swap: bool = False, overrides: dict | None = None, tag: str = "") -> list[Job]:
    """All jobs of one (language, model, seed) unit, parents before their branches."""
    t = study.training
    W, frac = t["windows_per_step"], t["decay_frac"]
    if content_scale != 1.0 and not tag:  # names must differ wherever configs do
        tag = f"-c{content_scale:g}"
    ref = ShardData(shard_dir(root, lang, 1.0, "main"), verify=False)
    content_bytes = int(ref.doc_bytes().sum() * study.models[model]["english_equivalent_fraction"] * content_scale)
    n_docs = ref.docs_for_bytes(content_bytes)
    N1, bytes_ = ref.prefix(n_docs)
    jobs: list[Job] = []
    identity: dict[float, float] = {}  # level r -> share of one pass its deployed branch reads
    if curve and deployed and study.endpoints.get("identity_branches") and 1.0 in levels:
        for lv in levels:
            if lv > 1.0:
                d, L = ShardData(shard_dir(root, lang, lv, "main"), verify=False), study.seq_len(lv)
                identity[lv] = round(N1 / (W * L)) / ((d.prefix(n_docs)[0] - 1) // L // W)
    variants = [(lv, "main") for lv in levels] + ([(study.tokenizer["swap_level"], "swap")] if swap else [])
    for level, tok in variants:
        data = ShardData(shard_dir(root, lang, level, tok), verify=False)
        L = study.seq_len(level)
        N = data.prefix(n_docs)[0]
        steps = (N - 1) // L // W
        warm = max(1, round(t["warmup_frac"] * steps))
        base = f"{lang}-{model}-r{level:g}-{tok}-s{seed}{tag}"
        r_meas = data.manifest["tokenizer"]["meta"].get("r_measured", level)
        main = Job(base, lang, model, seed, level, tok, "final", str(data.dir), n_docs, L, steps, warm, 1.0,
                   steps * W * L, bytes_, r_meas, controls=["shuffle", "stride"], overrides=dict(overrides or {}))
        branches: list[tuple[str, int, float]] = []
        if level == 1.0 and tok == "main" and curve:
            branches += [(curve_endpoint(f), round(f * steps), f) for f in study.endpoints["data_curve_fractions"]]
            branches += [(f"identity_{lv:g}", round(f * steps), f) for lv, f in identity.items()]
        if level > 1.0 and tok == "main" and deployed:
            b = round(N1 / (W * L))
            branches.append(("deployed", b, b / steps))
        for endpoint, b, f in branches:
            start = decay_start(b, frac)
            if not warm < start <= decay_start(steps, frac):
                raise ValueError(f"{base}/{endpoint}: branch at {start} is outside the parent's constant phase")
            main.keep_ckpts.append(start)
        jobs.append(main)
        for endpoint, b, f in branches:
            jobs.append(Job(f"{base}-{endpoint}", lang, model, seed, level, tok, endpoint, str(data.dir), n_docs, L,
                            b, warm, f, b * W * L, int(bytes_ * f), r_meas, parent=base,
                            branch_step=decay_start(b, frac), overrides=dict(overrides or {})))
        main.keep_ckpts = sorted(set(main.keep_ckpts))
    return jobs


def grid(study: Study, root: Path, which: str, langs: list[str] | None = None, models: list[str] | None = None,
         seeds: list[int] | None = None) -> list[Job]:
    """`main`: the preregistered grid. `scout`: the pilot, at 20% of the content (no
    deployed branches, so no level contrast exists to peek at beyond the sealed one)."""
    seeds = seeds if seeds is not None else list(study.seeds)
    models = models or list(study.models)
    jobs: list[Job] = []
    if which == "main":
        for model in models:
            for lang in langs or list(study.languages):
                for seed in seeds:
                    swap = lang == study.tokenizer["swap_language"]
                    jobs += unit_jobs(study, root, lang, model, seed, study.levels, swap=swap)
        return jobs
    if which == "scout":
        sc = study.scout
        lo, hi = min(study.levels), max(study.levels)
        for lr in sc["lrs"]:  # learning-rate check at the extreme levels, seed 0
            if lr == study.training["lr"]:
                continue  # the variance runs below are this point
            jobs += unit_jobs(study, root, sc["lr_language"], "M", 0, [lo, hi], sc["content_scale"], curve=False,
                              deployed=False, overrides={"lr": lr}, tag=f"-scout-lr{lr:g}")
        for lang in sc["variance_languages"]:  # seed noise + data curve (positive control 1)
            for seed in sc["seeds"]:
                jobs += unit_jobs(study, root, lang, "M", seed, [lo, hi], sc["content_scale"], deployed=False,
                                  tag="-scout")
        for seed in sc["swap_seeds"]:  # negative control 1
            jobs += unit_jobs(study, root, study.tokenizer["swap_language"], "M", seed, [study.tokenizer["swap_level"]],
                              sc["content_scale"], curve=False, deployed=False, swap=True, tag="-scout")
        return jobs
    raise ValueError(f"unknown grid {which!r}")


def dump(jobs: list[Job], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([asdict(j) for j in jobs], sort_keys=False))


def load(path: Path) -> list[Job]:
    return [Job(**d) for d in yaml.safe_load(Path(path).read_text())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="main", choices=["main", "scout"])
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    study = load_study(args.study)
    jobs = grid(study, Path(args.root), args.grid, args.langs, args.models, args.seeds)
    dump(jobs, Path(args.out))
    tokens = sum(j.cost_tokens for j in jobs)
    print(f"{len(jobs)} jobs, {tokens / 1e9:.3f}B tokens to train -> {args.out}")


if __name__ == "__main__":
    main()
