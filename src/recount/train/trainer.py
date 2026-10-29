"""Train and evaluate one endpoint (a `plan.Job`).

The loop is microlm's (microlm.train.trainer.train), unchanged in what it computes: same
model, optimizer, WSD schedule, bf16 autocast, gradient clipping, exact-resume checkpoints,
and run IDs that hash every field able to change the trained model. What differs is the
data, read in document order from a dose shard (recount.train.stream), and branching:
a main run saves full checkpoints at its branches' start steps, and a branch resumes from
its parent's checkpoint with its own (shorter) schedule.

Rerunning a job resumes it; a finished job (summary.json) is skipped.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from microlm.model import flops_per_token
from microlm.train.checkpoint import load_checkpoint, save_checkpoint, save_model_only, set_rng_state
from microlm.train.config import RunConfig
from microlm.train.optim import build_optimizer
from microlm.train.schedule import lr_multiplier
from microlm.train.trainer import Logger, final_train_loss, hardware, peak_flops, truncate_metrics
from microlm.utils import atomic_write_text, git_sha, pick_device

from .. import runs_table
from ..eval.protocols import evaluate_stream
from ..model import build as build_model
from ..runs_table import budget_of
from ..study import Study
from ..tok.shard import ShardData
from .plan import Job
from .stream import DocStream


def masking(study: Study) -> bool:
    return bool(study.training.get("mask_unreachable", False))


def data_identity(data: ShardData, study: Study) -> str:
    """What the run ID hashes besides the configs: the shard, and whether rows are masked."""
    return data.hash + ("|mask_unreachable" if masking(study) else "")


def run_dir(root: Path, job: Job, study: Study, data: ShardData, overrides: dict | None = None) -> tuple[RunConfig, Path]:
    run = job.run_config(study, overrides)
    return run, root / "runs" / run.run_id(data_identity(data, study))


def train_job(job: Job, study: Study, root: Path, operational: dict | None = None) -> dict:
    """Train `job` to completion (or resume it). `operational` holds TrainConfig fields
    that do not enter the run ID: micro_batch_size, compile, device, log/eval cadence."""
    data = ShardData(job.shard_dir)
    run, out = run_dir(root, job, study, data, operational)
    tc = run.train
    device = pick_device(tc.device)
    dev_type = device.split(":")[0]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"{out.name}: already finished, skipping")
        return json.loads((out / "summary.json").read_text())
    if run.model.vocab_size != data.vocab_size:
        raise ValueError(f"model vocab {run.model.vocab_size} != tokenizer vocab {data.vocab_size}")
    atomic_write_text(out / "config.json", json.dumps(run.to_dict() | {"job": job.__dict__, "data": data_identity(data, study)},
                                                      indent=2))

    random.seed(tc.seed)
    np.random.seed(tc.seed)
    torch.manual_seed(tc.seed)  # identical initialization at every level of a seed (same shapes)
    model = build_model(run.model, data.tokenizer.unreachable_ids(), masking(study)).to(device)
    counts = model.base.check_param_counts()
    optimizer = build_optimizer(model.base, tc)
    stream = DocStream(data.split("train"), data.bos_id, tc.seq_len, tc.data_seed, job.n_docs)
    W, mb, accum, total_steps = tc.windows_per_step, tc.micro_batch_size, tc.grad_accum, tc.total_steps
    if total_steps * W > stream.n_windows:
        raise ValueError(f"{total_steps} steps need {total_steps * W:,} windows; content has {stream.n_windows:,}")
    fpt = flops_per_token(run.model, tc.seq_len)
    peak = peak_flops(device, tc.peak_flops)

    start_step, train_seconds = 0, 0.0
    ckpt_path = out / "ckpt_latest.pt"
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        start_step, train_seconds = ck["step"], ck["train_seconds"]
        truncate_metrics(out / "metrics.jsonl", start_step)
        print(f"resumed {out.name} at step {start_step}")
    elif job.parent is not None:
        ck = load_checkpoint(find_run(root, job.parent) / f"ckpt_step{job.branch_step}.pt")
        parent = RunConfig.from_dict(ck["run_config"])
        for s in range(ck["step"]):  # a branch must share its parent's schedule up to the branch point
            if not math.isclose(lr_multiplier(s, parent.train), lr_multiplier(s, tc)):
                raise ValueError(f"{job.name}: schedule diverges from {job.parent} at step {s}")
        if ck["step"] != job.branch_step:
            raise ValueError(f"{job.name}: parent checkpoint is at step {ck['step']}, not {job.branch_step}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        start_step, train_seconds = ck["step"], ck["train_seconds"]
        print(f"{out.name}: branched from {job.parent} at step {start_step}")

    wandb_run = None
    if tc.wandb_project:
        import wandb

        wandb_run = wandb.init(project=tc.wandb_project, name=out.name, id=out.name, resume="allow",
                               config=run.to_dict() | {"job": job.__dict__})
    logger = Logger(out / "metrics.jsonl", wandb_run)
    fwd = torch.compile(model) if (tc.compile and dev_type == "cuda") else model
    use_bf16 = tc.dtype == "bf16" and dev_type != "cpu"
    autocast = torch.autocast(device_type=dev_type, dtype=torch.bfloat16) if use_bf16 else nullcontext()
    params = [p for p in model.parameters() if p.requires_grad]

    def sync():
        if dev_type == "cuda":
            torch.cuda.synchronize()
        elif dev_type == "mps":
            torch.mps.synchronize()

    print(f"{out.name}: {counts['non_embedding'] / 1e6:.2f}M non-embedding params, {total_steps} steps x "
          f"{tc.tokens_per_step:,} tokens (seq {tc.seq_len}), from step {start_step}, device {device}")
    diverged, steps_done = False, start_step
    for step in range(start_step, total_steps):
        mult = lr_multiplier(step, tc)
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * mult
        sync()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for k in range(accum):
            x, y = stream.batch(step * W + k * mb, mb, device)
            with autocast:
                _, loss = fwd(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss = (loss_sum / accum).item()
        sync()
        dt = time.perf_counter() - t0
        train_seconds += dt
        steps_done = step + 1
        tok_s = tc.tokens_per_step / dt
        rec = {"step": step + 1, "tokens": (step + 1) * tc.tokens_per_step, "train_loss": train_loss,
               "lr_mult": mult, "grad_norm": grad_norm.item(), "step_seconds": dt,
               "train_seconds": train_seconds, "tokens_per_sec": tok_s}
        if peak:
            rec["mfu"] = fpt["total"] * tok_s / peak
        if not math.isfinite(train_loss):
            diverged = True
            logger.log(rec | {"diverged": True})
            print(f"step {step + 1}: loss is {train_loss}; stopping")
            break
        last = step + 1 == total_steps
        if (step + 1) % tc.log_interval == 0 or last:
            msg = (f"step {step + 1:>5}/{total_steps}  loss {train_loss:.4f}  gnorm {rec['grad_norm']:.3f}  "
                   f"lr {mult:.3f}  {tok_s / 1e3:7.1f}k tok/s")
            print(msg + (f"  mfu {rec['mfu']:.1%}" if "mfu" in rec else ""), flush=True)
        logger.log(rec)
        stopping = tc.stop_at_step is not None and step + 1 >= tc.stop_at_step and not last
        if (step + 1) % tc.ckpt_interval == 0 or last or stopping or (step + 1) in tc.keep_ckpts:
            extra = dict(run_id=out.name, run_config=run.to_dict(), git_sha=git_sha(), train_seconds=train_seconds)
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, step=step + 1, **extra)
            if (step + 1) in tc.keep_ckpts:
                save_checkpoint(out / f"ckpt_step{step + 1}.pt", model=model, optimizer=optimizer, step=step + 1, **extra)
        if stopping:
            print(f"{out.name}: stopped at step {step + 1} (stop_at_step); rerun to continue")
            return {}

    summary = {
        "run_id": out.name, "job": job.__dict__, "config": run.to_dict(), "data_identity": data_identity(data, study),
        "masked_rows": int(model.unreachable.sum()),
        "git_sha": git_sha(), "hardware": hardware(), "params": counts, "flops_per_token": fpt,
        "steps": steps_done, "tokens": steps_done * tc.tokens_per_step,
        "train_flops": fpt["total"] * steps_done * tc.tokens_per_step,
        "trained_tokens": (steps_done - (job.branch_step or 0)) * tc.tokens_per_step,
        "train_seconds": train_seconds, "diverged": diverged,
        "final_train_loss": final_train_loss(out / "metrics.jsonl"),
    }
    if peak and train_seconds:
        summary["mfu"] = fpt["total"] * summary["trained_tokens"] / train_seconds / peak
    save_model_only(out / "model_final.pt", model=model, step=steps_done, run_id=out.name,
                    run_config=run.to_dict(), git_sha=git_sha())
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2))
    ckpt_path.unlink(missing_ok=True)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"{out.name}: trained {summary['trained_tokens'] / 1e6:.1f}M tokens in {train_seconds / 60:.1f} min")
    return summary


def find_run(root: Path, name: str) -> Path:
    hits = sorted((root / "runs").glob(f"{name}__*"))
    hits = [h for h in hits if h.name.rsplit("__", 1)[0] == name]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one run directory for {name}, found {[h.name for h in hits]}")
    return hits[0]


# ---------------------------------------------------------------------------------------
# Evaluation


def evaluate_job(job: Job, study: Study, root: Path, device: str = "auto", batch_size: int = 16,
                 operational: dict | None = None, runs_csv: str = "results/all_runs.csv") -> list[dict]:
    """Evaluate a finished endpoint under both context protocols (and its controls), and
    append the rows to `runs_csv` (relative to root). Returns the rows."""
    data = ShardData(job.shard_dir)
    run, out = run_dir(root, job, study, data, operational)
    summary = json.loads((out / "summary.json").read_text())
    if summary["diverged"]:
        print(f"{out.name}: diverged; not evaluated")
        return []
    device = pick_device(device)
    model = build_model(run.model, data.tokenizer.unreachable_ids(), masking(study)).to(device)
    model.load_state_dict(load_checkpoint(out / "model_final.pt", map_location=device)["model"])
    ev = study.evaluation
    base_w = study.training["base_seq_len"]
    unreachable = torch.tensor(data.tokenizer.unreachable_ids(), dtype=torch.long)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    common = {
        "lang": job.lang, "model": job.model, "seed": job.seed, "level": job.level, "tokenizer": job.tokenizer,
        "endpoint": job.endpoint, "budget": budget_of(job.endpoint, job.level), "r_measured": job.r_measured,
        "content_fraction": job.content_fraction, "train_tokens": summary["tokens"], "train_bytes": job.train_bytes,
        "steps": summary["steps"], "train_flops": summary["train_flops"], "run_id": out.name,
        "endpoint_id": out.name, "git_sha": git_sha(), "timestamp": stamp,
    }
    if job.overrides:  # scout LR check: not part of the design grid
        common["tokenizer"] = job.tokenizer + "".join(f"+{k}={v:g}" for k, v in sorted(job.overrides.items()))
    rows: list[dict] = []
    sets = ["heldout"] + (["flores"] if data.has_split("flores") else [])
    protocols = {"tok": base_w, "byte": job.seq_len}
    for eval_set in sets:
        tokens = data.split(eval_set)
        n_sent = data.manifest["splits"][eval_set].get("n_sentences")
        cache: dict[tuple, object] = {}

        def score(window: int, stride: int, shuffle: bool, wasted: bool):
            key = (window, stride, shuffle)
            if key not in cache:
                cache[key] = evaluate_stream(model, tokens, data.token_bytes, window, stride, device, batch_size,
                                             study.training["dtype"], ev["shuffle_seed"] if shuffle else None,
                                             unreachable if wasted and len(unreachable) else None)
            return cache[key]

        for ctx, window in protocols.items():
            stride = max(1, round(window * ev["context_stride_frac"]))
            res = score(window, stride, False, True)
            row = common | {"ctx": ctx, "eval_set": eval_set, "control": "", "bpb": res.bpb, "nats": res.nats,
                            "n_bytes": res.n_bytes, "n_tokens": res.n_tokens, "window": window, "stride": stride,
                            "wasted_mass": res.wasted_mass if res.wasted_mass is not None else ""}
            if n_sent:
                row |= {"n_sentences": n_sent, "bits_per_sentence": res.bits_per(n_sent)}
            rows.append(row)
            if eval_set != "heldout":
                continue
            if "shuffle" in job.controls:
                sh = score(window, stride, True, False)
                rows.append(row | {"control": "shuffle", "bpb": sh.bpb, "nats": sh.nats, "wasted_mass": ""})
            if "stride" in job.controls and ctx == "byte":
                s2 = max(1, round(window * ev["robustness_stride_frac"]))
                st = score(window, s2, False, False)
                for c in protocols if job.level == 1.0 else [ctx]:  # at r = 1 the protocols coincide
                    rows.append(row | {"ctx": c, "control": "stride", "bpb": st.bpb, "nats": st.nats,
                                       "stride": s2, "wasted_mass": ""})
    runs_table.append_rows(root / runs_csv, rows)
    (out / "evaluated.json").write_text(json.dumps({"rows": len(rows), "timestamp": stamp}))
    print(f"{out.name}: " + "  ".join(f"{r['eval_set']}/{r['ctx']}{'/' + r['control'] if r['control'] else ''} "
                                      f"{r['bpb']:.4f}" for r in rows))
    return rows
