"""The training loop.

    python -m microlm.train.trainer --config configs/baseline.yaml [--set train.seed=1 ...]

Rerunning the same command resumes from runs/<run_id>/ckpt_latest.pt if it exists, so an
interrupted run loses at most `ckpt_interval` steps. A run that already finished
(summary.json exists) is skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ..data.loader import TokenData, WindowStream
from ..eval.induction import induction_probe
from ..eval.losses import evaluate_tokens, extrapolation_eval
from ..model import Transformer, flops_per_token
from ..utils import atomic_write_text, git_sha, pick_device
from .checkpoint import load_checkpoint, save_checkpoint, save_model_only, set_rng_state
from .config import RunConfig, load_run_config
from .optim import build_optimizer
from .schedule import lr_multiplier

# Dense bf16 tensor-core peaks (FLOP/s), for MFU. Override with train.peak_flops.
PEAK_FLOPS = {
    "H100 SXM": 989e12, "H100 80GB HBM3": 989e12, "H100 PCIe": 756e12, "H100": 756e12,
    "A100": 312e12, "L40S": 362e12, "RTX 6000 Ada": 364e12, "RTX 4090": 165e12,
    "RTX 3090": 71e12, "A10G": 70e12, "A10": 125e12, "L4": 121e12, "RTX A6000": 155e12,
}


def peak_flops(device: str, override: float | None) -> float | None:
    if override:
        return override
    if not device.startswith("cuda"):
        return None
    name = torch.cuda.get_device_name()
    for key, value in PEAK_FLOPS.items():  # most specific keys first
        if key in name:
            return value
    return None


def hardware() -> dict:
    hw = {"platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name()
        hw["cuda"] = torch.version.cuda
    return hw


class Logger:
    def __init__(self, path: Path, wandb_run=None):
        self.f = open(path, "a")
        self.wandb = wandb_run

    def log(self, record: dict) -> None:
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        if self.wandb is not None:
            self.wandb.log(record, step=record["step"])


def truncate_metrics(path: Path, step: int) -> None:
    """Drop records logged after the checkpoint we resume from; they will be logged again."""
    if not path.exists():
        return
    keep = [ln for ln in path.read_text().splitlines() if ln and json.loads(ln)["step"] <= step]
    atomic_write_text(path, "".join(ln + "\n" for ln in keep))


def final_train_loss(path: Path, last: int = 10) -> float | None:
    losses = [json.loads(ln)["train_loss"] for ln in path.read_text().splitlines() if ln]
    return float(np.mean(losses[-last:])) if losses else None


@torch.no_grad()
def diagnostics(model, x: torch.Tensor, autocast) -> dict:
    """Per-layer residual RMS and attention entropy (nats) on a fixed batch."""
    rms, entropy = [], []

    def on_probs(p):
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)  # (B, H, T)
        entropy.append(ent.mean().item())

    model.eval()
    with autocast:
        model(x, probs_hook=on_probs, resid_hook=lambda r: rms.append(r.float().pow(2).mean().sqrt().item()))
    model.train()
    return {"resid_rms": rms, "attn_entropy": entropy}


def train(run: RunConfig) -> dict | None:
    tc = run.train
    device = pick_device(tc.device)
    dev_type = device.split(":")[0]

    data = TokenData(tc.data_dir, verify=tc.verify_data)
    run.model.vocab_size = data.vocab_size  # vocab is a property of the data, not a free choice
    run_id = run.run_id(data.hash)
    out = Path(tc.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"{run_id}: already finished, skipping")
        return json.loads((out / "summary.json").read_text())
    cfg_record = run.to_dict() | {"run_id": run_id, "data_identity": data.hash}
    atomic_write_text(out / "config.json", json.dumps(cfg_record, indent=2))

    # -- model, optimizer, data -------------------------------------------------------
    random.seed(tc.seed)
    np.random.seed(tc.seed)
    torch.manual_seed(tc.seed)
    model = Transformer(run.model).to(device)
    counts = model.check_param_counts()
    optimizer = build_optimizer(model, tc)
    stream = WindowStream(data.split("train"), tc.seq_len, tc.data_order, tc.data_seed)
    W, mb, accum, total_steps = tc.windows_per_step, tc.micro_batch_size, tc.grad_accum, tc.total_steps
    if total_steps * W > stream.n_windows:
        raise ValueError(f"{total_steps} steps need {total_steps * W:,} windows; train split has {stream.n_windows:,}")
    val = data.split("val_iid")
    diag_x = torch.from_numpy(np.stack([val[i * tc.seq_len : (i + 1) * tc.seq_len] for i in range(2)]).astype(np.int64)).to(device)

    fpt = flops_per_token(run.model, tc.seq_len)
    peak = peak_flops(device, tc.peak_flops)

    start_step, train_seconds = 0, 0.0
    ckpt_path = out / "ckpt_latest.pt"
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        if ck["run_id"] != run_id:
            raise ValueError(f"checkpoint belongs to {ck['run_id']}, not {run_id}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        start_step, train_seconds = ck["step"], ck["train_seconds"]
        truncate_metrics(out / "metrics.jsonl", start_step)
        print(f"resumed {run_id} at step {start_step}")
    elif tc.init_from:
        ck = load_checkpoint(tc.init_from)
        parent = RunConfig.from_dict(ck["run_config"])
        for s in range(ck["step"]):  # the branch must share the parent's schedule up to here
            if not math.isclose(lr_multiplier(s, parent.train), lr_multiplier(s, tc)):
                raise ValueError(f"init_from: schedules diverge at step {s}; this is not a valid branch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        start_step, train_seconds = ck["step"], ck["train_seconds"]
        print(f"branched from {tc.init_from} at step {start_step}")

    wandb_run = None
    if tc.wandb_project:
        import wandb

        wandb_run = wandb.init(project=tc.wandb_project, name=run_id, id=run_id, resume="allow", config=cfg_record)
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

    def evaluate(max_tokens):
        r = evaluate_tokens(fwd, val, data.token_bytes, tc.seq_len, max_tokens, mb, device, tc.dtype)
        return {"val_loss": r["loss"], "val_bpb": r["bpb"], "val_tokens": r["n_tokens"]}

    print(f"{run_id}: {counts['non_embedding'] / 1e6:.2f}M non-embedding / {counts['total'] / 1e6:.2f}M total params, "
          f"{total_steps} steps x {tc.tokens_per_step:,} tokens, accum {accum} x {mb}, device {device}")

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
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)  # returns the pre-clip norm
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss = (loss_sum / accum).item()  # also synchronizes
        sync()
        dt = time.perf_counter() - t0
        train_seconds += dt
        steps_done = step + 1

        tok_s = tc.tokens_per_step / dt
        rec = {
            "step": step + 1,
            "tokens": (step + 1) * tc.tokens_per_step,
            "train_loss": train_loss,
            "lr_mult": mult,
            "grad_norm": grad_norm.item(),
            "step_seconds": dt,
            "train_seconds": train_seconds,
            "tokens_per_sec": tok_s,
        }
        if peak:
            rec["mfu"] = fpt["total"] * tok_s / peak

        if not math.isfinite(train_loss):
            diverged = True
            logger.log(rec | {"diverged": True})
            print(f"step {step + 1}: loss is {train_loss}; stopping")
            break
        last = step + 1 == total_steps
        if (step + 1) % tc.eval_interval == 0 or last or step == start_step:
            rec |= evaluate(tc.eval_tokens)
        if tc.diag_interval and ((step + 1) % tc.diag_interval == 0 or step == start_step):
            rec |= diagnostics(model, diag_x, autocast)
        if (step + 1) % tc.log_interval == 0 or "val_loss" in rec:
            msg = f"step {step + 1:>5}/{total_steps}  loss {train_loss:.4f}  gnorm {rec['grad_norm']:.3f}  " \
                  f"lr {mult:.3f}  {tok_s / 1e3:7.1f}k tok/s"
            if "mfu" in rec:
                msg += f"  mfu {rec['mfu']:.1%}"
            if "val_loss" in rec:
                msg += f"  val {rec['val_loss']:.4f} ({rec['val_bpb']:.4f} bpb)"
            print(msg, flush=True)
        logger.log(rec)

        stopping = tc.stop_at_step is not None and step + 1 >= tc.stop_at_step and not last
        if (step + 1) % tc.ckpt_interval == 0 or last or stopping or (step + 1) in tc.keep_ckpts:
            extra = dict(run_id=run_id, run_config=run.to_dict(), git_sha=git_sha(),
                         train_seconds=train_seconds, data_cursor=(step + 1) * W)
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, step=step + 1, **extra)
            if (step + 1) in tc.keep_ckpts:
                kept = out / f"ckpt_step{step + 1}.pt"
                if tc.keep_ckpts_full:
                    save_checkpoint(kept, model=model, optimizer=optimizer, step=step + 1, **extra)
                else:  # weights only: ~4x smaller, enough for evaluation and induction-head analysis
                    save_model_only(kept, model=model, step=step + 1, **extra)
        if stopping:
            print(f"{run_id}: stopped at step {step + 1} (stop_at_step); rerun the same command to continue")
            return None

    # -- final evaluation ---------------------------------------------------------------
    summary = {
        "run_id": run_id,
        "name": run.name,
        "config": run.to_dict(),
        "data_identity": data.hash,
        "git_sha": git_sha(),
        "hardware": hardware(),
        "params": counts,
        "flops_per_token": fpt,
        "steps": steps_done,
        "tokens": steps_done * tc.tokens_per_step,
        "train_flops": fpt["total"] * steps_done * tc.tokens_per_step,
        "train_seconds": train_seconds,
        "diverged": diverged,
        "final_train_loss": final_train_loss(out / "metrics.jsonl"),
    }
    if not diverged:
        final = evaluate_tokens(fwd, val, data.token_bytes, tc.seq_len, tc.final_eval_tokens, mb, device,
                                tc.dtype, position_loss=True)
        summary |= {"val_iid_loss": final["loss"], "val_iid_bpb": final["bpb"], "val_iid_tokens": final["n_tokens"]}
        summary["val_iid_position_loss"] = final["position_loss"]
        if data.has_split("val_ood"):
            ood = evaluate_tokens(fwd, data.split("val_ood"), data.token_bytes, tc.seq_len, None, mb, device, tc.dtype)
            summary |= {"val_ood_loss": ood["loss"], "val_ood_bpb": ood["bpb"]}
        if data.has_split("val_long") and tc.extrapolation_lengths:
            summary["extrapolation"] = extrapolation_eval(
                model, data.split("val_long"), data.bos_id, data.token_bytes, tc.extrapolation_lengths,
                device, tc.dtype, max_docs=tc.extrapolation_docs)
        probe = induction_probe(model, data.vocab_size, data.bos_id, device, n=tc.seq_len // 2)
        summary |= {f"induction_{k}": v for k, v in probe.items()}
        if peak and train_seconds:
            summary["mfu"] = fpt["total"] * summary["tokens"] / train_seconds / peak
    # A finished run keeps its weights only (~4x smaller than the resumable checkpoint).
    save_model_only(out / "model_final.pt", model=model, step=steps_done, run_id=run_id,
                    run_config=run.to_dict(), git_sha=git_sha())
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2))
    ckpt_path.unlink(missing_ok=True)
    print(f"{run_id}: done. val_iid {summary.get('val_iid_loss', float('nan')):.4f} nats, "
          f"{summary.get('val_iid_bpb', float('nan')):.4f} bpb, {train_seconds / 60:.1f} min")
    if wandb_run is not None:
        wandb_run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb_run.finish()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="e.g. train.lr=6e-3 model.pos_type=rope")
    args = ap.parse_args()
    train(load_run_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
