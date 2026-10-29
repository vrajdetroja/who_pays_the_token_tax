"""Post-hoc evaluation of a finished run.

    python -m microlm.eval.run_eval runs/<run_id> [--ckpt ckpt_step400.pt]

The trainer already runs this evaluation at the end of every run (it is in summary.json);
this script is for kept intermediate checkpoints and for re-evaluating on another device.

Writes runs/<run_id>/eval_extrapolation.json: loss and BPB on val_long at each context
length (same documents at every length), plus loss bucketed by position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.loader import TokenData
from ..model import ModelConfig, Transformer
from ..train.checkpoint import load_checkpoint
from ..utils import atomic_write_text, pick_device
from .induction import induction_probe
from .losses import extrapolation_eval


def load_run_model(run_dir: str | Path, device: str, ckpt: str = "model_final.pt"):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    model = Transformer(ModelConfig.from_dict(cfg["model"]))
    state = load_checkpoint(run_dir / ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg, state["step"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--ckpt", default="model_final.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg, step = load_run_model(args.run_dir, device, args.ckpt)
    data = TokenData(cfg["train"]["data_dir"], verify_splits=["val_long"])
    dtype = cfg["train"]["dtype"]
    result = {
        "run_id": cfg["run_id"],
        "step": step,
        "extrapolation": extrapolation_eval(model, data.split("val_long"), data.bos_id, data.token_bytes,
                                            args.lengths, device, dtype, max_docs=args.max_docs),
        "induction": induction_probe(model, data.vocab_size, data.bos_id, device,
                                     n=cfg["train"]["seq_len"] // 2),
    }
    out = Path(args.run_dir) / f"eval_extrapolation_{Path(args.ckpt).stem}.json"
    atomic_write_text(out, json.dumps(result, indent=2))
    for L, r in result["extrapolation"]["lengths"].items():
        print(f"  L={L:>6}: " + (f"loss {r['loss']:.4f}  bpb {r['bpb']:.4f}" if "loss" in r else r["error"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
