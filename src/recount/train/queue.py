"""Run a queue of jobs in order: train (or resume), then evaluate, one job at a time.

    python -m recount.train.queue queues/main.yaml --set micro_batch_size=32 compile=true

Queue order puts each main run before its branches, so a branch's parent checkpoint
exists when it starts. Finished jobs are skipped and interrupted ones resume, so after a
crash the same command picks up where it stopped. `--set` takes only operational fields
(they never change a run's ID).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from microlm.train.config import OPERATIONAL_FIELDS

from ..study import load_study
from . import plan
from .trainer import evaluate_job, find_run, train_job


def run_queue(jobs: list[plan.Job], study, root: Path, operational: dict, eval_device: str = "auto",
              runs_csv: str = "results/all_runs.csv") -> int:
    done = 0
    for i, job in enumerate(jobs):
        print(f"\n=== [{i + 1}/{len(jobs)}] {job.name}", flush=True)
        train_job(job, study, root, operational)
        out = find_run(root, job.name)
        if not (out / "evaluated.json").exists():
            evaluate_job(job, study, root, device=eval_device, operational=operational, runs_csv=runs_csv)
        done += 1
    return done


def parse_operational(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        if k not in OPERATIONAL_FIELDS:
            raise SystemExit(f"--set {k}: only operational fields are allowed ({sorted(OPERATIONAL_FIELDS)})")
        out[k] = yaml.safe_load(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue")
    ap.add_argument("--study", default="configs/study.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--set", nargs="*", default=[], dest="operational")
    ap.add_argument("--eval-device", default="auto")
    ap.add_argument("--runs-csv", default="results/all_runs.csv", help="results/scout_runs.csv for the scout")
    args = ap.parse_args()
    run_queue(plan.load(Path(args.queue)), load_study(args.study), Path(args.root),
              parse_operational(args.operational), args.eval_device, args.runs_csv)


if __name__ == "__main__":
    main()
