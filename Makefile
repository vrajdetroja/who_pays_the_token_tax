# Every stage of the study, in order. Training (the pilot and the main grid) needs a GPU;
# everything else runs on a CPU. Python reading downloaded data runs with -I.
# The local venv if there is one, else the Python on the PATH.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
LANGS := eus_Latn rus_Cyrl arb_Arab cmn_Hani tam_Taml
SCRATCH ?= /tmp/recount-smoke

.PHONY: test smoke sim power data tokenizers table1 plan-scout plan-scout2 plan-main analysis appendix figures

test:                ## unit tests: vendored microlm (107) + recount
	$(PY) -m pytest

smoke:               ## end-to-end on CPU with synthetic text (~1 min); must pass before training
	$(PY) -m recount.smoke --root $(SCRATCH)

sim:                 ## the preregistered analysis on simulated data with known truth
	$(PY) -m recount.analysis.simulate --out results/simulated_runs.csv
	$(PY) -m recount.analysis.analyze --runs results/simulated_runs.csv --out results/simulated_analysis.json

power:               ## power by simulation (PREREGISTRATION §6)
	$(PY) -m recount.analysis.power --reps 2000 --out results/power_simulation.csv

data:                ## FineWeb-2 samples, decontamination, splits -> data/text/<lang>/
	$(PY) -I -m recount.data.prepare --lang $(LANGS)

tokenizers:          ## full tokenizers, dose search, shards -> data/tokenizers, data/shards
	$(PY) -I -m recount.tok.build --lang $(LANGS)

table1:              ## token premiums of production tokenizers on FLORES-200
	$(PY) -I -m recount.table1 --out results/table1

plan-scout:          ## the pilot (scout) queue
	$(PY) -m recount.train.plan --grid scout --out queues/scout.yaml

plan-scout2:         ## the second scout (amendment 1): seed noise and data curve at the amended LR
	$(PY) -m recount.train.plan --grid scout2 --out queues/scout2.yaml

plan-main:           ## the main grid queue
	$(PY) -m recount.train.plan --grid main --out queues/main.yaml

appendix:            ## appendix tables from the real runs, and segmentation examples (needs data/)
	$(PY) -m recount.analysis.appendix --runs results/all_runs.csv --out results/appendix
	$(PY) -I -m recount.tok.examples --out results/appendix/segmentation.md

analysis:            ## the preregistered analysis on real runs
	$(PY) -m recount.analysis.analyze --runs results/all_runs.csv --out results/analysis.json

figures:             ## Figure 1 and Figure 2 from a runs table (default: the simulated one)
	$(PY) -m recount.analysis.figures --runs $(or $(RUNS),results/simulated_runs.csv) --out results/figures
