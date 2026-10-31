# The canonical-tokenization gap

FLORES-200 devtest sentences, each scored alone after BOS. *Gap*: bits per byte by which scoring only the canonical token sequence overstates the text's BPB, estimated by importance sampling over the token sequences that spell it (Chirkova et al., 2023; recount.eval.canonical); half-draw beside it as a convergence check. *Reaches D*: the gap minus the same unit's r = 1 reference gap, the part that would inflate the level contrast. Mean row: over languages.


## Model M, seed 0 (200 sentences per endpoint)

| r | endpoint | language | canonical BPB | gap | gap, half the draws | draws canonical | reaches D |
|---|---|---|---|---|---|---|---|
| 1 | reference | Arabic | 1.0155 | 0.00000 | 0.00000 | 0% |  |
| 1 | reference | Chinese | 1.7841 | 0.00056 | 0.00004 | 0% |  |
| 1 | reference | Basque | 1.4032 | 0.00000 | 0.00000 | 0% |  |
| 1 | reference | Russian | 0.7992 | 0.00000 | 0.00000 | 0% |  |
| 1 | reference | Tamil | 0.5247 | 0.00000 | 0.00000 | 0% |  |
| 1 | reference | **mean** | 1.1053 | 0.00011 | 0.00001 | 0% |  |
| 1.5 | as deployed | Arabic | 1.0764 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | as deployed | Chinese | 1.9173 | 0.00021 | 0.00013 | 0% | -0.00035 |
| 1.5 | as deployed | Basque | 1.4999 | 0.00000 | 0.00000 | 0% | -0.00000 |
| 1.5 | as deployed | Russian | 0.8741 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | as deployed | Tamil | 0.5435 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | as deployed | **mean** | 1.1822 | 0.00004 | 0.00003 | 0% | -0.00007 |
| 1.5 | content-matched | Arabic | 1.0416 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | content-matched | Chinese | 1.8591 | 0.00023 | 0.00000 | 0% | -0.00033 |
| 1.5 | content-matched | Basque | 1.4532 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | content-matched | Russian | 0.8419 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | content-matched | Tamil | 0.5344 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 1.5 | content-matched | **mean** | 1.1460 | 0.00005 | 0.00000 | 0% | -0.00007 |
| 2.25 | as deployed | Arabic | 1.1227 | 0.00000 | 0.00000 | 0% | -0.00000 |
| 2.25 | as deployed | Chinese | 2.0582 | 0.00000 | 0.00000 | 0% | -0.00056 |
| 2.25 | as deployed | Basque | 1.5685 | 0.00000 | 0.00000 | 0% | -0.00000 |
| 2.25 | as deployed | Russian | 0.9417 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 2.25 | as deployed | Tamil | 0.5717 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 2.25 | as deployed | **mean** | 1.2525 | 0.00000 | 0.00000 | 0% | -0.00011 |
| 2.25 | content-matched | Arabic | 1.0438 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 2.25 | content-matched | Chinese | 1.9167 | 0.00000 | 0.00000 | 0% | -0.00056 |
| 2.25 | content-matched | Basque | 1.4548 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 2.25 | content-matched | Russian | 0.8474 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 2.25 | content-matched | Tamil | 0.5385 | 0.00000 | 0.00000 | 0% | +0.00000 |
| 2.25 | content-matched | **mean** | 1.1602 | 0.00000 | 0.00000 | 0% | -0.00011 |
