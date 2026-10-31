# The canonical-tokenization gap, by beam search (amendment 4)

The sentences of canonical_gap.md, each scored alone after BOS. *Gap*: bits per byte by which scoring only the canonical token sequence overstates the text's BPB, from a beam over the token sequences that spell it (recount.eval.canonical_beam): a lower bound, exact once the beam keeps every spelling. *Reaches D*: the gap minus the same unit's r = 1 reference gap. *Subset*: the first sentences at both widths; B = 16 is called unconverged where B = 64 raises the subset's gap by more than 25% and by more than 1e-05 bits per byte. *Openings*: each subset sentence's longest opening whose canonical tokens at every level are the sentence's own and that has at most 5,000 spellings at r = 1, every spelling scored; *recovered*: the share of the exact non-canonical probability the beam found. Mean row: over languages.


## Model M, seed 0 (200 sentences per endpoint)

| r | endpoint | language | gap, B = 16 | reaches D | subset, B = 16 | subset, B = 64 | converged |
|---|---|---|---|---|---|---|---|
| 1 | reference | Arabic | 0.006348 |  | 0.008421 | 0.008457 | yes |
| 1 | reference | Chinese | 0.013105 |  | 0.009662 | 0.009721 | yes |
| 1 | reference | Basque | 0.008600 |  | 0.008503 | 0.008580 | yes |
| 1 | reference | Russian | 0.006258 |  | 0.009486 | 0.009536 | yes |
| 1 | reference | Tamil | 0.002899 |  | 0.002819 | 0.002843 | yes |
| 1 | reference | **mean** | 0.007442 |  | 0.007778 | 0.007827 |  |
| 1.5 | as deployed | Arabic | 0.003681 | -0.002667 | 0.001126 | 0.001144 | yes |
| 1.5 | as deployed | Chinese | 0.002050 | -0.011055 | 0.003292 | 0.003295 | yes |
| 1.5 | as deployed | Basque | 0.005071 | -0.003529 | 0.003678 | 0.003747 | yes |
| 1.5 | as deployed | Russian | 0.003198 | -0.003060 | 0.003079 | 0.003105 | yes |
| 1.5 | as deployed | Tamil | 0.001197 | -0.001702 | 0.000768 | 0.000778 | yes |
| 1.5 | as deployed | **mean** | 0.003039 | -0.004403 | 0.002389 | 0.002414 |  |
| 1.5 | content-matched | Arabic | 0.002970 | -0.003378 | 0.001381 | 0.001395 | yes |
| 1.5 | content-matched | Chinese | 0.001900 | -0.011205 | 0.004026 | 0.004029 | yes |
| 1.5 | content-matched | Basque | 0.003410 | -0.005190 | 0.003104 | 0.003149 | yes |
| 1.5 | content-matched | Russian | 0.003185 | -0.003072 | 0.001724 | 0.001745 | yes |
| 1.5 | content-matched | Tamil | 0.001157 | -0.001742 | 0.000582 | 0.000602 | yes |
| 1.5 | content-matched | **mean** | 0.002524 | -0.004918 | 0.002163 | 0.002184 |  |
| 2.25 | as deployed | Arabic | 0.000717 | -0.005631 | 0.000800 | 0.000803 | yes |
| 2.25 | as deployed | Chinese | 0.000080 | -0.013026 | 0.000097 | 0.000097 | yes |
| 2.25 | as deployed | Basque | 0.001338 | -0.007261 | 0.002388 | 0.002391 | yes |
| 2.25 | as deployed | Russian | 0.001395 | -0.004862 | 0.000937 | 0.000949 | yes |
| 2.25 | as deployed | Tamil | 0.000389 | -0.002510 | 0.000302 | 0.000305 | yes |
| 2.25 | as deployed | **mean** | 0.000784 | -0.006658 | 0.000905 | 0.000909 |  |
| 2.25 | content-matched | Arabic | 0.000400 | -0.005948 | 0.000461 | 0.000463 | yes |
| 2.25 | content-matched | Chinese | 0.000098 | -0.013008 | 0.000079 | 0.000079 | yes |
| 2.25 | content-matched | Basque | 0.001126 | -0.007474 | 0.001731 | 0.001733 | yes |
| 2.25 | content-matched | Russian | 0.000660 | -0.005598 | 0.000697 | 0.000702 | yes |
| 2.25 | content-matched | Tamil | 0.000271 | -0.002628 | 0.000126 | 0.000129 | yes |
| 2.25 | content-matched | **mean** | 0.000511 | -0.006931 | 0.000619 | 0.000621 |  |

### Openings: exact against the beam (B = 16)

| r | endpoint | language | openings | bytes | exact gap | beam gap | recovered, mean | recovered, lowest |
|---|---|---|---|---|---|---|---|---|
| 1 | reference | Arabic | 25 | 13.8 | 0.002590 | 0.002590 | 100.0% | 100.0% |
| 1 | reference | Chinese | 25 | 19.6 | 0.005685 | 0.005685 | 100.0% | 100.0% |
| 1 | reference | Basque | 22 | 14.0 | 0.025375 | 0.025375 | 100.0% | 100.0% |
| 1 | reference | Russian | 23 | 14.6 | 0.009160 | 0.009160 | 100.0% | 100.0% |
| 1 | reference | Tamil | 21 | 15.5 | 0.001293 | 0.001293 | 100.0% | 100.0% |
| 1 | reference | **mean** | 23 | 15.5 | 0.008821 | 0.008821 | 100.0% | 100.0% |
| 1.5 | as deployed | Arabic | 25 | 13.8 | 0.007235 | 0.007235 | 100.0% | 100.0% |
| 1.5 | as deployed | Chinese | 25 | 19.6 | 0.000300 | 0.000300 | 100.0% | 100.0% |
| 1.5 | as deployed | Basque | 22 | 14.0 | 0.006686 | 0.006686 | 100.0% | 100.0% |
| 1.5 | as deployed | Russian | 23 | 14.6 | 0.010331 | 0.010331 | 100.0% | 100.0% |
| 1.5 | as deployed | Tamil | 21 | 15.5 | 0.000281 | 0.000281 | 100.0% | 100.0% |
| 1.5 | as deployed | **mean** | 23 | 15.5 | 0.004967 | 0.004967 | 100.0% | 100.0% |
| 1.5 | content-matched | Arabic | 25 | 13.8 | 0.013563 | 0.013563 | 100.0% | 100.0% |
| 1.5 | content-matched | Chinese | 25 | 19.6 | 0.000726 | 0.000726 | 100.0% | 100.0% |
| 1.5 | content-matched | Basque | 22 | 14.0 | 0.002501 | 0.002501 | 100.0% | 100.0% |
| 1.5 | content-matched | Russian | 23 | 14.6 | 0.000738 | 0.000738 | 100.0% | 100.0% |
| 1.5 | content-matched | Tamil | 21 | 15.5 | 0.000256 | 0.000256 | 100.0% | 100.0% |
| 1.5 | content-matched | **mean** | 23 | 15.5 | 0.003557 | 0.003557 | 100.0% | 100.0% |
| 2.25 | as deployed | Arabic | 25 | 13.8 | 0.010401 | 0.010401 | 100.0% | 100.0% |
| 2.25 | as deployed | Chinese | 25 | 19.6 | 0.000105 | 0.000105 | 100.0% | 100.0% |
| 2.25 | as deployed | Basque | 22 | 14.0 | 0.000251 | 0.000251 | 100.0% | 100.0% |
| 2.25 | as deployed | Russian | 23 | 14.6 | 0.006786 | 0.006786 | 100.0% | 100.0% |
| 2.25 | as deployed | Tamil | 21 | 15.5 | 0.001441 | 0.001441 | 100.0% | 100.0% |
| 2.25 | as deployed | **mean** | 23 | 15.5 | 0.003797 | 0.003797 | 100.0% | 100.0% |
| 2.25 | content-matched | Arabic | 25 | 13.8 | 0.001133 | 0.001133 | 100.0% | 100.0% |
| 2.25 | content-matched | Chinese | 25 | 19.6 | 0.000056 | 0.000056 | 100.0% | 100.0% |
| 2.25 | content-matched | Basque | 22 | 14.0 | 0.000063 | 0.000063 | 100.0% | 100.0% |
| 2.25 | content-matched | Russian | 23 | 14.6 | 0.001666 | 0.001666 | 100.0% | 100.0% |
| 2.25 | content-matched | Tamil | 21 | 15.5 | 0.001460 | 0.001460 | 100.0% | 100.0% |
| 2.25 | content-matched | **mean** | 23 | 15.5 | 0.000875 | 0.000875 | 100.0% | 100.0% |
