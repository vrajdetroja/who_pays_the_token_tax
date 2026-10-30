# Controls per language

Held-out FineWeb-2, bits per byte. Positive 1: the data-curve slope b (BPB per e-fold of content; 12.5/25/50/100%), mean over seeds with a t-interval, how many seeds' curves fall monotonically, and the lowest R². Positive 2: the rise in BPB when each window's unscored left context is replaced by distant text (mean over a seed's evaluations, then over seeds). Negative 2: stride ¼ window against the main ½, the largest |difference| and how many evaluations exceed the registered 0.001. Negative 1: the Russian swap tokenizer at r = 1.5 minus the main one.


## Model M (SESOI 0.0209)

| language | data-curve b | monotone | min R² | shuffle rise | stride max |Δ| (over 0.001) | swap − main |
|---|---|---|---|---|---|---|
| Arabic | +0.178 [+0.159, +0.196] | 5/5 | 0.962 | +0.0708 [+0.0701, +0.0716] | 0.0017 (20/20) | — |
| Chinese | +0.375 [+0.361, +0.389] | 5/5 | 0.987 | +0.1208 [+0.1198, +0.1218] | 0.0023 (20/20) | — |
| Basque | +0.203 [+0.196, +0.209] | 5/5 | 0.965 | +0.0792 [+0.0785, +0.0799] | 0.0016 (20/20) | — |
| Russian | +0.148 [+0.145, +0.150] | 5/5 | 0.972 | +0.0577 [+0.0575, +0.0580] | 0.0018 (25/25) | -0.0026 [-0.0085, +0.0033] |
| Tamil | +0.088 [+0.081, +0.096] | 5/5 | 0.956 | +0.0315 [+0.0311, +0.0319] | 0.0004 (0/20) | — |

## Model S (SESOI 0.0237)

| language | data-curve b | monotone | min R² | shuffle rise | stride max |Δ| (over 0.001) | swap − main |
|---|---|---|---|---|---|---|
| Arabic | +0.208 [+0.187, +0.229] | 3/3 | 0.999 | +0.0465 [+0.0375, +0.0555] | 0.0007 (0/12) | — |
| Chinese | +0.310 [+0.272, +0.348] | 3/3 | 0.986 | +0.0779 [+0.0649, +0.0909] | 0.0013 (3/12) | — |
| Basque | +0.294 [+0.259, +0.328] | 3/3 | 0.989 | +0.0348 [+0.0230, +0.0466] | 0.0003 (0/12) | — |
| Russian | +0.203 [+0.186, +0.220] | 3/3 | 0.995 | +0.0277 [+0.0230, +0.0323] | 0.0005 (0/15) | -0.0161 [-0.0734, +0.0411] |
| Tamil | +0.112 [+0.102, +0.121] | 3/3 | 0.997 | +0.0202 [+0.0183, +0.0221] | 0.0002 (0/12) | — |
