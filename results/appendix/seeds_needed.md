# Seeds needed per language

For the content-matched residual D(byte, byte) at r = 2.25: its seed SD within each language (from the seeds run, so itself rough: 4 degrees of freedom for M, 2 for S), and the seeds a per-language claim would need at that SD, with the model's SESOI (the cost of losing 10% of the training text) as the effect of interest. *Detect*: a two-sided t-test at α = 0.05 finds a true residual of one SESOI with power 0.8. *Show negligible*: two one-sided tests against ±SESOI pass with power 0.8 when the true residual is 0. The pooled analysis needs neither: it pools the five languages, which is why per-language results are descriptive.

| model | language | seeds run | seed SD | SESOI | seeds to detect | seeds to show negligible |
|---|---|---|---|---|---|---|
| M | Arabic | 5 | 0.0091 | 0.0209 | 4 | 4 |
| M | Chinese | 5 | 0.0037 | 0.0209 | 3 | 2 |
| M | Basque | 5 | 0.0025 | 0.0209 | 3 | 2 |
| M | Russian | 5 | 0.0027 | 0.0209 | 3 | 2 |
| M | Tamil | 5 | 0.0029 | 0.0209 | 3 | 2 |
| S | Arabic | 3 | 0.0075 | 0.0237 | 4 | 3 |
| S | Chinese | 3 | 0.0513 | 0.0237 | 39 | 42 |
| S | Basque | 3 | 0.0222 | 0.0237 | 9 | 9 |
| S | Russian | 3 | 0.0036 | 0.0237 | 3 | 2 |
| S | Tamil | 3 | 0.0177 | 0.0237 | 7 | 7 |
