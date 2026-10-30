# Training conditions

Means over the units (language × seed) of each condition. Ratios are against each unit's own r = 1 reference, which reads the whole content once. Training FLOPs per token follow microlm: 6 × non-embedding parameters, plus the unembedding (6 d V) and attention (12 L T d, growing with the window T), times the tokens trained. Every condition at level r trains with windows of round(512 r) tokens, so each step reads the same bytes.


## Model M

| r | condition | units | window | steps | tokens ×ref | bytes read | FLOPs ×ref (range) | FLOPs per byte ×ref |
|---|---|---|---|---|---|---|---|---|
| 1 | reference: full tokenizer, one pass over the content | 25 | 512 | 4,167 | 1.00 | 100% | 1.00 (1.00–1.00) | 1.00 |
| 1 | data curve: 12.5% of the content | 25 | 512 | 521 | 0.12 | 12% | 0.12 (0.12–0.13) | 1.00 |
| 1 | data curve: 25.0% of the content | 25 | 512 | 1,042 | 0.25 | 25% | 0.25 (0.25–0.25) | 1.00 |
| 1 | data curve: 50.0% of the content | 25 | 512 | 2,083 | 0.50 | 50% | 0.50 (0.50–0.50) | 1.00 |
| 1 | identity branch: the bytes the r = 1.5 as-deployed branch reads | 25 | 512 | 2,772 | 0.67 | 67% | 0.67 (0.66–0.67) | 1.00 |
| 1 | identity branch: the bytes the r = 2.25 as-deployed branch reads | 25 | 512 | 1,848 | 0.44 | 44% | 0.44 (0.44–0.45) | 1.00 |
| 1.5 | content-matched: one pass over the content | 25 | 768 | 4,176 | 1.50 | 100% | 1.60 (1.58–1.61) | 1.60 |
| 1.5 | content-matched: one pass over the content (swap tokenizer) | 5 | 768 | 4,663 | 1.49 | 100% | 1.58 (1.58–1.58) | 1.58 |
| 2.25 | content-matched: one pass over the content | 25 | 1152 | 4,177 | 2.26 | 100% | 2.60 (2.57–2.64) | 2.60 |
| 1.5 | as deployed: the reference's token budget | 25 | 768 | 2,778 | 1.00 | 67% | 1.06 (1.06–1.06) | 1.60 |
| 2.25 | as deployed: the reference's token budget | 25 | 1152 | 1,852 | 1.00 | 44% | 1.15 (1.15–1.15) | 2.60 |

## Model S

| r | condition | units | window | steps | tokens ×ref | bytes read | FLOPs ×ref (range) | FLOPs per byte ×ref |
|---|---|---|---|---|---|---|---|---|
| 1 | reference: full tokenizer, one pass over the content | 15 | 512 | 1,389 | 1.00 | 100% | 1.00 (1.00–1.00) | 1.00 |
| 1 | data curve: 12.5% of the content | 15 | 512 | 173 | 0.12 | 12% | 0.12 (0.12–0.13) | 1.00 |
| 1 | data curve: 25.0% of the content | 15 | 512 | 347 | 0.25 | 25% | 0.25 (0.25–0.25) | 1.00 |
| 1 | data curve: 50.0% of the content | 15 | 512 | 694 | 0.50 | 50% | 0.50 (0.50–0.50) | 1.00 |
| 1 | identity branch: the bytes the r = 1.5 as-deployed branch reads | 15 | 512 | 924 | 0.67 | 67% | 0.67 (0.66–0.67) | 1.00 |
| 1 | identity branch: the bytes the r = 2.25 as-deployed branch reads | 15 | 512 | 616 | 0.44 | 44% | 0.44 (0.44–0.45) | 1.00 |
| 1.5 | content-matched: one pass over the content | 15 | 768 | 1,392 | 1.50 | 100% | 1.60 (1.58–1.61) | 1.60 |
| 1.5 | content-matched: one pass over the content (swap tokenizer) | 3 | 768 | 1,556 | 1.49 | 100% | 1.58 (1.58–1.58) | 1.58 |
| 2.25 | content-matched: one pass over the content | 15 | 1152 | 1,392 | 2.26 | 100% | 2.61 (2.58–2.65) | 2.61 |
| 1.5 | as deployed: the reference's token budget | 15 | 768 | 926 | 1.00 | 67% | 1.06 (1.06–1.06) | 1.60 |
| 2.25 | as deployed: the reference's token budget | 15 | 1152 | 617 | 1.00 | 44% | 1.16 (1.16–1.16) | 2.61 |
