# PREREGISTRATION.md — The tokenization tax: fertility or accounting?

Version 1 · committed 2026-10-29, before any model of this study was trained · amendments
appended at the end, dated

## 1. Question

Everyone assumes high tokenizer fertility makes a language worse-modeled. That has not been
tested within a language, at fixed text, with training budgets and context windows counted
both in tokens and in bytes. How much of the fertility quality penalty disappears under
content-denominated accounting?

## 2. Hypotheses (direction and size)

All effects are bits per UTF-8 byte (BPB) on held-out FineWeb-2 text, relative to the same
language's full-vocabulary (r = 1) reference. D(budget, context) is that penalty with the
training budget and the evaluation context each counted in tokens (tok) or bytes (byte).

- **H1 (primary).** At r = 2.25, model M, pooled over languages, the share of the as-deployed
  penalty removed by content accounting, S = 1 − D(byte, byte) / D(tok, tok), is at least 0.5.
  Prediction: S = 0.50, plausible range 0.25–0.80. My point prediction sits on the kill line on
  purpose: my own earlier vocabulary ablation (micro-lm-ablations §4.6: a 2,048-entry vocabulary,
  ~1.44× finer than 16,384, lost 0.065 BPB at matched FLOPs and parameters while reading less text
  but carrying three more layers) suggests a content-matched residual of a few hundredths of a bit
  per byte already near r = 1.5.
- **H2.** D(tok, tok) at r = 2.25 lies in 0.04–0.20 BPB; D(byte, byte) lies in 0.01–0.10 BPB.
- **H3.** The budget channel equals its matched prediction within ±30% (ratio 0.70–1.30). The
  matched prediction is the BPB cost, at r = 1, of reading only the bytes the as-deployed branch
  reads (§5, identity branches); equality means finer segmentation neither helps nor hurts per
  byte read. The fitted form, b · ln r with b the data-curve slope, is reported beside it.
- **H4 (scale).** D(byte, byte) is smaller for model M than for model S.
- **H5 (dose).** In the as-deployed regime the penalty rises monotonically with ln r:
  D(tok, tok) at 2.25 exceeds D(tok, tok) at 1.5, which exceeds 0.

## 3. Both abstracts

**A (accounting dominates).** Multilingual tokenizers split some languages into far more tokens
than others, and this premium is widely blamed for worse model quality. We ask how much of the
quality penalty comes from fertility itself and how much from counting training budgets and
context windows in tokens. We train [N] small language models on identical text in five
typologically diverse languages, varying only the tokenizer's vocabulary allocation, and
evaluate bits-per-byte under token- and content-denominated accounting. At 2.25× fertility,
[X]% (95% CI [a, b]) of the penalty disappears when budgets and contexts are counted in
content; the residual, [Y] bits/byte, is smaller than the effect of discarding 10% of the
training text. Multilingual data mixtures, context limits and evaluation budgets should be
stated in content units, not tokens.

**B (residual dominates).** Same first three sentences. Even with identical text, identical
byte context and 2.25× more compute per byte, finer segmentation costs [Y] bits/byte (95% CI
[a, b]), equivalent to discarding [Z]% of the training data; content accounting removes only
[X]% of the penalty. Equalizing data cannot close this gap: vocabulary allocation is a
first-order fairness lever, and audits should report it alongside data share.

## 4. Operationalization

I am interested in whether a language's higher token count per unit of content makes models
worse at that language, beyond what it costs in budget. I operationalize this as the
bits-per-byte penalty of models trained on identical text with tokenizers allocating 1×, 1.5×
and 2.25× tokens per byte, evaluated under token- and byte-denominated training budgets and
context windows. This does not capture downstream task competence or long-context use in large
models, and the report discusses how that bounds the conclusions.

| Construct | Operationalization | Gap |
|---|---|---|
| Fertility (dose) | r = tokens/byte under the dose tokenizer ÷ tokens/byte under the language's full 16,384-entry tokenizer, measured on held-out text. Levels 1.0, 1.5, 2.25 (±3%). | Allocation-driven fertility, not language-intrinsic fertility. Tokens per whitespace word is used nowhere. |
| Model quality | BPB on held-out FineWeb-2 (primary) and FLORES-200 devtest (secondary, bits per sentence), canonical tokenization. Every target token is scored exactly once, so the byte denominator is identical across levels and protocols. | Next-byte prediction, not task competence. Tiny models. Canonical scoring slightly overstates BPB. |
| Accounting regime | Training budget in tokens (the reference's token count T) or bytes (one pass over the full content); evaluation context of 512 tokens (ctx-tok) or round(512 r) tokens (ctx-byte). | Training windows are byte-matched in every condition, so the as-deployed condition is slightly generous: the measured tax is a lower bound. |
| Effective context | Bytes of content in a fixed token window (Table 1); D(·, tok) − D(·, byte) in the models. | Long-context use at scale is not measured. |
| Cost | Token premium vs. English on FLORES-200 devtest (Table 1). | Ignores reasoning tokens, caching discounts and per-character pricing. |

## 5. Design

**Languages.** Basque (eus_Latn), Russian (rus_Cyrl), Arabic (arb_Arab), Chinese (cmn_Hani in
FineWeb-2; FLORES zho_Hans), Tamil (tam_Taml). FineWeb-2 holds about 4.6 GB of Basque against a
0.64 GB budget, so the Indonesian fallback is not needed. FineWeb-2's Chinese is kept as sampled:
Mandarin in Han script, ~86% Simplified by bytes (13.6% of the content bytes and 13.3% of the
held-out bytes are Traditional, by a common-character heuristic); FLORES zho_Hans is Simplified.

**Data.** FineWeb-2, sampled by random parquet row group across up to 40 randomly chosen files
per language (row groups are clustered by crawl, so a file's first documents are one era of the
web). Per language, shuffled with a fixed seed:

- training content: byte premium × 600 MB, with the byte-premium tool's values (Arnett, Chang &
  Bergen 2024: NLLB, else FLORES, else Bible; Mandarin uses the tool's zho_hans NLLB row):
  Basque 1.060, Russian 1.823, Arabic 1.465, Chinese 0.936, Tamil 2.729;
- tokenizer sample: 200 MB, disjoint from the content; a second disjoint 200 MB for the swap
  tokenizer (Russian);
- from FineWeb-2's test split: 2 MB × premium for the dose search and a disjoint 2 MB × premium
  held-out evaluation set.

Every set is decontaminated against FLORES-200 dev + devtest (13-grams of words; of characters
for Chinese; NFKC, lowercased, punctuation removed), and test documents identical to a sampled
training document are dropped. Model S reads the first third of its language's content.

**Tokenizers.** Byte-level BPE, 16,384 entries (256 bytes + 16,127 merges + one separator), one
per language, trained with Hugging Face `tokenizers` and encoded identically by the from-scratch
reference encoder (tested token for token). Pre-tokenization is GPT-4's pattern with combining
marks (\p{M}) kept with letters, as in o200k_base; GPT-4's pattern splits Tamil words at every
vowel sign. Dose levels keep the first k merges (binary search on the dose set, verified on the
held-out set within ±3%). Rows a level can never produce stay in the embedding table, so every
level has identical parameters and identical ids for shared tokens, and they are **masked out of
the softmax in training and evaluation**: no probability is spent on impossible tokens, which
would otherwise cost more at higher r and in shorter training.

**Models.** Pre-norm decoder, rotary positions, GELU MLP, tied embeddings, no biases, bf16
autocast. M = 6 layers × 384 wide × 6 heads (10.6M non-embedding + 6.3M embedding parameters);
S = 4 × 256 × 4 (3.1M + 4.2M).

**Training.** AdamW (β = 0.9, 0.95; weight decay 0.1; clip 1.0), learning rate from the scout
(1e-3 / 2e-3 / 4e-3; default 2e-3). Warmup–stable–decay: warmup over 2% of the run's steps,
constant, linear cooldown to zero over the final 20% of every endpoint. 64 windows per step;
window = round(512 r) tokens (512 / 768 / 1,152), so every step reads the same bytes on average.
Common random numbers: within a seed, every level gets the same initialization and reads the
same documents in the same order (documents are permuted by the seed, then cut into windows).
Seeds 0, 1, 2.

**Endpoints per (language, model, seed).**

| Run | Trains on | Endpoints |
|---|---|---|
| r = 1.0 (reference) | one pass over the content (T tokens) | final (both regimes); data-curve cooldown branches at 12.5 / 25 / 50%; identity branches reading exactly the share of the pass each level's as-deployed branch reads |
| r = 1.5 | one pass over the content | content-matched final; as-deployed cooldown branch at T tokens |
| r = 2.25 | one pass over the content | content-matched final; as-deployed cooldown branch at T tokens |
| r = 1.5 swap (Russian) | one pass, second tokenizer | content-matched final (negative control 1) |

A cooldown branch with B steps starts from its parent's checkpoint at step B − round(0.2 B),
inside the parent's constant phase. Every endpoint is evaluated under ctx-tok and ctx-byte with
sliding windows advanced by half a window.

**Controls.**

- Negative 1: the Russian swap tokenizer (disjoint sample, same r ± 2%) changes BPB by no more
  than seed noise (its 95% interval covers 0). Measured: it matches r within 0.2% (1.4975 vs.
  1.5002) and shares 90% of its merged tokens with the main tokenizer's level (1,712 of 1,899), so
  it tests the marginal tenth of the merges: a pass rules out sensitivity to *which* frequent-
  enough merges are kept, not to radically different segmentations at the same r.
- Negative 2: the main protocol at stride ¼ window agrees with stride ½ within 0.001 BPB for
  every evaluated endpoint.
- Positive 1: the data curve (12.5 / 25 / 50 / 100%) falls monotonically in every unit; its
  slope b sets the SESOI and the exchange rate.
- Positive 2: replacing each window's unscored left context with distant text raises BPB. If the
  pooled rise is not above the SESOI, the context channel is reported as uninformative and
  dropped from the claims.

## 6. Power

SESOI = b · ln(1/0.9) ≈ 0.105 b: the BPB cost of discarding 10% of the training text. With
b ≈ 0.07, SESOI ≈ 0.0074 BPB.

Power by simulation, through the same pooling code the analysis runs
(`python -m recount.analysis.power`; 1,000 replicates per cell; true D(tok, tok) = 0.08):

| σΔ | τ | seeds | TOST passes when residual = 0 | H1 falsified when S = 0.3 | H1 falsified when S = 0.5 (size) | coverage |
|---|---|---|---|---|---|---|
| 0.002 | 0.004 | 3 | 0.88 | 1.00 | 0.03 | 0.94 |
| 0.004 | 0.004 | 3 | 0.80 | 1.00 | 0.03 | 0.94 |
| 0.004 | 0.004 | 5 | 0.85 | 1.00 | 0.03 | 0.96 |
| 0.004 | 0.008 | 3 | 0.21 | 0.96 | 0.02 | 0.94 |
| 0.004 | 0.008 | 5 | 0.21 | 0.95 | 0.02 | 0.96 |

H1 is well powered everywhere. The equivalence claim ("no residual") is limited by the spread
between languages (τ), not by seeds: going from 3 to 5 seeds barely moves it, while 8 languages
instead of 5 raise it from 0.80 to 0.98 at τ = 0.004 and from 0.21 to 0.46 at τ = 0.008. σΔ and
τ are measured by the scout and the table is recomputed in an amendment before the main grid.
Prior expectation for σΔ: in micro-lm-ablations the seed SD at this model size was 0.0036
nats/token ≈ 0.0013 BPB (78% of it GPU-kernel non-determinism), so σΔ ≈ √2 × 0.0013 ≈ 0.002 BPB.

## 7. Analysis plan

Per (language, seed, level): BPB in four cells (R-tok / R-byte × ctx-tok / ctx-byte), minus the
reference. Decomposition: Shapley average over the two orderings of {budget, context}, so
budget channel + context channel + content-matched residual = D(tok, tok) exactly (asserted in
code).

**Pooling.** Random-effects meta-analysis over languages, DerSimonian–Laird τ², Hartung–Knapp–
Sidik–Jonkman interval (df = languages − 1). Each language contributes its mean over seeds;
within-language variances are pooled across languages, because a 3-seed variance rests on 2
degrees of freedom. With equal seed counts this interval equals the one-sample t-interval of the
language means. Per-language variances and the Knapp–Hartung "modified" interval are
sensitivity analyses. Per-language results: mean over seeds, t-interval (df = seeds − 1),
descriptive only.

**H1 (the one confirmatory test).** L = ½ D(tok, tok) − D(byte, byte) per (language, seed),
pooled as above. S ≥ 0.5 ⟺ L ≥ 0, so **H1 is falsified iff the pooled 95% interval for L lies
entirely below 0**, and supported iff it lies entirely above 0. The interval for S is the set of
s whose contrast (1 − s) D(tok, tok) − D(byte, byte) is not rejected (Fieller's construction
with HKSJ pooling, primary); the parametric bootstrap (10,000 draws from a bivariate t at the
pooled estimates) is a sensitivity analysis.

**Residual ≈ 0.** Two one-sided tests against ±SESOI at α = 0.05 (the pooled 90% interval for
D(byte, byte) inside ±SESOI).

**Secondary (intervals, no corrections claimed).** H2: pooled 95% intervals for D(tok, tok) and
D(byte, byte). H3: pooled ratio of the budget channel to its matched prediction, and the pooled
interval of their difference; the fitted form φ_budget ÷ (b · ln r) with the measured r beside
it. H4: pooled difference of D(byte, byte) between M and S (unpaired). H5: pooled interval of
D(tok, tok)(2.25) − D(tok, tok)(1.5). Primary evaluation set: held-out FineWeb-2; FLORES-200 is
secondary and analyzed identically in bits per sentence.

**Diagnostics reported, not tested.** FLOPs per condition; probability on masked rows (zero by
construction, as a check); the canonical-tokenization gap on 200 sentences per level (Chirkova
et al. 2023).

## 8. Falsification condition

H1 is false if the pooled 95% interval for L = ½ D(tok, tok) − D(byte, byte) lies entirely
below 0, i.e. the content-matched residual is more than half of the as-deployed penalty.

## 9. Stopping rule

Exactly 3 seeds per (language, level, size). No seeds added after any level contrast has been
computed. The scout seals every level contrast's mean (`analyze --seal`: numbers in git-ignored
`sealed/`, SHA-256 in `results/seals/`); I open it only after the main analysis.

Any runs after the main grid (more languages, seeds, dose levels, or model sizes) are
exploratory: preceded by a dated amendment saying what I had seen, reported separately, and
never part of the confirmatory analysis above.

## 10. What I expect to be surprised by

- The context channel is near zero, because 17M-parameter models use little context.
- Chinese behaves unlike Tamil, the other 3-byte script: Chinese's 2.25× level splits characters
  into byte fragments, while Tamil's is near character level.
- The content-matched residual is larger than predicted: fine segmentation costs more per byte
  than the extra compute per byte buys back.
- The budget channel misses its matched prediction by more than 30%: finer tokens extract more
  (or less) from each byte read than coarse ones.

## Decisions made at registration

Taken before any model of this study was trained, with the evidence then in hand:

1. **Scope: as designed.** Considered and declined: three more languages (what the power
   simulation favors for the equivalence claim), five seeds, a larger third model, a 3.375× dose
   rung (Table 1 places Tamil at 2.7× under DeepSeek-V3 and 5.4× under Qwen3, beyond the top
   dose).
2. **Interval for S:** Fieller–HKSJ test inversion primary; bootstrap as sensitivity.
3. **Within-language variance:** pooled across languages; per-language as sensitivity.
4. **Identity branches: added** (about +4% compute), matched to each as-deployed branch's bytes.
5. **Byte premiums:** the tool's NLLB-first values. FLORES-200-direct premiums are 8–17% higher
   for Tamil, Russian and Arabic; within-language contrasts do not depend on this.
6. **Unreachable vocabulary rows: masked** in training and evaluation. A CPU check with a tiny,
   undertrained model on the real Basque shards left 14–24% of the probability on them at the
   content-matched endpoints and 39–84% at the as-deployed branches, which would have biased S
   toward "accounting".
7. **Chinese script mix: kept** (above).

## Amendments

<date> — what changed — why — what I had seen when I changed it
