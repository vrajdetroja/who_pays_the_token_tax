# Table 1: token premium vs. English on FLORES-200 devtest (2026-10-29)

Premium = tokens for the same 1,012 sentences / tokens in English; it is also the cost multiplier at any per-token price. A 128k-token window holds 128k / premium tokens' worth of English-equivalent content.

| Language | o200k_base (GPT-4o/4.1/5) | Qwen3 | Mistral Nemo (Tekken) | DeepSeek-V3 | BLOOM (2022) | XGLM (2021) | mean | 128k window holds |
|---|---|---|---|---|---|---|---|---|
| English | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 128k |
| Chinese (Simplified) | 1.25 | 1.00 | 1.40 | 0.94 | 0.91 | 1.12 | 1.11 | 116k |
| Arabic (MSA) | 1.38 | 1.63 | 1.24 | 1.62 | 1.15 | 1.29 | 1.38 | 92k |
| Basque | 1.60 | 1.87 | 1.53 | 1.74 | 1.15 | 1.34 | 1.54 | 83k |
| Russian | 1.42 | 1.75 | 1.50 | 1.58 | 2.46 | 1.20 | 1.65 | 77k |
| Hindi | 1.57 | 4.42 | 1.80 | 2.80 | 1.28 | 1.41 | 2.21 | 58k |
| Bengali | 1.70 | 5.03 | 2.05 | 2.02 | 1.18 | 1.53 | 2.25 | 57k |
| Yoruba | 2.17 | 2.69 | 2.81 | 2.83 | 1.62 | 1.93 | 2.34 | 55k |
| Thai | 1.96 | 2.55 | 2.37 | 1.78 | 4.58 | 1.19 | 2.41 | 53k |
| Tamil | 1.98 | 6.11 | 2.18 | 3.07 | 1.28 | 1.54 | 2.69 | 48k |
| Amharic | 5.78 | 4.05 | 7.43 | 5.97 | 5.00 | 1.53 | 4.96 | 26k |
| Burmese | 3.16 | 8.93 | 3.22 | 5.30 | 9.99 | 1.89 | 5.41 | 24k |

Position on the study's dose axis (tokens per byte relative to the language's own 16,384-entry monolingual tokenizer; the study's levels are 1.0, 1.5, 2.25):

| Language | BLOOM (2022) | DeepSeek-V3 | Mistral Nemo (Tekken) | Qwen3 | XGLM (2021) | o200k_base (GPT-4o/4.1/5) |
|---|---|---|---|---|---|---|
| Arabic (MSA) | 1.01 | 1.40 | 1.11 | 1.44 | 1.19 | 1.19 |
| Basque | 1.02 | 1.52 | 1.38 | 1.67 | 1.25 | 1.40 |
| Chinese (Simplified) | 0.89 | 0.90 | 1.38 | 0.98 | 1.14 | 1.19 |
| Russian | 1.91 | 1.21 | 1.19 | 1.37 | 0.98 | 1.08 |
| Tamil | 1.13 | 2.66 | 1.94 | 5.40 | 1.41 | 1.70 |

Not included (could not be loaded):

- Llama 3 (meta-llama/Llama-3.2-1B): GatedRepoError: 403 Client Error. (Request ID: Root=1-6ae3643d-76c2155903d8ba7a6b32ebb7;a234972a-be89-4959-95d3-56af
- Gemma 3 (google/gemma-3-1b-pt): GatedRepoError: 403 Client Error. (Request ID: Root=1-6ae3643e-1b5d147070e571eb045a1909;c56c1b3c-7d26-47f0-a676-4628
