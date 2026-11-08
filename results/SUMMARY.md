# W3PHIAI results summary

Recomputed from the per-edge CSV files in `work/`. Detailed logs are in `results/`.

## Coverage of indications and contraindications (drug-matched sample)

| Text searched | Indications named | Contraindications named | Within-drug OR [95% CI] | p | drugs | RBG 95% CI |
|---|---|---|---|---|---|---|
| Wikipedia | 20.33% (n=1,722) | 9.81% (n=1,722) | 2.43 [2.01, 2.94] | 1.9e-18 | 574 | [1.98, 2.98] |
| PubMed abstracts (balanced query) | 26.31% (n=1,722) | 11.44% (n=1,722) | 2.77 [2.33, 3.28] | 5.5e-29 | 574 | [2.30, 3.33] |
| Both together | 29.04% (n=1,722) | 15.91% (n=1,722) | 2.19 [1.86, 2.57] | 1.7e-20 | 574 | [1.85, 2.59] |
| PubMed abstracts (single query) | 23.34% (n=1,722) | 10.22% (n=1,722) | 2.77 [2.31, 3.32] | 1.9e-25 | 574 | [2.27, 3.37] |
| Wikipedia + PubMed (single query) | 26.66% (n=1,722) | 15.10% (n=1,722) | 2.10 [1.77, 2.48] | 3.0e-17 | 574 | [1.76, 2.49] |

## Closed-book probe (five options, chance 20%)

| Relation | Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct | Gain |
|---|---|---|---|
| indication | 31.50% (n=600) | 50.83% (n=600) | +19.33 |
| off-label use | 28.50% (n=600) | 35.00% (n=600) | +6.50 |
| contraindication | 22.17% (n=600) | 23.83% (n=600) | +1.67 |

Paired bootstrap of the gain difference: see `results/w3_stats.txt`.

## Exact-name match in Wikipedia, probe sample

| Relation | tier 0 | tiers 0-2 (named) | n |
|---|---|---|---|
| indication | 15.67% | 16.83% | 600 |
| off-label use | 9.83% | 10.50% | 600 |
| contraindication | 5.50% | 6.67% | 600 |

## Negative control: Hetionet treats vs palliates (Wikipedia)

treats named 57.33% (n=600), palliates named 56.10% (n=385); odds ratio 1.05, p = 0.7.
Within-drug MH odds ratio 2.48 [1.29, 4.75], p = 0.017, 51 drugs.

## Where the other numbers are

| Paper item | File |
|---|---|
| Within-drug tests and degree strata, Wikipedia | `results/dilution_wiki.txt` |
| Within-drug tests and degree strata, PubMed | `results/dilution_pubmed.txt` |
| Within-drug tests and degree strata, both corpora | `results/dilution_union.txt` |
| Hetionet control, full test | `results/dilution_hetionet.txt` |
| Wilson intervals, paired gain from scale, harder-distractor check | `results/w3_stats.txt` |
| Silent share per corpus, single and balanced PubMed query | `results/second_corpus_*.txt` |
| Floor and contribution of each naming level | `results/name_levels_depth*.txt` |
| MONDO table size | `results/mondo_parents.txt` |

