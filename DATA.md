# Data

All inputs are public. The pipeline downloads them (`bash run_all.sh`, steps
`download` and `text`); this repository ships only what the analysis reads
after those steps, plus every per-edge result. Text from Wikipedia, PubMed and
FDA labels is not redistributed.

## Sources and versions

| Source | Version used | Use | License |
|---|---|---|---|
| PrimeKG `kg.csv` (Chandak et al. 2023), Harvard Dataverse file 6180620 | downloaded 2026-10-28; sha256 `0a5a010f…5c69` | drug–disease edges | see the dataset page (doi:10.7910/DVN/IXA7BM) |
| MONDO disease ontology | releases/2026-10-06; sha256 `2631e8e1…ed60b` | disease families (child→parent) | CC BY 4.0 |
| Disease Ontology (DOID) | releases/2026-09-30 | Hetionet disease families | CC0 1.0 |
| Hetionet v1.0 edges and nodes | GitHub hetio/hetionet | negative control (treats vs. palliates) | CC0 1.0 (see repository) |
| English Wikipedia | fetched 2026-10/11 through the MediaWiki API | drug articles, redirect names | CC BY-SA 4.0 (not redistributed) |
| PubMed abstracts | fetched 2026-10/11 through NCBI E-utilities | second prose corpus | publisher copyright (not redistributed) |
| FDA structured product labels | openFDA drug label API, fetched 2026-11 | label text, up to 5 labels per drug | public domain (not redistributed) |

Exact hashes: `results/data_versions.txt`.

## Counts

| Item | Count | Log |
|---|---|---|
| PrimeKG rows | 8,100,498 | `results/prep_primekg.txt` |
| Indication edges (drug–disease) | 9,388 | `results/prep_primekg.txt` |
| Contraindication edges | 30,675 | `results/prep_primekg.txt` |
| Off-label use edges | 2,568 | `results/prep_primekg.txt` |
| MONDO child→parent rows / diseases with a parent | 17,801 / 11,788 | `results/mondo_parents.txt` |
| Drug-matched sample: drugs, edges per relation | 574 drugs, 1,722 indication + 1,722 contraindication (3 of each per drug) | `results/build_matched.txt` |
| Drugs with an FDA label; edges per relation in the label analysis | 472 of 574; 1,416 | `results/fetch_labels.txt`, `results/label_coverage.txt` |
| Closed-book probe sample | 600 edges per relation (indication, off-label, contraindication), 5 options | `results/build_probe_sample.txt` |
| Harder-distractor questions | 856 (indication 450, contraindication 406), sibling distractors + "Not enough information" | `results/w3_stats.txt` |
| Hetionet treats / palliates edges measured | 600 / 385 (of 755 / 390) | `results/prep_hetionet.txt`, `results/dilution_hetionet.txt` |

## Files in `derived_data/`

| File | Rows | Contents |
|---|---|---|
| `primekg/pkg_ind_d2d.csv`, `pkg_con_d2dis.csv`, `pkg_off_d2dis.csv` | 9,388 / 30,675 / 2,568 | PrimeKG drug–disease relations (PrimeKG columns) |
| `primekg/pkg_pc.csv` | 64,388 | PrimeKG disease–disease edges |
| `primekg/matched_indication.csv`, `matched_contraindication.csv` | 1,785 each | drug-matched sample before the text step |
| `primekg/matched_edges_pm.csv`, `matched_edges_pm_balanced.csv` | 3,444 | tiers in PubMed (single and balanced query) |
| `hetionet/het_all.csv`, `het_CtD.csv`, `het_CpD.csv` | 1,145 / 755 / 390 | Hetionet compound–disease edges |
| `hetionet/doid_child_parent.csv` | 103 | DOID child→parent table for the Hetionet diseases |

Per-edge results (text tiers per corpus and every model answer) are in `results/csv/`.

## File formats

**Tier files** (`*_edges*.csv`, `me_*.csv`, `p1_edges_d2*.csv`, `globi_tiers.csv`): one row per graph edge.
`tier` is how plainly the text names the target: 0 exact name, 1 synonym, 2 alternative or common name
(Wikipedia redirect), 3 only an ancestor (family: up to two ontology levels up), 4 not at all. Tiers 0-2 count
as "named". `category_only` marks tier 3, `art_len` the article length, `cooccurrence` how many articles mention
both names, `out_degree` the source's number of edges in the relation. Files with several texts add
`tier_wiki`, `tier_second` (PubMed or label) and `tier_union`.

**Closed-book probe** (`p1_edges_d2_<model>.csv`): the tier columns plus `mc_correct` (1 if the five-option
question was answered correctly), `mc_gold_logprob`, `mc_best_logprob`, `mc_margin`, `mc_n_choices`.

**Abstention questions** (`abstain_<model>_<task>.csv`): one row per question and condition. `group` is
named / category (family only) / silent; `condition` is `closed` (no page) or `passage` (with the page);
`n_options` counts all options including "Not enough information to tell", `n_sibling` how many distractors
come from the answer's own family; `pick` and `outcome` (correct / wrong / abstain) give the answer;
`p_pick`, `p_decline`, `p_true` are the normalized option probabilities. Tasks: `clean` (base prompt),
`instruct` (told to decline on a broad category), `edit_family` / `edit_remove` (counterfactual page edits).

