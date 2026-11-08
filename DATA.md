# 📚 LABEL data

All our inputs are public. The pipeline downloads them for you.
We ship only the files the analysis reads, plus every per-edge result.
We do not ship text from Wikipedia, PubMed or FDA labels.

## 🌐 Sources

| Source | Version | We use it for | License |
|---|---|---|---|
| 🗄️ PrimeKG `kg.csv` (Chandak et al. 2023), Harvard Dataverse file 6180620 | 2026-10-28, sha256 `0a5a010f…5c69` | drug → disease edges | see doi:10.7910/DVN/IXA7BM |
| 📁 MONDO disease ontology | 2026-10-06, sha256 `2631e8e1…ed60b` | disease families | CC BY 4.0 |
| 📁 Disease Ontology (DOID) | 2026-09-30 | Hetionet disease families | CC0 1.0 |
| 🧫 Hetionet v1.0 | GitHub hetio/hetionet | control test (treats vs. palliates) | CC0 1.0 |
| 📖 English Wikipedia | MediaWiki API, 2026-10/11 | drug pages, common names | CC BY-SA 4.0 (not shipped) |
| 🔬 PubMed abstracts | NCBI E-utilities, 2026-10/11 | second text source | publisher copyright (not shipped) |
| 🏷️ FDA drug labels | openFDA API, 2026-11 | label text, up to 5 labels per drug | public domain (not shipped) |

Exact hashes: `results/data_versions.txt`.

## 🔢 Counts

| Item | Count | Log |
|---|---:|---|
| 🗄️ PrimeKG rows | 8,100,498 | `results/prep_primekg.txt` |
| ✅ Indication edges | 9,388 | `results/prep_primekg.txt` |
| ⛔ Contraindication edges | 30,675 | `results/prep_primekg.txt` |
| 🔸 Off-label edges | 2,568 | `results/prep_primekg.txt` |
| 📁 MONDO child → parent rows / diseases with a parent | 17,801 / 11,788 | `results/mondo_parents.txt` |
| 💊 Matched sample | 574 drugs; 1,722 edges per relation (3 + 3 per drug) | `results/build_matched.txt` |
| 🏷️ Drugs with an FDA label; edges per relation | 472; 1,416 | `results/fetch_labels.txt`, `results/label_coverage.txt` |
| ❓ Closed-book questions | 600 per relation, 5 options | `results/build_probe_sample.txt` |
| 🧪 Harder questions | 856 (450 indication + 406 contraindication) | `results/w3_stats.txt` |
| 🧫 Hetionet treats / palliates measured | 600 / 385 (of 755 / 390) | `results/prep_hetionet.txt` |

## 🗂️ Files in `derived_data/`

| File | Rows | What it holds |
|---|---:|---|
| `primekg/pkg_ind_d2d.csv`, `pkg_con_d2dis.csv`, `pkg_off_d2dis.csv` | 9,388 / 30,675 / 2,568 | PrimeKG drug → disease relations |
| `primekg/pkg_pc.csv` | 64,388 | PrimeKG disease → disease edges |
| `primekg/matched_indication.csv`, `matched_contraindication.csv` | 1,785 each | matched sample before the text step |
| `primekg/matched_edges_pm.csv`, `matched_edges_pm_balanced.csv` | 3,444 | tiers in PubMed |
| `hetionet/het_all.csv`, `het_CtD.csv`, `het_CpD.csv` | 1,145 / 755 / 390 | Hetionet compound → disease edges |
| `hetionet/doid_child_parent.csv` | 103 | DOID families for the Hetionet diseases |

Per-edge results live in `results/csv/`.

## 🧾 File formats

**📖 Tier files** (`*_edges*.csv`, `me_*.csv`, `p1_edges_d2*.csv`, `globi_tiers.csv`). One row per graph edge. The `tier` column says how plainly the text names the target:

| Tier | Meaning | Counts as |
|:---:|---|---|
| 0 | exact name | 🟢 named |
| 1 | synonym | 🟢 named |
| 2 | common name (a Wikipedia redirect) | 🟢 named |
| 3 | only a parent, up to two levels up | 🟡 family only |
| 4 | not at all | ⚪ silent |

Other columns: `category_only` (tier 3), `art_len` (article length), `cooccurrence` (articles that mention both names), `out_degree` (the source's edges in the relation). Files with two texts add `tier_wiki`, `tier_second` and `tier_union`.

**❓ Closed-book probe** (`p1_edges_d2_<model>.csv`). The tier columns, plus `mc_correct` (1 if the model got the 5-option question right), `mc_gold_logprob`, `mc_best_logprob`, `mc_margin`, `mc_n_choices`.

**🤖 Abstention questions** (`abstain_<model>_<task>.csv`). One row per question and condition.

| Column | Meaning |
|---|---|
| `group` | named / category (family only) / silent |
| `condition` | `closed` (no page) or `passage` (with the page) |
| `n_options`, `n_sibling` | all options, and how many wrong options come from the answer's family |
| `pick`, `outcome` | the model's pick, and correct / wrong / abstain |
| `p_pick`, `p_decline`, `p_true` | option probabilities |

Tasks: `clean` (base prompt), `instruct` (told to decline on a broad category), `edit_family` / `edit_remove` (page edits).
