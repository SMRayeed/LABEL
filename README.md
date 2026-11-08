# LABEL

**LABEL: Literature Asymmetry between Efficacy and Limits in Drug Knowledge and Language Models**
Code, data, results and paper for the W3PHIAI-27 workshop paper (AAAI 2027).

Every drug carries two kinds of fact: its efficacy (what it treats) and its
limits (who must not take it). Prose names a drug's efficacy about twice as
often as its limits; we call this the **LABEL asymmetry**. It disappears in the
FDA drug label, measured on the same facts with the same matcher. Language
models follow the prose: across five model sizes, contraindication accuracy
stays near chance while indication accuracy rises.

- Paper: [`paper/w3phiai.pdf`](paper/w3phiai.pdf), supplement: [`paper/w3phiai_supplement.pdf`](paper/w3phiai_supplement.pdf)
- Every number in the paper is in [`results/`](results/); [`results/SUMMARY.md`](results/SUMMARY.md) and [`results/EXTRAS.md`](results/EXTRAS.md) collect the tables.
- Data sources, versions, counts and licenses: [`DATA.md`](DATA.md).

## Key results

| Finding | Number | File |
|---|---|---|
| Wikipedia names indications vs. contraindications (574 drugs, 1,722 edges each) | 20.33% vs. 9.81%, within-drug OR 2.43 [2.01, 2.94] | `results/dilution_wiki.txt` |
| PubMed abstracts (balanced query) | 26.31% vs. 11.44%, OR 2.77 [2.33, 3.28] | `results/dilution_pubmed.txt` |
| Wikipedia + PubMed | 29.04% vs. 15.91%, OR 2.19 [1.86, 2.57] | `results/dilution_union.txt` |
| Whole FDA label (472 drugs, 1,416 edges each) | 23.73% vs. 22.88%, OR 1.05 [0.88, 1.25] | `results/label_coverage.txt`, `results/dilution_label_any.txt` |
| Label's own Contraindications section | names 2.61% of contraindications | `results/label_coverage.txt` |
| Closed-book accuracy, Qwen2.5 0.5B → 14B (chance 20%) | indication 31.5 → 49.8%; contraindication 22.2 → 27.2% | `results/EXTRAS.md` §2 |
| Page that names the disease, pooled 7–8B models | +11.1 (indication) vs. +14.2 (contraindication) points; difference −3.1 [−11.6, +5.1] | `results/EXTRAS.md` §1 |
| Hetionet control (treats vs. palliates) | 57.33% vs. 56.10%, OR 1.05, p = 0.70 | `results/dilution_hetionet.txt` |

## Repository layout

| Path | Contents |
|---|---|
| `run_all.sh` | Entry point for the full pipeline (download → analysis). |
| `config.sh`, `config_extend.sh` | All settings: SLURM account, models, query terms, data URLs, time limits. |
| `pipeline.sh` | The exact command behind each step. |
| `lib/common.sh` | Shared helpers: environment setup, downloads, SLURM submission. |
| `scripts/` | Analysis scripts (standard-library Python, except the model probes). |
| `tests/` | Synthetic fixture and tiny random model for the offline smoke test. |
| `derived_data/` | Inputs built by the pipeline that the analysis reads (PrimeKG relation splits, matched samples, Hetionet edges, DOID table). |
| `results/` | Every log and table in the paper; `results/csv/` holds the per-edge files (text tiers and model answers). |
| `tools/check_results.sh` | Recomputes every table from the shipped files and compares with `results/`. |
| `paper/` | LaTeX source (AAAI-27 kit), figures, figure scripts and the PDFs. |

## Check the results (5 minutes, no network, no GPU)

```bash
bash tools/check_results.sh
```

This copies `derived_data/` and `results/csv/` into `check/work/`, reruns the
whole analysis step (within-drug tests, degree strata, Wilson intervals,
bootstraps, label coverage, scale curve, retrieval effects, failure styles,
summary) and
compares each new log with `results/`. Expected output: `identical: 14
different: 0`. It needs only Python 3.9+.

## Reproduce from scratch

Requirements: Python 3.9+, network access for the download, text and extend
steps (Harvard Dataverse, GitHub, Wikipedia, NCBI E-utilities, openFDA,
Hugging Face), and SLURM for the model jobs (12 CPUs × 12 GB each; models run
in fp32 on CPU). About 3 GB of disk for data and caches and about 90 GB in
`$HF_HOME` for the models. On first use `run_all.sh` creates a CPU environment
at `$ENV_DIR` (torch 2.5.1, transformers 4.48.3).

```bash
bash run_all.sh --smoke          # offline test of every step on synthetic data (minutes)
export CONTACT="you@example.org" # e-mail for the Wikipedia User-Agent
export NCBI_API_KEY=...          # optional: faster PubMed (never commit it)
export OPENFDA_API_KEY=...       # optional: higher openFDA rate limit
bash run_all.sh                  # login node, inside tmux; submits SLURM jobs
bash run_all.sh --status
```

The run is complete when `results/SUMMARY.md` appears. Every step skips what is
already on disk; a job that hits its time limit resumes when you rerun the
same command (model answers are cached as they are written).

| Step | Runs on | What it does | Main outputs |
|---|---|---|---|
| `download` | login node | PrimeKG `kg.csv`, MONDO, Disease Ontology, Hetionet | `data/`, `results/data_versions.txt` |
| `prepare` | any | Splits PrimeKG into drug–disease relations; MONDO and DOID child→parent tables; Hetionet treats/palliates | `work/pkg_*.csv`, `work/mondo_child_parent.csv`, `work/het_*.csv` |
| `text` | login node | Wikipedia articles and redirect names, PubMed abstracts; labels each edge as named / family only / silent | `work/p1_edges_d2.csv`, `work/matched_edges*.csv`, `work/me_*.csv` |
| `extend` | login node | openFDA drug labels (up to 5 per drug, three texts kept); scale models | `work/me_label_*.csv`, `results/label_coverage.*` |
| `models` | SLURM | Closed-book probe (Qwen2.5 0.5B–14B); sibling-distractor questions with and without the page (Qwen2.5 0.5B/7B, OLMo 2 7B, Granite 3.1 8B) | `work/p1_edges_d2_<model>.csv`, `work/abstain_<model>_clean.csv` |
| `analysis` | any | All statistics, summary tables and figures | `results/` |

```bash
bash run_all.sh --from text      # start at a step
bash run_all.sh --only analysis  # one step alone
```

Wikipedia and PubMed change over time, so a fresh run measures today's text.
The per-edge files in `results/csv/` record what the paper measured.

## Where each paper result comes from

| Paper item | Script | Output |
|---|---|---|
| Coverage and within-drug odds ratios (Wikipedia, PubMed, both) | `dilution_test.py`, `w3_summary.py` | `results/dilution_{wiki,pubmed,union}.txt`, `results/SUMMARY.md` |
| Single vs. balanced PubMed query | `second_corpus.py`, `dilution_test.py` | `results/second_corpus_*.txt`, `results/dilution_pubmed_single.txt` |
| FDA label coverage and within-drug test | `fetch_labels.py`, `label_coverage.py`, `dilution_test.py` | `results/label_coverage.txt`, `results/dilution_label_{own,any,allthree}.txt` |
| Degree strata and logistic degree control | `dilution_test.py` | `results/dilution_*.txt` (test 3) |
| Scale curve and gap bootstrap over drugs | `run_probe_cpu.py`, `w3_extras.py` | `results/EXTRAS.md` §2, `results/csv/p1_edges_d2_*.csv` |
| Paired gain from scale, harder distractors | `w3_stats.py`, `abstain_probe.py` | `results/w3_stats.txt` |
| Retrieval by naming group | `w3_extras.py` | `results/EXTRAS.md` §1 |
| Failure styles; declines by page type | `w3_failure.py` | `results/w3_failure.txt` |
| Hetionet negative control | `prep_hetionet.py`, `dilution_test.py` | `results/dilution_hetionet.txt` |
| Naming levels, chance floor | `name_tiers.py`, `gonogo_p1.py` | `results/name_levels_depth{1,2}.txt` |
| Figures | `paper/build_figures.py`, `paper/drawio/` | `paper/fig_*.pdf` |

## Paper and figures

`paper/` builds with the AAAI-27 author kit:

```bash
cd paper && latexmk -pdf w3phiai.tex && latexmk -pdf w3phiai_supplement.tex
python3 build_figures.py           # data figures (matplotlib)
```

The two method figures are draw.io files (`paper/drawio/*.drawio`; open at
app.diagrams.net). `python3 drawio/build_drawio.py` regenerates them;
`npm install mxgraph playwright && node drawio/render_pdf.js drawio/fig_router.drawio`
exports a PDF. Icons are Microsoft Fluent Emoji 3D (MIT license) in `paper/icons/`.

## License

Code: MIT (`LICENSE`). Derived data follow their sources' licenses (`DATA.md`).
Wikipedia, PubMed and FDA label text is not redistributed; the pipeline fetches it.

## Citation

See `CITATION.cff`.
