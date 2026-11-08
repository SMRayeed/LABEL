# Settings for run_all.sh. Every value can also be set in the environment,
# e.g.  ACCOUNT=PAS2136 bash run_all.sh

# ---------------------------------------------------------------- SLURM
ACCOUNT="${ACCOUNT:-PAS2136}"
PARTITION="${PARTITION:-cpu}"
CPUS="${CPUS:-12}"
MEM_PER_CPU="${MEM_PER_CPU:-12G}"
TIME_SMALL="${TIME_SMALL:-04:00:00}"     # 0.5B model jobs
TIME_LARGE="${TIME_LARGE:-12:00:00}"     # 7-8B model jobs; resubmit to resume
TIME_ANALYSIS="${TIME_ANALYSIS:-01:00:00}"

# ---------------------------------------------------------------- Python
# An existing CPU environment is reused when present. Otherwise run_all.sh
# creates it with the versions below (CPU build of torch).
ENV_DIR="${ENV_DIR:-$HOME/cpu_probe_env}"
PYTHON="${PYTHON:-python3}"
PY_MODULE="${PY_MODULE:-}"               # e.g. python/3.12 to `module load` first
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.48.3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
HF_HOME="${HF_HOME:-$HOME/hf_cache}"

# ---------------------------------------------------------------- models
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# closed-book probe (Table 2) uses the two Qwen sizes
PROBE_MODELS="${PROBE_MODELS:-$SMALL_MODEL $LARGE_MODEL}"
# harder-distractor check (sibling options) uses four models
ABSTAIN_MODELS="${ABSTAIN_MODELS:-$SMALL_MODEL $LARGE_MODEL allenai/OLMo-2-1124-7B-Instruct ibm-granite/granite-3.1-8b-instruct}"

# ---------------------------------------------------------------- web access
# Wikipedia asks scripts to identify themselves; an e-mail or URL here is
# added to the User-Agent. An NCBI API key raises the PubMed rate limit
# from 3 to 10 requests per second.
CONTACT="${CONTACT:-}"
NCBI_API_KEY="${NCBI_API_KEY:-}"

# The two PubMed query fragments for the balanced corpus, separated by "|".
# One search per fragment, 20 abstracts each (40 per drug in total).
PUBMED_TERMS="${PUBMED_TERMS:-treat*[tiab] OR therap*[tiab] OR efficac*[tiab]|contraindicat*[tiab] OR adverse[tiab] OR toxicit*[tiab] OR \"side effect\"[tiab] OR \"side effects\"[tiab] OR induced[tiab]}"

# ---------------------------------------------------------------- reuse
# A directory holding earlier caches (wiki_cache, gbif_cache, pubmed_cache).
# When set, they are copied into work/ before any fetching, so the text is
# the same text the paper was measured on and nothing is downloaded twice.
REUSE_FROM="${REUSE_FROM:-}"

# ---------------------------------------------------------------- data sources
PRIMEKG_URL="${PRIMEKG_URL:-https://dataverse.harvard.edu/api/access/datafile/6180620}"
MONDO_URL="${MONDO_URL:-https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo}"
DOID_URL="${DOID_URL:-https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo}"
HETIO_EDGES_URL="${HETIO_EDGES_URL:-https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz}"
HETIO_NODES_URL="${HETIO_NODES_URL:-https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv}"

# ---------------------------------------------------------------- sampling
SEED="${SEED:-0}"
PROBE_N="${PROBE_N:-600}"          # random probe sample, edges per relation
MATCH_MIN="${MATCH_MIN:-3}"        # drugs need at least this many of each relation
MATCH_PER_DRUG="${MATCH_PER_DRUG:-3}"
PER_BAND="${PER_BAND:-150}"        # abstention questions per group per relation
