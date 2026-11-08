# Shared helpers for run_all.sh. Sourced, not executed.

log ()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn () { printf '[%s] WARNING: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die ()  { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

# have FILE...  -> true when every file exists and is non-empty
have () { local f; for f in "$@"; do [[ -s "$f" ]] || return 1; done; return 0; }

# py SCRIPT ARGS...  -> run a pipeline script from scripts/
py () { local s="$1"; shift; "$PY" "$SCR/$s" "$@"; }

# step_log NAME CMD...  -> run CMD from $WORK, copy its output to results/NAME.txt
step_log () {
    local name="$1"; shift
    local out="$RES/$name.txt"
    log "run  $name"
    ( cd "$WORK" && "$@" ) > >(tee "$out") 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        warn "$name exited with status $rc (log: $out)"
        return $rc
    fi
    return 0
}

# ------------------------------------------------------------------ network
has_network () {
    [[ "${SMOKE:-0}" == "1" ]] && return 1
    curl -s -o /dev/null --max-time 15 \
        "https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&format=json" \
        && return 0
    return 1
}

need_network () {
    [[ "${SMOKE:-0}" == "1" ]] && return 0
    has_network || die "this step fetches from Wikipedia, PubMed and Hugging Face, and
this machine has no outbound network. On OSC, run it on a login node:
    bash run_all.sh --from $1"
}

fetch_url () {      # fetch_url URL DEST
    local url="$1" dest="$2"
    if have "$dest"; then log "have $(basename "$dest")"; return 0; fi
    log "download $(basename "$dest")"
    curl -L --fail --retry 5 --retry-delay 10 -o "$dest.part" "$url" \
        || { rm -f "$dest.part"; die "download failed: $url"; }
    mv "$dest.part" "$dest"
}

# ------------------------------------------------------------------ python
setup_env () {
    if [[ -n "${PY_MODULE}" ]] && command -v module >/dev/null 2>&1; then
        module load "$PY_MODULE" || warn "module load $PY_MODULE failed"
    fi
    if [[ "${VIRTUAL_ENV:-}" != "" && "${VIRTUAL_ENV}" != "$ENV_DIR" ]]; then
        deactivate 2>/dev/null || true
    fi
    if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
        [[ "${SMOKE:-0}" == "1" ]] && die "no environment at $ENV_DIR"
        need_network env
        "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
            || die "$PYTHON is $("$PYTHON" -V 2>&1); transformers needs Python 3.9 or newer.
Set PY_MODULE in config.sh to a newer Python module (list them with: module spider python),
or point ENV_DIR at an existing environment."
        log "creating the Python environment at $ENV_DIR"
        "$PYTHON" -m venv "$ENV_DIR" || die "python -m venv failed"
        # shellcheck disable=SC1091
        source "$ENV_DIR/bin/activate"
        pip install --upgrade pip >/dev/null
        pip install --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION" \
            || die "torch install failed"
        pip install "transformers==$TRANSFORMERS_VERSION" huggingface_hub \
            numpy matplotlib || die "pip install failed"
    else
        # shellcheck disable=SC1091
        source "$ENV_DIR/bin/activate"
    fi
    PY="$ENV_DIR/bin/python"
    export HF_HOME
    export TOKENIZERS_PARALLELISM=false
    "$PY" - <<'PYCHK' || die "the environment at $ENV_DIR cannot import torch and transformers"
import torch, transformers
print(f"python env   torch {torch.__version__}, transformers {transformers.__version__}")
PYCHK
}

offline_models () {
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    local t="${SLURM_CPUS_PER_TASK:-$CPUS}"
    export OMP_NUM_THREADS="$t" MKL_NUM_THREADS="$t"
}

tag_of () { basename "$1"; }

fetch_models () {
    local m
    for m in "$@"; do
        [[ -d "$m" ]] && continue           # a local path, nothing to download
        log "model $m"
        "$PY" "$SCR/fetch_model.py" --model "$m" --cache "$HF_HOME" \
            || die "could not download $m"
    done
}

# ------------------------------------------------------------------ reuse
reuse_caches () {
    [[ -z "${REUSE_FROM}" ]] && return 0
    [[ -d "$REUSE_FROM" ]] || die "REUSE_FROM=$REUSE_FROM is not a directory"
    local d
    for d in wiki_cache gbif_cache pubmed_cache candidate_wiki_cache; do
        if [[ -d "$REUSE_FROM/$d" ]]; then
            log "reuse $REUSE_FROM/$d"
            mkdir -p "$WORK/$d"
            cp -rn "$REUSE_FROM/$d/." "$WORK/$d/"
        fi
    done
    if [[ -f "$REUSE_FROM/pubmed_pairs.jsonl" && ! -f "$WORK/pubmed_pairs.jsonl" ]]; then
        cp "$REUSE_FROM/pubmed_pairs.jsonl" "$WORK/"
    fi
}

# Warn when a reused PubMed cache was built with different query fragments,
# since second_corpus.py keys the balanced corpus on a hash of them.
check_pubmed_terms () {
    [[ -d "$WORK/pubmed_cache" ]] || return 0
    "$PY" - "$PUBMED_TERMS" "$WORK/pubmed_cache" <<'PYCHK'
import glob, hashlib, os, re, sys
terms = [t.strip() for t in sys.argv[1].split("|") if t.strip()]
want = hashlib.sha1("|".join(terms).encode("utf-8")).hexdigest()[:8]
seen = {}
for p in glob.glob(os.path.join(sys.argv[2], "*__q*.txt")):
    m = re.search(r"__q([0-9a-f]{8})\.txt$", p)
    if m:
        seen[m.group(1)] = seen.get(m.group(1), 0) + 1
if seen and want not in seen:
    print(f"WARNING: pubmed_cache holds a balanced corpus under query hash(es) "
          f"{', '.join(seen)}, but PUBMED_TERMS hashes to {want}. Those files "
          f"will not be used and the balanced corpus will be fetched again. "
          f"Set PUBMED_TERMS in config.sh to the fragments of the earlier run "
          f"to reuse it.", file=sys.stderr)
elif want in seen:
    print(f"pubmed_cache: balanced corpus for these terms present "
          f"({seen[want]:,} files, hash {want})")
PYCHK
}

# ------------------------------------------------------------------ slurm
# submit NAME TIME DEPENDENCY CMD  -> prints the job id
submit () {
    local name="$1" time="$2" dep="$3"; shift 3
    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
    sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
        --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" --time="$time" \
        --job-name="$name" --output="$LOGS/%x_%j.out" "${depflag[@]}" \
        --chdir="$ROOT" --wrap="$*"
}

is_small () { [[ "$1" == "$SMALL_MODEL" || "$1" == *0.5B* || -d "$1" ]]; }
