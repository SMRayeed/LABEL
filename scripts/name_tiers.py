#!/usr/bin/env python3
"""
Is the trace failure a NOMENCLATURE problem or a GRANULARITY problem?

Answers it as a number, per relation, by widening what counts as a match in
four tiers and measuring how much each widening buys:

    T0  the target's accepted scientific name only          (the current measure)
    T1  + its scientific synonyms                           (still 1-to-1)
    T2  + its English common names                          <- NOMENCLATURE
    T3  + the names of its ancestors, scientific and common <- GRANULARITY

    nomenclature lift = T2 - T0      (floor-corrected)
    granularity  lift = T3 - T2      (floor-corrected)

If the nomenclature lift is small and the granularity lift is large, the
failures are not a naming problem and a name table does not fix them. If the
nomenclature lift is large, a name table does most of the work and the
resolution claim has to shrink.

WHY THE FLOOR IS NOT OPTIONAL HERE. Widening the match set makes accidental
matches more likely, and T2/T3 add ordinary words -- "rodent", "mammals",
"respiratory disease". A raw lift at T3 therefore overstates, sometimes
grossly. So the random-partner floor is recomputed AT EVERY TIER against
entities the knowledge base says are not connected, and every lift reported is
the difference of floor-corrected rates.

ENTITY TYPES MUST BE HELD CONSTANT. Compare only relations that connect the
same two entity pools, so the naming convention is identical across them:
GloBI taxon->taxon, PrimeKG drug->disease, Hetionet compound->disease. A
spread across such relations is a statement about relations and not about
names. Comparing across entity types (Hetionet PCiC against CcSE, say)
reintroduces exactly the confound this script exists to remove.

NETWORK MANNERS. Requests are batched (50 titles per redirect query, 8 per
article query), paced by a token budget, and backed off on 429/503 with
Retry-After respected. A FAILED fetch is never written to the cache -- only a
successful fetch that found nothing. That distinction matters: caching a
failure silently converts a rate-limit error into a measured zero.

Usage:
    # always look first
    python3 name_tiers.py --edges /path/to/dir --inspect
    python3 name_tiers.py --edges kg.csv --col-relation display_relation --inspect

    # PrimeKG: three single-relation extracts, same entity types
    python3 name_tiers.py --edges pkg_ind_d2d.csv pkg_con_d2dis.csv pkg_off_d2dis.csv \\
        --names redirects --any-names --n 600 --ancestors pkg_pc.csv \\
        --csv primekg_tiers.csv

    # GloBI: taxa, with GBIF for synonyms, common names and the clade ladder
    python3 name_tiers.py --edges interactions.parquet --n 600 \\
        --relations "eats,preysOn,parasiteOf,pollinates,visitsFlowersOf,hasHost" \\
        --csv globi_tiers.csv

Stdlib only, except that reading .parquet needs pyarrow.
"""

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

WIKI_API = "https://en.wikipedia.org/w/api.php"
GBIF = "https://api.gbif.org/v1"

# Single ordinary words are screened out of T0/T1 targets, as the original
# measurement does. They are NOT screened out of T2/T3 -- admitting them is the
# point of those tiers -- which is why those tiers carry their own floor.
GENERIC = {
    "water", "wood", "soil", "sand", "rock", "grass", "moss", "algae",
    "bacteria", "fungi", "plant", "animal", "insect", "worm", "fish",
    "bird", "mammal", "tree", "leaf", "seed", "fruit", "flower", "nectar",
    "pollen", "detritus", "carrion", "litter", "mud", "ice", "salt",
}

SUGGESTED_GLOBI = [
    "eats", "preysOn", "parasiteOf", "pathogenOf",
    "ectoparasiteOf", "endoparasiteOf", "pollinates",
    "visitsFlowersOf", "hasHost", "hasVector",
]


# ---------------------------------------------------------------- transport

class Fetcher:
    """Paced HTTP with real backoff.

    get() returns the parsed body on success, or None when the request could
    not be completed. Callers must treat None as "unknown", never as "empty",
    and must not cache it."""

    def __init__(self, rps=4.0, contact="", tries=6, timeout=90):
        self.gap = 1.0 / max(0.2, rps)
        self.tries = tries
        self.timeout = timeout
        self.last = 0.0
        self.penalty = 0.0          # grows on 429, decays on success
        self.n_req = 0
        self.n_throttled = 0
        self.n_failed = 0
        ua = "NameTierStudy/2.0 (research script; batched; stdlib urllib)"
        if contact:
            ua += f" ({contact})"
        self.headers = {"User-Agent": ua, "Accept": "application/json",
                        "Accept-Encoding": "gzip"}

    def _pace(self):
        want = self.gap + self.penalty
        dt = time.time() - self.last
        if dt < want:
            time.sleep(want - dt)
        self.last = time.time()

    def _slow_down(self, note):
        self.n_throttled += 1
        self.penalty = min(8.0, self.penalty * 1.6 + 0.4)
        print(f"    throttled ({note}); pacing +{self.penalty:.1f}s/request",
              file=sys.stderr)

    def get(self, url):
        for attempt in range(self.tries):
            self._pace()
            self.n_req += 1
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    body = json.loads(raw, strict=False)
                # the API can report overload in a 200 body
                err = body.get("error", {}) if isinstance(body, dict) else {}
                if err.get("code") in ("maxlag", "readonly"):
                    self._slow_down(err.get("code"))
                    time.sleep(min(60.0, 5.0 * (2 ** attempt)))
                    continue
                self.penalty *= 0.85
                return body
            except urllib.error.HTTPError as e:
                if e.code in (429, 503, 502, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = min(120.0, 6.0 * (2 ** attempt))
                    if ra and str(ra).strip().isdigit():
                        wait = max(wait, min(180.0, float(ra)))
                    self._slow_down(f"HTTP {e.code}")
                    time.sleep(wait)
                    continue
                if e.code in (400, 404):
                    self.n_failed += 1
                    return None
                time.sleep(min(30.0, 3.0 * (2 ** attempt)))
            except Exception as e:                    # noqa: BLE001
                if attempt == self.tries - 1:
                    print(f"    request failed: {str(e)[:80]}", file=sys.stderr)
                time.sleep(min(30.0, 3.0 * (2 ** attempt)))
        self.n_failed += 1
        return None

    def report(self):
        return (f"{self.n_req:,} requests, {self.n_throttled} throttle events, "
                f"{self.n_failed} unrecoverable")


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ------------------------------------------------------------------- caches

def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.strip())[:150]


#: characters that cannot go in a filename. Taxon names contain some of
#: them -- "Bombus terrestris/lucorum" is a real GloBI aggregate -- and a
#: slash would otherwise be read as a directory separator.
UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def art_path(cache, name):
    """Cache path for an article. Spaces become underscores, as before, so
    an existing cache keeps working for every name without a special
    character in it."""
    safe = UNSAFE.sub("-", name.strip().replace(" ", "_"))
    return os.path.join(cache, safe[:180] + ".txt")


def read_article(cache, name):
    """Cached article text, or None when not cached."""
    p = art_path(cache, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_article(cache, name, text):
    os.makedirs(cache, exist_ok=True)
    with open(art_path(cache, name), "w", encoding="utf-8") as fh:
        fh.write(text)


def name_path(cache, kind, key):
    return os.path.join(cache, f"{kind}__{_safe(str(key))}.json")


def read_names(cache, kind, key):
    p = name_path(cache, kind, key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_names(cache, kind, key, obj):
    os.makedirs(cache, exist_ok=True)
    tmp = name_path(cache, kind, key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, name_path(cache, kind, key))


EMPTY = {"synonyms": [], "vernacular": [], "ancestors": []}


# --------------------------------------------------- wikipedia, in batches

def fetch_articles(fx, names, cache, batch=8):
    """Cache the wikitext of each name's article. Missing pages cache as ''.

    Batched, and the normalized/redirects maps are followed back so each
    article lands under the name that was asked for."""
    todo = [n for n in dict.fromkeys(names) if read_article(cache, n) is None]
    if not todo:
        return 0
    done = 0
    for group in chunks(todo, batch):
        q = urllib.parse.urlencode({
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "formatversion": "2",
            "redirects": "1", "maxlag": "5",
            "titles": "|".join(group)})
        body = fx.get(f"{WIKI_API}?{q}")
        if body is None:
            continue                      # failure: leave uncached, try later
        query = body.get("query", {}) or {}
        # asked-for title -> title as the API ended up calling it
        fwd = {}
        for m in (query.get("normalized") or []):
            fwd[m.get("from")] = m.get("to")
        for m in (query.get("redirects") or []):
            for k, v in list(fwd.items()):
                if v == m.get("from"):
                    fwd[k] = m.get("to")
            fwd.setdefault(m.get("from"), m.get("to"))
        text_by_title = {}
        for page in (query.get("pages") or []):
            title = page.get("title")
            if page.get("missing"):
                text_by_title[title] = ""
                continue
            txt = ""
            for rev in (page.get("revisions") or []):
                txt = (rev.get("slots", {}).get("main", {}) or {}).get("content") or ""
            text_by_title[title] = txt
        for asked in group:
            final = fwd.get(asked, asked)
            if final in text_by_title:
                write_article(cache, asked, text_by_title[final])
                done += 1
            elif asked in text_by_title:
                write_article(cache, asked, text_by_title[asked])
                done += 1
        print(f"    articles {done}/{len(todo)}", end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    return done


def fetch_redirect_names(fx, names, cache, batch=50):
    """Cache the set of titles redirecting to each entity's article.

    Domain-independent, so this is the name source for PrimeKG and Hetionet
    where GBIF does not apply: the redirects to a drug or disease article are
    its synonym set as the encyclopedia records it -- brand names, salt forms,
    abbreviations, older clinical names.

    One caveat for the write-up: redirects pool scientific synonyms and
    ordinary names together, so T1 equals T2 in this mode and only the
    combined nomenclature lift is identified. That is the quantity the
    question turns on, so nothing needed is lost."""
    todo = [n for n in dict.fromkeys(names)
            if read_names(cache, "redir", n) is None]
    if not todo:
        return 0
    done = 0
    for group in chunks(todo, batch):
        acc = defaultdict(list)
        cont, guard = {}, 0
        ok = True
        while guard < 40:
            guard += 1
            params = {"action": "query", "prop": "redirects",
                      "titles": "|".join(group), "rdlimit": "max",
                      "rdnamespace": "0", "format": "json",
                      "formatversion": "2", "redirects": "1", "maxlag": "5"}
            params.update(cont)
            body = fx.get(f"{WIKI_API}?{urllib.parse.urlencode(params)}")
            if body is None:
                ok = False
                break
            query = body.get("query", {}) or {}
            fwd = {}
            for m in (query.get("normalized") or []):
                fwd[m.get("from")] = m.get("to")
            for m in (query.get("redirects") or []):
                fwd.setdefault(m.get("from"), m.get("to"))
            for page in (query.get("pages") or []):
                title = page.get("title")
                for rd in (page.get("redirects") or []):
                    t = (rd.get("title") or "").strip()
                    if t and "(" not in t:
                        acc[title].append(t)
                acc.setdefault(title, [])
            cont = body.get("continue") or {}
            if not cont:
                break
        if not ok:
            continue                      # failure: leave uncached
        for asked in group:
            final = fwd.get(asked, asked)
            vern = acc.get(final, acc.get(asked, []))
            write_names(cache, "redir", asked,
                        {"synonyms": [], "vernacular": sorted(set(vern)),
                         "ancestors": []})
            done += 1
        print(f"    name sets {done}/{len(todo)}", end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    return done


# -------------------------------------------------------- gbif, for taxa

RANKS = ["genus", "family", "order", "class", "phylum", "kingdom"]


def fetch_gbif_names(fx, names, cache, ancestor_vernaculars=True):
    """Cache synonyms, English common names and the clade ladder per taxon.

    GBIF has no batch endpoint, so this is one taxon at a time -- but ancestor
    vernaculars are cached by GBIF key, and ancestors repeat heavily across
    species, so the per-rank lookups collapse after the first few hundred
    taxa."""
    todo = [n for n in dict.fromkeys(names)
            if read_names(cache, "gbif", n) is None]
    if not todo:
        return 0
    done = 0
    for name in todo:
        m = fx.get(f"{GBIF}/species/match?"
                   + urllib.parse.urlencode({"name": name, "strict": "false"}))
        if m is None:
            continue                      # failure: leave uncached
        out = {"synonyms": [], "vernacular": [], "ancestors": []}
        key = m.get("usageKey")
        if key:
            syn = fx.get(f"{GBIF}/species/{key}/synonyms?limit=100")
            for r in ((syn or {}).get("results") or []):
                for field in ("canonicalName", "scientificName"):
                    if r.get(field):
                        out["synonyms"].append(r[field])
            ver = fx.get(f"{GBIF}/species/{key}/vernacularNames?limit=100")
            for r in ((ver or {}).get("results") or []):
                if r.get("language") in (None, "eng", "en") and r.get("vernacularName"):
                    out["vernacular"].append(r["vernacularName"])
            for rank in RANKS:
                sci = m.get(rank)
                if not sci:
                    continue
                anc = {"rank": rank, "scientific": sci, "vernacular": []}
                k2 = m.get(rank + "Key")
                if k2 and ancestor_vernaculars:
                    hit = read_names(cache, "gbifkey", k2)
                    if hit is None:
                        v2 = fx.get(f"{GBIF}/species/{k2}/vernacularNames?limit=60")
                        if v2 is not None:
                            vs = sorted({r["vernacularName"] for r in (v2.get("results") or [])
                                         if r.get("language") in (None, "eng", "en")
                                         and r.get("vernacularName")})
                            write_names(cache, "gbifkey", k2, {"vernacular": vs})
                            hit = {"vernacular": vs}
                    if hit:
                        anc["vernacular"] = hit.get("vernacular", [])
                out["ancestors"].append(anc)
        out["synonyms"] = sorted(set(out["synonyms"]))
        out["vernacular"] = sorted(set(out["vernacular"]))
        write_names(cache, "gbif", name, out)
        done += 1
        if done % 25 == 0:
            print(f"    taxon names {done}/{len(todo)}", end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    return done


# ------------------------------------------------- ancestors from a table

def load_ancestors(path, cols_arg="", flip=False):
    """child -> [parent, ...] for T3 outside biology.

    The file is read with the same column detection as the edges, because a
    PrimeKG extract is not a bare two-column table: it carries the relation
    and type columns too, and taking the first two columns positionally picks
    up 'disease_disease' as if it were an entity name.

    `cols_arg` overrides detection as "child,parent" column names. `flip`
    swaps them, for a table stored parent-first."""
    if not path:
        return {}
    cols = columns(path)
    if cols_arg:
        want = [c.strip() for c in cols_arg.split(",")]
        if len(want) != 2 or any(c not in cols for c in want):
            sys.exit(f"--ancestor-cols {cols_arg!r} does not name two columns "
                     f"of {path}.\nColumns are: {', '.join(cols)}")
        c_child, c_parent = want
    else:
        gs, gt, _gr = guess_cols(cols)
        if not (gs and gt):
            sys.exit(f"cannot find the two name columns in {path}.\n"
                     f"Columns are: {', '.join(cols)}\n"
                     f"Name them with --ancestor-cols child,parent")
        c_child, c_parent = gs, gt
    if flip:
        c_child, c_parent = c_parent, c_child
    anc = defaultdict(list)
    n_rows = 0
    for row in open_rows(path, [c_child, c_parent]):
        ch = (row.get(c_child) or "").strip()
        pa = (row.get(c_parent) or "").strip()
        if ch and pa and ch.lower() != pa.lower():
            anc[ch].append(pa)
            n_rows += 1
    out = {k: sorted(set(v)) for k, v in anc.items()}
    print(f"  ancestors from {os.path.basename(path)}: child={c_child} "
          f"parent={c_parent} -> {n_rows:,} rows, {len(out):,} children",
          file=sys.stderr)

    # A hierarchy has about as many distinct children as it has rows. Many
    # rows collapsing onto a handful of children means those columns hold
    # category labels, not entity names -- a relation or type column picked
    # up by mistake. Fail here rather than let the run finish and report a
    # granularity column built out of the word "parent-child".
    # A symmetric table carries no direction, so climbing it walks the graph
    # both ways and collects siblings and descendants as if they were
    # ancestors. PrimeKG stores most relations in both directions, so this is
    # the common case and it has to stop the run rather than produce a
    # granularity column built from a thousand names per target.
    if out:
        pairs = {(c, p) for c, ps in out.items() for p in ps}
        rev = sum(1 for c, p in pairs if (p, c) in pairs)
        if pairs and rev / len(pairs) > 0.5:
            sys.exit(
                f"\n{path} is stored in both directions: {rev:,} of "
                f"{len(pairs):,} pairs also appear reversed.\n"
                f"Parent and child cannot be told apart in it, so climbing it "
                f"collects descendants and\nsiblings alongside ancestors and "
                f"T3 becomes meaningless. --flip-ancestors cannot help.\n\n"
                f"Build a directed table from the source ontology instead, "
                f"then pass that as --ancestors.\n"
                f"For PrimeKG diseases the source is MONDO and "
                f"mondo_parents.py does this.")

    if n_rows >= 100 and len(out) < 25:
        worst = max(out, key=lambda k: len(out[k]))
        sys.exit(
            f"\n{path} does not look like a hierarchy.\n"
            f"Reading child={c_child} parent={c_parent} gave {n_rows:,} rows "
            f"but only {len(out)} distinct children.\n"
            f"A hierarchy has roughly one child per row; this has one child "
            f"per {n_rows // max(1, len(out)):,} rows,\nso those columns hold "
            f"category labels rather than entity names. The most common child "
            f"value is {worst!r}.\n\nColumns available:\n  {', '.join(cols)}\n"
            f"\nName the right two with --ancestor-cols child,parent.")
    return out


def climb(name, anc, depth=2):
    """Ancestor names up `depth` levels, in the shape tier_sets expects."""
    out, frontier, seen = [], [name], {name.lower()}
    for _ in range(depth):
        nxt = []
        for n in frontier:
            for p in anc.get(n, []):
                if p.lower() in seen:
                    continue
                seen.add(p.lower())
                out.append({"rank": "parent", "scientific": p, "vernacular": []})
                nxt.append(p)
        frontier = nxt
        if not frontier:
            break
    return out


# ---------------------------------------------------------------- matching

MATCH_RULE = "word"


def mentions(text, phrase, rule=None):
    """Does the text name this entity?

    Three rules, because the rule chosen moves the headline number and any
    comparison against an earlier measurement has to hold it fixed:

      word       whole phrase on word boundaries, trailing 's' tolerated.
                 "asthma" does not match "asthmatic". The default.
      strict     as above but no plural tolerance.
      substring  plain case-insensitive containment. "asthma" matches
                 "asthmatic", "colitis" matches "ulcerative colitis". This
                 inflates the rate and is offered only to reproduce a figure
                 produced under it, never to report."""
    rule = rule or MATCH_RULE
    p = phrase.strip()
    if len(p) < 4:
        return False
    if rule == "substring":
        return p.lower() in text.lower()
    tail = r"s?" if rule == "word" else r""
    pat = r"\b" + r"\s+".join(re.escape(w) for w in p.split()) + tail + r"\b"
    return re.search(pat, text, re.IGNORECASE) is not None


def too_generic(name):
    n = name.strip().lower()
    return " " not in n and n in GENERIC


def tier_sets(target, names):
    """The four cumulative match sets for one target entity."""
    t0 = [target]
    t1 = t0 + [s for s in names.get("synonyms", []) if s.lower() != target.lower()]
    t2 = t1 + list(names.get("vernacular", []))
    t3 = list(t2)
    for anc in names.get("ancestors", []):
        t3.append(anc["scientific"])
        t3.extend(anc.get("vernacular", []))

    def dedupe(xs):
        seen, out = set(), []
        for x in xs:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out
    return [dedupe(t0), dedupe(t1), dedupe(t2), dedupe(t3)]


def hit_tiers(text, sets, rule=None):
    """Which cumulative tiers match. Monotone by construction."""
    out, found = [], False
    for s in sets:
        if not found:
            found = any(mentions(text, p, rule) for p in s)
        out.append(found)
    return out


# -------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


# ------------------------------------------------------------ input files

TABULAR = (".csv", ".tsv", ".parquet", ".pq", ".csv.gz", ".tsv.gz", ".txt.gz",
           ".sif", ".sif.gz", ".tab", ".tab.gz")


def _opener(path):
    return gzip.open if path.lower().endswith(".gz") else open


def tabular_files(root):
    out = []
    for base, _dirs, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(TABULAR):
                p = os.path.join(base, f)
                try:
                    out.append((p, os.path.getsize(p)))
                except OSError:
                    pass
    return sorted(out)


def human(nbytes):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if nbytes >= div:
            return f"{nbytes / div:.1f} {unit}"
    return f"{nbytes} B"


def columns(path):
    if path.lower().endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            sys.exit("reading parquet needs pyarrow: pip install pyarrow")
        return list(pq.ParquetFile(path).schema_arrow.names)
    with _opener(path)(path, "rt", encoding="utf-8",
                       errors="replace", newline="") as fh:
        first = fh.readline()
    delim = "\t" if first.count("\t") > first.count(",") else ","
    return next(csv.reader([first], delimiter=delim))


def open_rows(path, want=None):
    """Stream dict rows from csv, tsv, gzipped either, or parquet."""
    if path.lower().endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        cols = [c for c in (want or []) if c in have] or None
        for batch in pf.iter_batches(batch_size=65536, columns=cols):
            d = batch.to_pydict()
            keys = list(d.keys())
            if not keys:
                continue
            for i in range(len(d[keys[0]])):
                yield {k: ("" if d[k][i] is None else str(d[k][i])) for k in keys}
        return
    fh = _opener(path)(path, "rt", encoding="utf-8",
                       errors="replace", newline="")
    try:
        first = fh.readline()
        if not first:
            return
        delim = "\t" if first.count("\t") > first.count(",") else ","
        fields = next(csv.reader([first], delimiter=delim))
        for row in csv.DictReader(fh, fieldnames=fields, delimiter=delim):
            yield row
    finally:
        fh.close()


def guess_cols(cols):
    low = {c.lower(): c for c in cols}

    def pick(*patterns):
        for pat in patterns:
            for lc, orig in low.items():
                if pat(lc):
                    return orig
        return None

    # Order matters. Name-bearing columns win first, so PrimeKG's x_name is
    # preferred over its x_source provenance column; the species/taxon forms
    # come next for extracts like source_species; a bare "source"/"target"
    # is the last resort.
    entity_word = ("species", "taxon", "entity", "node", "organism", "id")
    src = pick(lambda c: "source" in c and "name" in c,
               lambda c: c in ("x_name", "subject", "head", "child"),
               lambda c: c.startswith("x_") and "name" in c,
               lambda c: "source" in c and any(k in c for k in entity_word),
               lambda c: c in ("source", "subj", "from"))
    tgt = pick(lambda c: "target" in c and "name" in c,
               lambda c: c in ("y_name", "object", "tail", "parent"),
               lambda c: c.startswith("y_") and "name" in c,
               lambda c: "target" in c and any(k in c for k in entity_word),
               lambda c: c in ("target", "obj", "to"))
    rel = pick(lambda c: "interactiontype" in c and "name" in c,
               lambda c: c in ("display_relation", "relation", "predicate",
                               "interaction", "edge_type"),
               lambda c: "relation" in c)
    return src, tgt, rel


def survey(root):
    found = tabular_files(root)
    if not found:
        print(f"no tabular files under {root}")
        return
    print(f"\n{len(found)} tabular files under {root}\n")
    for p, size in found:
        rel = os.path.relpath(p, root)
        try:
            cols = columns(p)
        except Exception as e:                    # noqa: BLE001
            print(f"  {rel:<52} {human(size):>9}   unreadable: {str(e)[:40]}")
            continue
        gs, gt, gr = guess_cols(cols)
        print(f"  {rel:<52} {human(size):>9}   {'edges' if (gs and gt) else '     '}  "
              f"{len(cols)} cols: {', '.join(cols[:6])}"
              f"{' ...' if len(cols) > 6 else ''}")
        if gs and gt:
            print(f"  {'':<52} {'':>9}          source={gs} target={gt} "
                  f"relation={gr or '(none; filename used)'}")
    print("\nRows marked 'edges' have source and target name columns and can be "
          "passed to --edges.\nRe-run --inspect on one of them to see its "
          "relation values.")


def inspect(paths, c_rel, limit=60):
    for path in paths:
        if os.path.isdir(path):
            survey(path)
            continue
        cols = columns(path)
        print(f"\n{path}\n{'-' * min(len(path), 100)}")
        print(f"{len(cols)} columns:")
        for c in cols:
            print(f"  {c}")
        gs, gt, gr = guess_cols(cols)
        print("\nlikely columns for this script:")
        print(f"  --col-source   {gs or '?'}")
        print(f"  --col-target   {gt or '?'}")
        print(f"  --col-relation {gr or '(none found)'}")
        rel_col = c_rel if c_rel in cols else gr
        if not rel_col:
            stem = re.sub(r"\.(csv|tsv|parquet|pq)(\.gz)?$", "",
                          os.path.basename(path), flags=re.I)
            print(f"\nNo relation column, so every row in this file counts as "
                  f"one relation named {stem!r}.\nThat is the right shape for a "
                  f"single-relation extract: pass several such files to --edges "
                  f"at once\nand each is labelled by its own filename.")
            continue
        print(f"\nvalues of {rel_col!r}, by edge count:")
        counts = defaultdict(int)
        capped = False
        for n, row in enumerate(open_rows(path, [rel_col]), 1):
            counts[(row.get(rel_col) or "").strip()] += 1
            if n >= 4_000_000:
                capped = True
                break
        for v, k in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]:
            print(f"  {k:>12,}  {v}")
        if len(counts) > limit:
            print(f"  ... {len(counts) - limit:,} more")
        print(f"\n{len(counts):,} distinct values"
              + (" (first 4,000,000 rows only)" if capped else "")
              + "; copy the ones you want into --relations,\nquoting the whole "
                "list if any contains a space.")


def load_edges(paths, c_src, c_tgt, c_rel, relations, binomials=True):
    """Edges per relation, deduped, across one or more tabular files.

    Two input shapes are both handled: one file with a relation column, or
    several single-relation files, each labelled by its own filename stem."""
    want = set(relations) if relations else None
    by_rel = defaultdict(set)
    for path in paths:
        cols = columns(path)
        gs, gt, gr = guess_cols(cols)
        cs = c_src if c_src in cols else gs
        ct = c_tgt if c_tgt in cols else gt
        cr = c_rel if c_rel in cols else gr
        if not cs or not ct:
            print(f"  skip {os.path.basename(path)}: no source/target name "
                  f"columns among {', '.join(cols[:8])}", file=sys.stderr)
            continue
        stem = re.sub(r"\.(csv|tsv|parquet|pq)(\.gz)?$", "",
                      os.path.basename(path), flags=re.I)
        if not cr:
            print(f"  {os.path.basename(path)}: no relation column, so these "
                  f"edges are labelled {stem!r}", file=sys.stderr)
        fetch = [c for c in (cs, ct, cr) if c]
        kept = 0
        for row in open_rows(path, fetch):
            rel = (row.get(cr) or "").strip() if cr else stem
            if not rel:
                rel = stem
            if want and rel not in want:
                continue
            s = (row.get(cs) or "").strip()
            t = (row.get(ct) or "").strip()
            if not s or not t or s.lower() == t.lower():
                continue
            if binomials and (len(s.split()) < 2 or len(t.split()) < 2):
                continue
            if too_generic(t):
                continue
            by_rel[rel].add((s, t))
            kept += 1
        if len(paths) > 1:
            print(f"  {os.path.basename(path):<28} {kept:>9,} edges kept",
                  file=sys.stderr)
    return {r: sorted(v) for r, v in by_rel.items()}


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, nargs="+",
                    help="one edge file, several single-relation files, or a "
                         "directory to survey with --inspect")
    ap.add_argument("--col-source", default="sourceTaxonName")
    ap.add_argument("--col-target", default="targetTaxonName")
    ap.add_argument("--col-relation", default="interactionTypeName")
    ap.add_argument("--relations", default="",
                    help="comma-separated; omit to measure the largest "
                         "relations present in the files given")
    ap.add_argument("--max-relations", type=int, default=12)
    ap.add_argument("--n", type=int, default=600, help="measured edges PER relation")
    ap.add_argument("--floor-per-edge", type=int, default=3,
                    help="random non-partners per measured edge, at every tier")
    ap.add_argument("--min-chars", type=int, default=3000,
                    help="skip source articles shorter than this")
    ap.add_argument("--oversample", type=float, default=3.0,
                    help="candidate edges fetched per edge needed, since many "
                         "sources have no article or a stub")
    ap.add_argument("--wiki-cache", default="wiki_cache")
    ap.add_argument("--gbif-cache", default="gbif_cache")
    ap.add_argument("--names", choices=["gbif", "redirects", "none"], default="gbif",
                    help="gbif for taxa (GloBI); redirects for any domain "
                         "(PrimeKG, Hetionet); none leaves only T0 meaningful")
    ap.add_argument("--no-gbif", action="store_true", help="alias for --names none")
    ap.add_argument("--ancestors", default="",
                    help="hierarchy file for T3 when --names redirects; "
                         "PrimeKG's pkg_pc.csv works, columns auto-detected")
    ap.add_argument("--ancestor-cols", default="",
                    help="override detection as child,parent column names")
    ap.add_argument("--flip-ancestors", action="store_true",
                    help="the hierarchy file is stored parent-first")
    ap.add_argument("--ancestor-depth", type=int, default=2,
                    help="levels to climb for T3 (default 2). With GBIF this "
                         "counts ranks nearest first, so 1 is genus and 2 is "
                         "genus plus family. Deep climbs reach terms so broad "
                         "they appear in any article, which shows up as a "
                         "high T3 floor.")
    ap.add_argument("--ancestor-vern-cap", type=int, default=4,
                    help="common names kept per ancestor rank (default 4). A "
                         "clade can carry dozens, most of them vague.")
    ap.add_argument("--max-floor", type=float, default=15.0,
                    help="T3 floor above this marks the granularity column "
                         "unmeasurable rather than reporting it (default 15)")
    ap.add_argument("--no-ancestor-vernaculars", dest="anc_vern",
                    action="store_false", default=True,
                    help="skip per-clade common-name lookups (faster, and T3 "
                         "then uses scientific ancestor names only)")
    ap.add_argument("--binomials-only", dest="binomials", action="store_true",
                    default=True, help="require two-word names (default; taxa)")
    ap.add_argument("--any-names", dest="binomials", action="store_false",
                    help="allow single-word entity names (drugs, diseases)")
    ap.add_argument("--rps", type=float, default=4.0,
                    help="request budget per second before backoff (default 4)")
    ap.add_argument("--contact", default="",
                    help="optional contact string for the User-Agent; "
                         "Wikipedia's API etiquette asks for one on heavy use")
    ap.add_argument("--match", choices=["word", "strict", "substring"],
                    default="word", help="what counts as naming the entity; "
                                         "see mentions(). Default word.")
    ap.add_argument("--match-rules", action="store_true",
                    help="also report T0 under all three match rules, to "
                         "reconcile against a figure measured under another")
    ap.add_argument("--inspect", action="store_true",
                    help="print columns and relation values, then exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="", help="write per-relation results here")
    a = ap.parse_args()

    for p in a.edges:
        if os.path.exists(p):
            continue
        here = os.path.dirname(os.path.abspath(p)) or "."
        near = tabular_files(here) if os.path.isdir(here) else []
        msg = f"no such edge file: {p}"
        if near:
            msg += "\n\ntabular files under that directory:\n" + "\n".join(
                f"  {q}  ({human(s)})" for q, s in near[:40])
            msg += "\n\nPick one and re-run with --inspect to see its columns."
        sys.exit(msg)

    if a.inspect:
        inspect(a.edges, a.col_relation)
        return

    dirs = [p for p in a.edges if os.path.isdir(p)]
    if dirs:
        sys.exit(f"{dirs[0]} is a directory. Run --inspect on it to see what "
                 f"is inside, then pass the file(s) you want.")

    global MATCH_RULE
    MATCH_RULE = a.match
    rng = random.Random(a.seed)
    asked = [r.strip() for r in a.relations.split(",") if r.strip()]
    mode = "none" if a.no_gbif else a.names
    anc_tab = load_ancestors(a.ancestors, a.ancestor_cols, a.flip_ancestors)
    if mode == "redirects" and not anc_tab:
        print("note: --names redirects without --ancestors, so T3 equals T2 "
              "and the granularity lift is not measured in this run.",
              file=sys.stderr)
    elif anc_tab:
        print(f"  ancestors table: {len(anc_tab):,} entities have a parent",
              file=sys.stderr)
    fx = Fetcher(rps=a.rps, contact=a.contact)
    os.makedirs(a.wiki_cache, exist_ok=True)
    os.makedirs(a.gbif_cache, exist_ok=True)

    def names_for(entity):
        """Cached names only. Fetching happens in bulk, before scoring."""
        if mode == "gbif":
            n = read_names(a.gbif_cache, "gbif", entity)
            if not n:
                return EMPTY
            # GBIF returns the whole ladder from genus up to kingdom, nearest
            # first. Taking all of it puts "mammals" and "animals" in the
            # match set, which appear in any article and make T3 unreadable.
            # --ancestor-depth keeps only the nearest ranks, matching what it
            # means for a hierarchy file.
            anc = []
            for rk in (n.get("ancestors") or [])[:a.ancestor_depth]:
                anc.append({"rank": rk.get("rank", ""),
                            "scientific": rk.get("scientific", ""),
                            "vernacular": (rk.get("vernacular") or [])[:a.ancestor_vern_cap]})
            return {"synonyms": n.get("synonyms", []),
                    "vernacular": n.get("vernacular", []),
                    "ancestors": anc}
        if mode == "redirects":
            n = dict(read_names(a.gbif_cache, "redir", entity) or EMPTY)
            n["synonyms"] = []
            n["ancestors"] = climb(entity, anc_tab, a.ancestor_depth)
            return n
        return EMPTY

    print(f"loading edges from {a.edges} (names: {mode}) ...", file=sys.stderr)
    by_rel = load_edges(a.edges, a.col_source, a.col_target, a.col_relation,
                        asked or None, binomials=a.binomials)
    if not by_rel:
        sys.exit("no usable edges were read. Run --inspect to check the column "
                 "names,\nand quote --relations if a name contains a space.")

    if asked:
        rels = [r for r in asked if by_rel.get(r)]
        if not rels:
            sys.exit("none of the requested relations were found. Present "
                     "instead:\n" + "\n".join(
                         f"  {len(v):>9,}  {k}" for k, v in
                         sorted(by_rel.items(), key=lambda kv: -len(kv[1]))[:25]))
        for r in asked:
            if r not in rels:
                print(f"  note: {r!r} not found, skipping", file=sys.stderr)
    else:
        ranked = sorted(by_rel.items(), key=lambda kv: -len(kv[1]))
        rels = [r for r, _ in ranked[:a.max_relations]]
        if len(ranked) > len(rels):
            print(f"  measuring the {len(rels)} largest of {len(ranked)} "
                  f"relations found; pass --relations to choose",
                  file=sys.stderr)
    for r in rels:
        print(f"  {r:<28} {len(by_rel.get(r, [])):>9,} edges available",
              file=sys.stderr)

    results, samples = [], []
    all_tgts = set()
    for rel in rels:
        edges = list(by_rel[rel])
        rng.shuffle(edges)
        partners = defaultdict(set)
        for s, t in edges:
            partners[s].add(t)

        print(f"\n=== {rel} ===", file=sys.stderr)

        # Phase 1: fetch candidate source articles in bulk, then keep the
        # first n that are long enough. Oversampled because many sources have
        # no article at all.
        cand = edges[:max(a.n, int(a.n * a.oversample))]
        fetch_articles(fx, [s for s, _ in cand], a.wiki_cache)
        measurable = []
        for s, t in cand:
            txt = read_article(a.wiki_cache, s)
            if txt is not None and len(txt) >= a.min_chars:
                measurable.append((s, t))
            if len(measurable) >= a.n:
                break
        if len(measurable) < 20:
            print(f"  only {len(measurable)} measurable edges; skipping",
                  file=sys.stderr)
            continue

        # Phase 2: fetch name sets for every target in bulk. The floor draws
        # from this same set, so it costs no further requests and the floor
        # entities have name sets as rich as the real partners' -- which is
        # what makes the floor a fair comparison at T2 and T3.
        tgts = sorted({t for _, t in measurable})
        all_tgts |= set(tgts)
        if mode == "redirects":
            fetch_redirect_names(fx, tgts, a.gbif_cache)
        elif mode == "gbif":
            fetch_gbif_names(fx, tgts, a.gbif_cache,
                             ancestor_vernaculars=a.anc_vern)

        # Phase 3: score, entirely from cache.
        hits = [0, 0, 0, 0]
        fl_hits = [0, 0, 0, 0]
        rule_hits = {r: 0 for r in ("word", "strict", "substring")}
        n_anc = 0
        measured = fl_n = 0
        for src, tgt in measurable:
            text = read_article(a.wiki_cache, src)
            tset = tier_sets(tgt, names_for(tgt))
            n_anc += len(tset[3]) - len(tset[2])
            got = hit_tiers(text, tset)
            for i, g in enumerate(got):
                hits[i] += g
            if a.match_rules:
                for r in rule_hits:
                    rule_hits[r] += mentions(text, tgt, r)
            measured += 1
            for _ in range(a.floor_per_edge):
                for _try in range(12):
                    c = rng.choice(tgts)
                    if c not in partners[src] and c.lower() != src.lower():
                        break
                else:
                    continue
                fg = hit_tiers(text, tier_sets(c, names_for(c)))
                for i, g in enumerate(fg):
                    fl_hits[i] += g
                fl_n += 1
            if len(samples) < 12 and got[3] and not got[0]:
                lvl = "synonym" if got[1] else ("common name" if got[2] else "ancestor")
                samples.append((rel, src, tgt, lvl))

        rates = [wilson(h, measured)[0] for h in hits]
        floors = [wilson(h, fl_n)[0] if fl_n else 0.0 for h in fl_hits]
        corr = [max(0.0, rates[i] - floors[i]) for i in range(4)]
        results.append(dict(
            rel=rel, n=measured, fl_n=fl_n, rates=rates, floors=floors,
            corr=corr, nom=corr[2] - corr[0], gran=corr[3] - corr[2],
            ci=[wilson(h, measured) for h in hits],
            anc_per_edge=n_anc / max(1, measured),
            rules={r: 100 * k / max(1, measured) for r, k in rule_hits.items()}))
        per = n_anc / max(1, measured)
        print(f"  measured {measured}, floor draws {fl_n}, "
              f"{per:.1f} ancestor names per target, {fx.report()}",
              file=sys.stderr)
        if per > 8:
            print(f"  WARNING: {per:.0f} ancestor names per target is too many "
                  f"for T3 to mean anything.\n"
                  f"           Lower --ancestor-depth (now {a.ancestor_depth}) "
                  f"or --ancestor-vern-cap (now {a.ancestor_vern_cap}).",
                  file=sys.stderr)
        if measured < a.n:
            print(f"  note: wanted {a.n} but only {measured} candidates had an "
                  f"article of {a.min_chars}+ chars.\n"
                  f"        Raise --oversample (now {a.oversample}) to look "
                  f"at more edges per relation.", file=sys.stderr)

    if not results:
        sys.exit("nothing measured")

    # ------------------------------------------------------------- report
    print("\n" + "=" * 112)
    print("TRACE RATE BY MATCH TIER   (T0 name · T1 +synonyms · "
          "T2 +common names · T3 +ancestors)")
    print("=" * 112)
    print(f"{'relation':<24} {'n':>5} {'T0':>7} {'T1':>7} {'T2':>7} {'T3':>7}   "
          f"{'floor T0':>8} {'floor T2':>8} {'floor T3':>8}")
    print("-" * 112)
    for r in results:
        print(f"{r['rel']:<24} {r['n']:>5} "
              + " ".join(f"{v:>6.2f}%" for v in r["rates"]) + "   "
              + f"{r['floors'][0]:>7.2f}% {r['floors'][2]:>7.2f}% "
                f"{r['floors'][3]:>7.2f}%")

    print("\n" + "=" * 112)
    print("WHERE THE LIFT COMES FROM, after subtracting the floor")
    print("=" * 112)
    print(f"{'relation':<24} {'T0 corrected':>13} {'nomenclature':>13} "
          f"{'granularity':>13}   reading")
    print("-" * 112)
    for r in sorted(results, key=lambda r: -(r["gran"] - r["nom"])):
        bad_floor = r["floors"][3] > a.max_floor
        if bad_floor:
            v = f"granularity UNMEASURABLE (floor {r['floors'][3]:.0f}%)"
            gran = "       n/a"
        else:
            gran = f"{r['gran']:>11.2f}%"
            if r["gran"] > 2 * max(r["nom"], 1.0):
                v = "granularity dominates"
            elif r["nom"] > 2 * max(r["gran"], 1.0):
                v = "nomenclature dominates"
            elif max(r["nom"], r["gran"]) < 2.0:
                v = "neither helps"
            else:
                v = "mixed"
        print(f"{r['rel']:<24} {r['corr'][0]:>12.2f}% {r['nom']:>12.2f}% "
              f"{gran:>12}   {v}")
    print("-" * 112)
    good = [r for r in results if r["floors"][3] <= a.max_floor]
    print(f"{'mean across relations':<24} {'':>13} "
          f"{sum(r['nom'] for r in results) / len(results):>12.2f}% "
          + (f"{sum(r['gran'] for r in good) / len(good):>12.2f}%"
             if good else f"{'n/a':>12}"))

    if not good:
        print("\n" + "!" * 112)
        print(f"GRANULARITY NOT MEASURED. Every relation's T3 floor exceeds "
              f"{a.max_floor:.0f}%, which means random")
        print("non-partners match at that tier about as often as real ones, "
              "so the column carries no signal.")
        print(f"Ancestor names per target: "
              + ", ".join(f"{r['rel']} {r['anc_per_edge']:.0f}" for r in results))
        print("\nA target should have a handful of ancestors, not hundreds. "
              "Lower --ancestor-depth, or")
        print("build a directed hierarchy from the source ontology if the "
              "one in use carries no direction.")
        print("!" * 112)

    if all(r["anc_per_edge"] < 0.01 for r in results):
        print("\n" + "!" * 112)
        print("THE GRANULARITY COLUMN DID NOT RUN. No target in any relation "
              "got a single ancestor name,")
        print("so T3 is identical to T2 by construction and its 0.00% means "
              "'not measured', not 'no effect'.")
        if anc_tab:
            as_child = len(all_tgts & set(anc_tab))
            parents = {p for v in anc_tab.values() for p in v}
            as_parent = len(all_tgts & parents)
            print(f"\nThe table loaded {len(anc_tab):,} entities with a parent. "
                  f"Of the {len(all_tgts):,} targets measured:")
            print(f"  {as_child:,} appear in its CHILD column  <- this is the "
                  f"number that has to be large")
            print(f"  {as_parent:,} appear in its PARENT column")
            if as_parent > as_child:
                print("\nMore targets are parents than children, so the file "
                      "is stored parent-first.\nRe-run with --flip-ancestors.")
            else:
                print("\nNeither column matches the target names, so this is "
                      "the wrong columns rather than\nthe wrong order. Name "
                      "them explicitly with --ancestor-cols child,parent.")
            print("\nFirst rows as read (child -> parents):")
            for k, v in list(anc_tab.items())[:3]:
                print(f"  {k!r} -> {v[:4]!r}{' ...' if len(v) > 4 else ''}")
            print("\nA target name, for comparison:")
            for t in sorted(all_tgts)[:3]:
                print(f"  {t!r}")
        else:
            print("\nNo ancestors table was given, so pass one with "
                  "--ancestors to measure this column.")
        print("!" * 112)

    if a.match_rules:
        print("\n" + "=" * 112)
        print("T0 UNDER EACH MATCH RULE -- hold this fixed when comparing "
              "against an earlier figure")
        print("=" * 112)
        print(f"{'relation':<24} {'strict':>9} {'word':>9} {'substring':>11}"
              f"   {'substring inflation':>20}")
        print("-" * 112)
        for r in results:
            ru = r["rules"]
            print(f"{r['rel']:<24} {ru['strict']:>8.2f}% {ru['word']:>8.2f}% "
                  f"{ru['substring']:>10.2f}%   {ru['substring'] - ru['word']:>19.2f}%")
        print("\nA large substring column means partial-word hits: 'asthma' "
              "inside 'asthmatic', 'colitis'\ninside 'ulcerative colitis'. "
              "Report the word column; the substring one is for reconciling only.")

    if samples:
        print("\n" + "=" * 112)
        print("EXAMPLES RECOVERED ABOVE T0 -- read these before trusting the table")
        print("=" * 112)
        for rel, src, tgt, lvl in samples:
            print(f"  {rel:<18} {src:<30} -> {tgt:<30} found at: {lvl}")

    print(f"""
HOW TO READ IT

  nomenclature large, granularity small
      A name table does the work, and the resolution claim shrinks to an
      engineering note.

  granularity large, nomenclature small
      The text names groups, not synonyms. A name table does not fix it;
      resolving needs the hierarchy and a choice among members.

  both small
      The relation is not written about at any level of description. No
      text-side repair exists for it, which is itself a finding.

  A spread in the T0 column across relations over the SAME entity types is a
  statement about relations, not about names.

CAVEATS
  - T2 and T3 admit ordinary words, so their floors rise. Only the
    floor-corrected columns are comparable. A floor above roughly 15% at T3
    means that tier is reading vocabulary frequency, not evidence, and needs
    tightening to mentions near the relation verb.
  - A T3 hit says an ancestor was named, not the specific partner.
  - Matching is lexical, so a relation stated without any name is a miss at
    every tier, correctly.

network: {fx.report()}
""")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["relation", "n", "floor_n", "T0", "T1", "T2", "T3",
                        "T0_lo", "T0_hi", "T3_lo", "T3_hi",
                        "floor_T0", "floor_T1", "floor_T2", "floor_T3",
                        "T0_corrected", "nomenclature_lift", "granularity_lift"])
            for r in results:
                w.writerow([r["rel"], r["n"], r["fl_n"]]
                           + [f"{v:.2f}" for v in r["rates"]]
                           + [f"{r['ci'][0][1]:.2f}", f"{r['ci'][0][2]:.2f}",
                              f"{r['ci'][3][1]:.2f}", f"{r['ci'][3][2]:.2f}"]
                           + [f"{v:.2f}" for v in r["floors"]]
                           + [f"{r['corr'][0]:.2f}", f"{r['nom']:.2f}",
                              f"{r['gran']:.2f}"])
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
