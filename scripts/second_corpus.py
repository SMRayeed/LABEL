#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2. Whether the silent box is a property of the graph or of Wikipedia.

The paper's central number is the share of edges no text states. Measured
against one encyclopedia, that number is open to the obvious objection: the
fact may be written down somewhere else, and the finding would then be about
Wikipedia's coverage rather than about the graph. The objection has to be
answered with a second corpus, not with an argument.

This script adds one and re-measures. For every source entity it builds a
second document from a different body of text, applies the same tier
matching to it, and reports the silent box three ways: against the
encyclopedia alone, against the second corpus alone, and against the union.
The claim survives if the union leaves the silent box close to where the
encyclopedia alone put it.

Two corpora are supported, because the two graphs need different ones.

  pubmed       Abstracts of the most relevant articles for the entity, from
               NCBI E-utilities. The right second corpus for drugs and
               diseases: it is where the clinical literature actually lives,
               so a fact absent from it is absent from the field's own
               writing, not just from an encyclopedia.

  wikispecies  The species page from species.wikimedia.org, a taxonomic wiki
               with a different editorial basis from Wikipedia. The right
               second corpus for an interaction graph.

Two cautions belong in the write-up. An abstract set is not a full text, so
this measures what the literature says in its summaries; a fact stated only
in a methods section is counted silent, and the resulting coverage figure is
a lower bound on the literature rather than an estimate of it. And the
abstract set is selected by the entity's own name, so an abstract reaches
the measurement only when it is about that entity, which is the same
restriction the encyclopedia article carries.

    python3 second_corpus.py --scored matched_edges.csv \\
        --corpus pubmed --names-cache gbif_cache \\
        --ancestors mondo_child_parent.csv --ancestor-depth 2 \\
        --retmax 40 --out matched_edges_pm.csv

    python3 second_corpus.py --scored globi_edges.csv \\
        --corpus wikispecies --names-cache gbif_cache --names gbif \\
        --out globi_edges_ws.csv

Run from a login node; the compute nodes have no network. The corpus cache
makes a repeat run nearly free, and interrupting is safe.

Needs name_tiers.py beside it. Standard library otherwise.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import name_tiers as nt
except ImportError as e:
    sys.exit(f"name_tiers.py has to sit beside this script: {e}")

csv.field_size_limit(min(2**31 - 1, sys.maxsize))

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
WIKISPECIES = "https://species.wikimedia.org/w/api.php"


class RawFetcher:
    """Paced fetch that returns text rather than parsed JSON.

    E-utilities serves abstracts as plain text, which the JSON fetcher in
    name_tiers cannot read. The pacing follows NCBI's published limit: three
    requests a second without a key, ten with one.
    """

    def __init__(self, rps=2.5, contact="", tries=5, timeout=90):
        self.gap = 1.0 / max(0.2, rps)
        self.tries = tries
        self.timeout = timeout
        self.last = 0.0
        self.n_req = 0
        self.n_failed = 0
        ua = "NameTierStudy/2.0 (research script; stdlib urllib)"
        if contact:
            ua += f" ({contact})"
        self.headers = {"User-Agent": ua, "Accept-Encoding": "gzip"}

    def get(self, url):
        for attempt in range(self.tries):
            dt = time.time() - self.last
            if dt < self.gap:
                time.sleep(self.gap - dt)
            self.last = time.time()
            self.n_req += 1
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                return raw.decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(60.0, 4.0 * (2 ** attempt)))
                    continue
                if e.code in (400, 404):
                    self.n_failed += 1
                    return None
                time.sleep(min(30.0, 3.0 * (2 ** attempt)))
            except Exception as e:                           # noqa: BLE001
                if attempt == self.tries - 1:
                    print(f"    request failed: {str(e)[:80]}",
                          file=sys.stderr)
                time.sleep(min(30.0, 3.0 * (2 ** attempt)))
        self.n_failed += 1
        return None

    def report(self):
        return f"{self.n_req:,} requests, {self.n_failed} unrecoverable"


def cache_path(cache, name, suffix=""):
    return os.path.join(cache, f"{nt._safe(name)}{suffix}.txt")


def read_cached(cache, name, suffix=""):
    p = cache_path(cache, name, suffix)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_cached(cache, name, text, suffix=""):
    os.makedirs(cache, exist_ok=True)
    tmp = cache_path(cache, name, suffix) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, cache_path(cache, name, suffix))


def fetch_pubmed(fx, name, retmax, api_key, terms=()):
    """Abstracts of the most relevant PubMed records for one entity.

    Without `terms` this issues one relevance-sorted search on the entity
    name, and the records that come back are whichever PubMed ranks highest.
    That ranking is not neutral between relations: efficacy trials are more
    numerous and more cited than contraindication reports, so a fixed budget
    spent on one query can hand the efficacy side more of the corpus and
    produce a coverage gap that is a property of the retrieval rather than
    of the literature.

    With `terms` the budget is split evenly across one query per term, so
    each relation gets its own share of the corpus and that explanation is
    ruled out. The union of the returned records is fetched once.

    None means the request could not be completed, which is not the same as
    an empty result and must not be cached. An empty string means the search
    ran and matched nothing, which is a real measurement.
    """
    key = f"&api_key={api_key}" if api_key else ""
    frames = list(terms) or [""]
    per = max(1, retmax // len(frames))
    ids, failed = [], 0
    for frag in frames:
        q = f'"{name}"[Title/Abstract]'
        if frag:
            q += f" AND ({frag})"
        url = (f"{EUTILS}/esearch.fcgi?db=pubmed"
               f"&term={urllib.parse.quote(q)}"
               f"&retmax={per}&retmode=json&sort=relevance{key}")
        body = fx.get(url)
        if body is None:
            failed += 1
            continue
        try:
            got = json.loads(body)["esearchresult"].get("idlist", [])
        except (ValueError, KeyError):
            failed += 1
            continue
        ids.extend(got)
    if failed == len(frames):
        return None
    ids = list(dict.fromkeys(ids))
    if not ids:
        return ""
    url = (f"{EUTILS}/efetch.fcgi?db=pubmed&id={','.join(ids)}"
           f"&rettype=abstract&retmode=text{key}")
    text = fx.get(url)
    if text is None:
        return None
    return text


def fetch_wikispecies(fx, name):
    """The species.wikimedia.org page for one taxon, as plain extract text."""
    url = (f"{WIKISPECIES}?action=query&prop=extracts&explaintext=1"
           f"&redirects=1&format=json&titles="
           + urllib.parse.quote(name))
    body = fx.get(url)
    if body is None:
        return None
    try:
        pages = json.loads(body)["query"]["pages"]
    except (ValueError, KeyError):
        return None
    out = []
    for p in pages.values():
        if "missing" in p:
            continue
        out.append(p.get("extract") or "")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True,
                    help="an edge or scored file with source, target, tier")
    ap.add_argument("--corpus", choices=["pubmed", "wikispecies"],
                    default="pubmed")
    ap.add_argument("--corpus-cache", default="")
    ap.add_argument("--names", choices=["redirects", "gbif", "none"],
                    default="redirects",
                    help="must match what the build stage used")
    ap.add_argument("--names-cache", default="gbif_cache")
    ap.add_argument("--ancestors", default="")
    ap.add_argument("--ancestor-cols", default="")
    ap.add_argument("--flip-ancestors", action="store_true")
    ap.add_argument("--ancestor-depth", type=int, default=2)
    ap.add_argument("--retmax", type=int, default=40,
                    help="abstracts per entity; relevance-sorted")
    ap.add_argument("--extra-terms", default="",
                    help="pipe-separated PubMed query fragments. One search "
                         "is issued per fragment with the budget split "
                         "evenly, so each relation gets its own share of the "
                         "corpus and a coverage gap cannot be attributed to "
                         "the ranking. Fragments go in their own cache.")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="a document shorter than this counts as absent, so "
                         "a stub does not read as a genuine silence")
    ap.add_argument("--rps", type=float, default=0.0,
                    help="0 picks 2.5 without an API key and 8.0 with one")
    ap.add_argument("--api-key", default="",
                    help="an NCBI key raises the rate limit; optional")
    ap.add_argument("--contact", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N distinct sources, for a smoke test")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cache = a.corpus_cache or f"{a.corpus}_cache"
    rps = a.rps or (8.0 if a.api_key else 2.5)
    terms = [t.strip() for t in a.extra_terms.split("|") if t.strip()]
    # A balanced run retrieves a different corpus from a plain one, so it
    # needs its own cache entries. Without the suffix the second run would
    # read the first one's documents back and report it as a new result.
    suffix = ""
    if terms:
        import hashlib
        h = hashlib.sha1("|".join(terms).encode("utf-8")).hexdigest()[:8]
        suffix = f"__q{h}"
        print(f"balanced retrieval: {len(terms)} queries per entity, "
              f"{max(1, a.retmax // len(terms))} records each",
              file=sys.stderr)
        for t in terms:
            print(f"    AND ({t})", file=sys.stderr)

    if not os.path.exists(a.scored):
        sys.exit(f"missing: {a.scored}\n"
                 f"List the edge and scored files on disk with: "
                 f"ls -1 *edges*.csv *scored*.csv")
    with open(a.scored, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{a.scored} has no rows")
    for need in ("source", "target", "tier"):
        if need not in rows[0]:
            sys.exit(f"{a.scored} has no '{need}' column. Present: "
                     + ", ".join(sorted(rows[0])))

    anc = nt.load_ancestors(a.ancestors, a.ancestor_cols, a.flip_ancestors)

    def names_for(e):
        if a.names == "gbif":
            n = nt.read_names(a.names_cache, "gbif", e)
            if not n:
                return nt.EMPTY
            out = []
            for rk in (n.get("ancestors") or [])[:a.ancestor_depth]:
                out.append({"rank": rk.get("rank", ""),
                            "scientific": rk.get("scientific", ""),
                            "vernacular": (rk.get("vernacular") or [])[:4]})
            return {"synonyms": n.get("synonyms", []),
                    "vernacular": n.get("vernacular", []), "ancestors": out}
        if a.names == "redirects":
            n = dict(nt.read_names(a.names_cache, "redir", e) or nt.EMPTY)
            n["synonyms"] = []
            n["ancestors"] = nt.climb(e, anc, a.ancestor_depth)
            return n
        return nt.EMPTY

    sources = list(dict.fromkeys(r["source"] for r in rows))
    if a.limit:
        sources = sources[:a.limit]
        rows = [r for r in rows if r["source"] in set(sources)]
    todo = [s for s in sources if read_cached(cache, s, suffix) is None]
    print(f"{len(sources):,} distinct sources, {len(todo):,} to fetch "
          f"from {a.corpus} at {rps:.1f} req/s", file=sys.stderr)
    if todo:
        per_src = (len(terms) + 1) if a.corpus == "pubmed" else 1
        mins = len(todo) * per_src / rps / 60
        print(f"  roughly {mins:.0f} minutes; the cache makes a rerun free",
              file=sys.stderr)

    fx = RawFetcher(rps=rps, contact=a.contact)
    t0 = time.time()
    for i, s in enumerate(todo):
        text = (fetch_pubmed(fx, s, a.retmax, a.api_key, terms)
                if a.corpus == "pubmed" else fetch_wikispecies(fx, s))
        if text is None:
            continue            # unknown, not empty; left uncached to retry
        write_cached(cache, s, text, suffix)
        if (i + 1) % 10 == 0:
            rate = (i + 1) / max(1e-9, time.time() - t0)
            left = (len(todo) - i - 1) / max(1e-9, rate)
            print(f"    {i + 1}/{len(todo)}  ~{left / 60:.0f} min left",
                  end="\r", file=sys.stderr)
    if todo:
        print(" " * 48, end="\r", file=sys.stderr)
        print(f"  {fx.report()}", file=sys.stderr)

    # Re-measure. The tier machinery is the one the build used, so a tier
    # here means the same thing it means there.
    out, no_doc = [], 0
    for r in rows:
        src, tgt = r["source"], r["target"]
        doc = read_cached(cache, src, suffix) or ""
        if len(doc) < a.min_chars:
            no_doc += 1
            doc = ""
        sets = nt.tier_sets(tgt, names_for(tgt))
        hits = nt.hit_tiers(doc, sets) if doc else [False] * 4
        t2 = next((k for k, h in enumerate(hits) if h), 4)
        t1 = int(r["tier"])
        rec = dict(r)
        rec["tier_wiki"] = t1
        rec["tier_second"] = t2
        rec["tier_union"] = min(t1, t2)
        rec["second_chars"] = len(doc)
        out.append(rec)

    n = len(out)
    print(f"\n{a.corpus} documents shorter than {a.min_chars} chars, "
          f"counted absent: {no_doc:,} of {n:,}")

    def silent(key, cut):
        return sum(1 for r in out if r[key] > cut)

    print("\nSILENT BOX, three ways")
    print("  cut   encyclopedia   second corpus   union      union shrinks it by")
    for cut in (2, 3):
        w = 100 * silent("tier_wiki", cut) / n
        s = 100 * silent("tier_second", cut) / n
        u = 100 * silent("tier_union", cut) / n
        print(f"   <={cut}   {w:9.2f}%   {s:11.2f}%   {u:7.2f}%   "
              f"{w - u:+10.2f} points")

    print("\n  The union is the number the paper has to report. A union close")
    print("  to the encyclopedia column means the fact is absent from both")
    print("  bodies of text and the silent box is a property of the graph.")

    if "relation" in out[0]:
        print("\nBY RELATION, at cut <= 2")
        by = defaultdict(list)
        for r in out:
            by[r["relation"]].append(r)
        print("  relation                      n   encyc.   second    union")
        for rel in sorted(by):
            v = by[rel]
            m = len(v)
            w = 100 * sum(1 for r in v if r["tier_wiki"] > 2) / m
            s = 100 * sum(1 for r in v if r["tier_second"] > 2) / m
            u = 100 * sum(1 for r in v if r["tier_union"] > 2) / m
            print(f"  {rel:<24} {m:>6,} {w:7.2f}% {s:7.2f}% {u:7.2f}%")
        print("\n  An asymmetry between relations that holds in the second")
        print("  column as well is not an artifact of one corpus either.")

    path = a.out or a.scored.replace(".csv", f"_{a.corpus}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {path}")
    print(f"Re-run four_boxes.py on it with --scored {path} after renaming")
    print("tier_union to tier, to get the four-box table against both corpora.")


if __name__ == "__main__":
    main()
