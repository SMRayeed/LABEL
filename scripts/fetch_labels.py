#!/usr/bin/env python3
"""Fetch FDA drug labels (openFDA) for every drug in the matched sample.

For each drug, the openFDA drug label endpoint is searched by generic name,
then by substance name, then by brand name. Up to --per-drug labels are kept
and their sections are pooled, so a drug sold by several manufacturers is
covered by all of them. One JSON file per drug is written to --cache:

    {"drug": ..., "found": true, "query": "openfda.generic_name", "n_labels": 3,
     "indications": "...", "contraindications": "...", "warnings": "...",
     "full": "..."}

"indications" is the Indications and Usage section, "contraindications" the
Contraindications section, "warnings" the Boxed Warning, Warnings, Warnings
and Precautions and Precautions sections, and "full" every text section.

Network manners: one request at a time, a pause between requests, retries
with backoff on 429 and 5xx. Without an API key openFDA allows 240 requests
per minute and 1,000 per day, which covers 574 drugs. Set OPENFDA_API_KEY to
raise the daily limit. Drugs already in the cache are skipped, so a run that
stops resumes where it left off.

    python fetch_labels.py --edges matched_edges.csv --cache label_cache
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.fda.gov/drug/label.json"
FIELDS_IND = ["indications_and_usage"]
FIELDS_CON = ["contraindications"]
FIELDS_WARN = ["boxed_warning", "warnings", "warnings_and_cautions",
               "precautions", "general_precautions"]
SKIP = {"openfda", "id", "set_id", "version", "effective_time", "spl_product_data_elements",
        "package_label_principal_display_panel", "spl_unclassified_section",
        "references", "how_supplied", "storage_and_handling"}


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150]


def get(url, tries=5):
    req = urllib.request.Request(url, headers={"User-Agent": "w3phiai-repro/1.0"})
    wait = 2.0
    for _ in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:            # openFDA answers "no match" with 404
                return None
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(wait)
                wait *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(wait)
            wait *= 2
    raise RuntimeError(f"gave up after {tries} tries: {url}")


def text_of(label, fields):
    out = []
    for f in fields:
        for chunk in label.get(f, []) or []:
            if isinstance(chunk, str):
                out.append(chunk)
    return "\n".join(out)


def full_text(label):
    out = []
    for k, v in label.items():
        if k in SKIP or k.endswith("_table"):
            continue
        if isinstance(v, list):
            out.extend(x for x in v if isinstance(x, str))
    return "\n".join(out)


def fetch_one(drug, per_drug, key, pause):
    q = drug.lower().replace('"', "")
    for field in ("openfda.generic_name", "openfda.substance_name", "openfda.brand_name"):
        params = {"search": f'{field}:"{q}"', "limit": str(per_drug)}
        if key:
            params["api_key"] = key
        url = API + "?" + urllib.parse.urlencode(params)
        js = get(url)
        time.sleep(pause)
        res = (js or {}).get("results") or []
        if res:
            seen = set()
            parts = {"indications": [], "contraindications": [], "warnings": [], "full": []}
            for lab in res:
                for name, fields in (("indications", FIELDS_IND),
                                     ("contraindications", FIELDS_CON),
                                     ("warnings", FIELDS_WARN)):
                    t = text_of(lab, fields)
                    if t and (name, t) not in seen:
                        seen.add((name, t))
                        parts[name].append(t)
                t = full_text(lab)
                if t and ("full", t) not in seen:
                    seen.add(("full", t))
                    parts["full"].append(t)
            return {"drug": drug, "found": True, "query": field, "n_labels": len(res),
                    **{k: "\n\n".join(v) for k, v in parts.items()}}
    return {"drug": drug, "found": False, "query": "", "n_labels": 0,
            "indications": "", "contraindications": "", "warnings": "", "full": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", required=True, help="matched_edges.csv (column 'source')")
    ap.add_argument("--cache", default="label_cache")
    ap.add_argument("--per-drug", type=int, default=5)
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="first N drugs only (testing)")
    a = ap.parse_args()

    os.makedirs(a.cache, exist_ok=True)
    drugs = sorted({r["source"] for r in csv.DictReader(open(a.edges, encoding="utf-8"))})
    if a.limit:
        drugs = drugs[:a.limit]
    key = os.environ.get("OPENFDA_API_KEY", "").strip()
    todo = [d for d in drugs if not os.path.exists(os.path.join(a.cache, safe(d) + ".json"))]
    print(f"{len(drugs):,} drugs, {len(drugs) - len(todo):,} cached, {len(todo):,} to fetch"
          f"{' (API key set)' if key else ''}", file=sys.stderr)
    for i, d in enumerate(todo, 1):
        try:
            rec = fetch_one(d, a.per_drug, key, a.pause)
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: {type(e).__name__}: {e}; skipped, rerun to retry", file=sys.stderr)
            continue
        with open(os.path.join(a.cache, safe(d) + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        if i % 50 == 0 or i == len(todo):
            print(f"  {i:,}/{len(todo):,}", file=sys.stderr)

    found = 0
    for d in drugs:
        p = os.path.join(a.cache, safe(d) + ".json")
        if os.path.exists(p) and json.load(open(p, encoding="utf-8")).get("found"):
            found += 1
    print(f"labels found for {found:,} of {len(drugs):,} drugs "
          f"({100 * found / max(1, len(drugs)):.1f}%)")
    missing = len(drugs) - sum(os.path.exists(os.path.join(a.cache, safe(d) + ".json")) for d in drugs)
    if missing:
        print(f"{missing:,} drugs failed to fetch; run the same command again", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
