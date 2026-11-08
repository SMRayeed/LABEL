#!/usr/bin/env python3
"""Build a small synthetic stand-in for every input the pipeline downloads.

Used by `bash run_all.sh --smoke`. It writes files with the same layout as
the real ones -- a PrimeKG-style kg.csv, MONDO- and DOID-style .obo files,
Hetionet node and edge files -- and fills the web caches (Wikipedia
articles, redirect names, PubMed abstracts, candidate articles) so that no
step needs the network. The numbers that come out mean nothing; the test
is that every step runs and writes what the next one reads.

    python3 make_fixture.py --data smoke/data --work smoke/work --terms "a|b"
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import name_tiers as nt          # noqa: E402

WORDS = ["kelv", "brant", "mosk", "duran", "pell", "varn", "trisk", "olm",
         "zand", "ferro", "glim", "hask", "lorn", "quell", "rusk", "sorb"]
GROUPS = ["vascular", "metabolic", "neural"]
FILLER = ("This medicine is described in pharmacology references. Its "
          "chemistry, dosage forms and history are covered in several "
          "sections of the article. ")


def tree():
    """Root -> 3 groups -> 2 families each -> 6 leaves each."""
    nodes = [("disease", None)]
    fams, leaves = [], []
    w = iter(WORDS)
    for g in GROUPS:
        gname = f"{g} system disease"
        nodes.append((gname, "disease"))
        for _ in range(2):
            stem = next(w)
            fam = f"{stem}osis"
            nodes.append((fam, gname))
            fams.append(fam)
            for k in range(6):
                leaf = f"{stem}osis type {chr(65 + k)}{chr(65 + k)}{k + 1}x"
                nodes.append((leaf, fam))
                leaves.append(leaf)
    return nodes, fams, leaves


def write_obo(path, nodes, prefix, pad):
    ids = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("format-version: 1.2\ndata-version: fixture\n\n")
        for i, (name, _p) in enumerate(nodes, start=1):
            ids[name] = f"{prefix}:{i:0{pad}d}" if pad else f"{prefix}:{i}"
        for name, parent in nodes:
            fh.write(f"[Term]\nid: {ids[name]}\nname: {name}\n")
            if parent:
                fh.write(f"is_a: {ids[parent]} ! {parent}\n")
            fh.write("\n")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--terms", default="")
    ap.add_argument("--drugs", type=int, default=90)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.data, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)

    nodes, fams, leaves = tree()
    parent = {n: p for n, p in nodes}
    mondo = write_obo(os.path.join(a.data, "mondo.obo"), nodes, "MONDO", 7)
    doid = write_obo(os.path.join(a.data, "doid.obo"), nodes, "DOID", 0)
    num = {n: str(int(mondo[n].split(":")[1])) for n in mondo}

    drugs = [f"Fixtamab{i:03d}" for i in range(a.drugs)]
    cols = ["relation", "display_relation", "x_index", "x_id", "x_type",
            "x_name", "x_source", "y_index", "y_id", "y_type", "y_name",
            "y_source"]
    rows = []

    def edge(rel, disp, xi, xt, xn, xs, yi, yt, yn, ys):
        rows.append(dict(zip(cols, [rel, disp, "0", xi, xt, xn, xs, "0", yi, yt, yn, ys])))
        rows.append(dict(zip(cols, [rel, disp, "0", yi, yt, yn, ys, "0", xi, xt, xn, xs])))

    # disease_disease parent-child, both directions as PrimeKG stores it
    for n, p in nodes:
        if p:
            edge("disease_disease", "parent-child", num[n], "disease", n, "MONDO",
                 num[p], "disease", p, "MONDO")

    text_bits = {d: [] for d in drugs}
    pm_bits = {d: [] for d in drugs}
    rels = {}
    for i, d in enumerate(drugs):
        did = f"DB{i:05d}"
        pick = rng.sample(leaves, 9)
        ind, con, off = pick[:4], pick[4:8], pick[8:]
        rels[d] = (ind, con, off)
        for rel, tg in (("indication", ind), ("contraindication", con),
                        ("off-label use", off)):
            for t in tg:
                edge(rel, rel, did, "drug", d, "DrugBank", num[t], "disease", t, "MONDO")
                u = rng.random()
                # indications named more often than contraindications
                named_p = 0.45 if rel == "indication" else 0.2
                if u < named_p:
                    text_bits[d].append(f"{d} is used in {t}. ")
                elif u < named_p + 0.25:
                    text_bits[d].append(f"{d} is relevant to {parent[t]}. ")
                if rng.random() < named_p:
                    pm_bits[d].append(f"In patients with {t}, {d} was studied. ")

    with open(os.path.join(a.data, "kg.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Wikipedia articles for drugs, long enough for the 3000-character floor
    wiki = os.path.join(a.work, "wiki_cache")
    for d in drugs:
        body = (FILLER * 30) + "".join(text_bits[d]) + (FILLER * 5)
        nt.write_article(wiki, d, f"'''{d}''' is a drug. " + body)

    # redirect names for every disease: one lay name for some leaves
    names = os.path.join(a.work, "gbif_cache")
    for n, _p in nodes:
        vern = [n.replace("type", "form")] if rng.random() < 0.3 else []
        nt.write_names(names, "redir", n, {"synonyms": [], "vernacular": vern,
                                           "ancestors": []})

    # PubMed abstracts: the single-query corpus and the balanced one
    pm = os.path.join(a.work, "pubmed_cache")
    os.makedirs(pm, exist_ok=True)
    terms = [t.strip() for t in a.terms.split("|") if t.strip()]
    suffixes = [""]
    if terms:
        h = hashlib.sha1("|".join(terms).encode("utf-8")).hexdigest()[:8]
        suffixes.append(f"__q{h}")
    for d in drugs:
        doc = ("PMID 1. Abstract. " + FILLER * 4 + "".join(pm_bits[d]))
        for s in suffixes:
            with open(os.path.join(pm, f"{nt._safe(d)}{s}.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(doc)

    # candidate disease articles, naming some drugs that treat them
    cand = os.path.join(a.work, "candidate_wiki_cache")
    for n, _p in nodes:
        who = [d for d in drugs if n in rels[d][0] and rng.random() < 0.5]
        nt.write_article(cand, n, f"'''{n}''' is a disease. " + FILLER * 8
                         + "".join(f"It is treated with {d}. " for d in who))

    # PubMed pair and popularity counts in candidate_check.py's query format
    try:
        import candidate_check as cc
    except ImportError:
        cc = None
    if cc is not None:
        with open(os.path.join(a.work, "pubmed_pairs.jsonl"), "w", encoding="utf-8") as fh:
            for n, _p in nodes:
                t = cc.wiki_title(n).replace(chr(34), "")
                fh.write(json.dumps({"q": f'"{t}"[tiab]', "n": rng.randint(5, 5000)}) + "\n")
                for d in drugs:
                    for k, rel in enumerate(("indication", "contraindication", "off-label use")):
                        q = f'"{d}"[tiab] AND "{t}"[tiab]'
                        qual = cc.PM_QUAL.get(rel, "")
                        q += f" AND {qual}" if qual else ""
                        hit = n in rels[d][k]
                        fh.write(json.dumps({"q": q, "n": rng.randint(1, 30) if hit and rng.random() < 0.6 else 0}) + "\n")

    # Hetionet: same drugs, DOID-coded diseases, treats and palliates
    with open(os.path.join(a.data, "hetionet-v1.0-nodes.tsv"), "w", encoding="utf-8") as fh:
        fh.write("id\tname\tkind\n")
        for i, d in enumerate(drugs):
            fh.write(f"Compound::DB{i:05d}\t{d}\tCompound\n")
        for n in leaves:
            fh.write(f"Disease::{doid[n]}\t{n}\tDisease\n")
    with gzip.open(os.path.join(a.data, "hetionet-v1.0-edges.sif.gz"), "wt",
                   encoding="utf-8") as fh:
        fh.write("source\tmetaedge\ttarget\n")
        for i, d in enumerate(drugs):
            ind, con, off = rels[d]
            for t in ind[:2]:
                fh.write(f"Compound::DB{i:05d}\tCtD\tDisease::{doid[t]}\n")
            for t in ind[2:]:
                fh.write(f"Compound::DB{i:05d}\tCpD\tDisease::{doid[t]}\n")

    print(f"fixture: {len(drugs)} drugs, {len(leaves)} leaf diseases, "
          f"{len(rows):,} kg rows")


if __name__ == "__main__":
    main()
