#!/usr/bin/env python3
"""
Harvest research papers from free public APIs (no keys required).

Sources:
  - OpenAlex          (general, all disciplines)
  - Europe PMC        (field-specific: biomedical + preprints)
  - Unpaywall status  (via OpenAlex open_access flag)

Reads queries from data/queries.json, deduplicates (DOI -> fuzzy title),
writes results to data/papers.sqlite and data/papers.json.

Runs on a laptop or inside GitHub Actions. Network required at runtime.
"""

import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "papers.sqlite"
JSON_PATH = DATA_DIR / "papers.json"
QUERIES_PATH = DATA_DIR / "queries.json"

# Polite pool: OpenAlex asks for a contact email in the User-Agent / mailto.
CONTACT_EMAIL = "you@example.com"  # <-- change to your email (improves rate limits)
USER_AGENT = f"metaresearch-harvester (mailto:{CONTACT_EMAIL})"

PER_PAGE = 50  # results per source per query


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


# ---------- OpenAlex (general tab) ----------
def harvest_openalex(query, limit=PER_PAGE):
    q = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works?search={q}"
        f"&per-page={limit}&mailto={CONTACT_EMAIL}"
    )
    out = []
    try:
        data = _get(url)
    except Exception as e:
        print(f"  [openalex] error: {e}")
        return out
    for w in data.get("results", []):
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        oa = w.get("open_access", {}) or {}
        primary = w.get("primary_location") or {}
        src = (primary.get("source") or {}).get("display_name")
        out.append({
            "source_db": "openalex",
            "tab": "general",
            "id": w.get("id"),
            "doi": doi,
            "title": w.get("display_name"),
            "year": w.get("publication_year"),
            "venue": src,
            "cited_by": w.get("cited_by_count", 0),
            "is_oa": bool(oa.get("is_oa")),
            "oa_url": oa.get("oa_url"),
            "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
            "is_paid_hint": not bool(oa.get("is_oa")),
        })
    return out


def _reconstruct_abstract(inv):
    if not inv:
        return None
    positions = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


# ---------- Europe PMC (field-specific tab) ----------
def harvest_europepmc(query, limit=PER_PAGE):
    q = urllib.parse.quote(query)
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={q}&format=json&pageSize={limit}&resultType=core"
    )
    out = []
    try:
        data = _get(url)
    except Exception as e:
        print(f"  [europepmc] error: {e}")
        return out
    for r in data.get("resultList", {}).get("result", []):
        is_oa = r.get("isOpenAccess") == "Y"
        out.append({
            "source_db": "europepmc",
            "tab": "field",
            "id": f"EPMC:{r.get('id')}",
            "doi": r.get("doi"),
            "title": r.get("title"),
            "year": int(r["pubYear"]) if r.get("pubYear", "").isdigit() else None,
            "venue": r.get("journalTitle"),
            "cited_by": r.get("citedByCount", 0),
            "is_oa": is_oa,
            "oa_url": None,
            "abstract": r.get("abstractText"),
            "is_paid_hint": not is_oa,
        })
    return out


# ---------- Dedup ----------
def dedupe(papers):
    kept = []
    seen_doi = {}
    for p in papers:
        doi = (p.get("doi") or "").lower().strip() or None
        if doi and doi in seen_doi:
            _merge(kept[seen_doi[doi]], p)
            continue
        match_idx = None
        nt = _norm_title(p.get("title"))
        if nt:
            for i, k in enumerate(kept):
                if abs((p.get("year") or 0) - (k.get("year") or 0)) <= 1:
                    if SequenceMatcher(None, nt, _norm_title(k["title"])).ratio() > 0.92:
                        match_idx = i
                        break
        if match_idx is not None:
            _merge(kept[match_idx], p)
        else:
            if doi:
                seen_doi[doi] = len(kept)
            kept.append(dict(p))
    return kept


def _merge(base, other):
    base.setdefault("also_in", [])
    if other["source_db"] not in base["also_in"] and other["source_db"] != base["source_db"]:
        base["also_in"].append(other["source_db"])
    if not base.get("abstract") and other.get("abstract"):
        base["abstract"] = other["abstract"]
    if not base.get("doi") and other.get("doi"):
        base["doi"] = other["doi"]
    base["is_oa"] = base.get("is_oa") or other.get("is_oa")


# ---------- Persist ----------
def write_db(papers):
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS papers")
    con.execute("""
        CREATE TABLE papers (
            id TEXT, doi TEXT, title TEXT, year INTEGER, venue TEXT,
            cited_by INTEGER, is_oa INTEGER, oa_url TEXT, abstract TEXT,
            source_db TEXT, tab TEXT, is_paid_hint INTEGER, also_in TEXT
        )
    """)
    for p in papers:
        con.execute(
            "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.get("id"), p.get("doi"), p.get("title"), p.get("year"),
             p.get("venue"), p.get("cited_by", 0), int(bool(p.get("is_oa"))),
             p.get("oa_url"), p.get("abstract"), p.get("source_db"),
             p.get("tab"), int(bool(p.get("is_paid_hint"))),
             ",".join(p.get("also_in", []))),
        )
    con.commit()
    con.close()


def main():
    queries = json.loads(QUERIES_PATH.read_text())
    all_papers = []
    for entry in queries.get("searches", []):
        q = entry["query"]
        tabs = entry.get("tabs", ["general", "field"])
        print(f"Query: {q!r}  tabs={tabs}")
        if "general" in tabs:
            all_papers += harvest_openalex(q)
            time.sleep(0.3)
        if "field" in tabs:
            all_papers += harvest_europepmc(q)
            time.sleep(0.3)

    print(f"Raw records: {len(all_papers)}")
    deduped = dedupe(all_papers)
    print(f"After dedup: {len(deduped)}")

    write_db(deduped)
    paid = [p for p in deduped if p.get("is_paid_hint")]
    payload = {
        "generated_query_count": len(queries.get("searches", [])),
        "total": len(deduped),
        "free_open_access": sum(1 for p in deduped if p.get("is_oa")),
        "paid_or_closed_hint": len(paid),
        "note": (
            "Analysis uses only open-access papers. Closed/paid papers are listed "
            "as coverage hints; Scopus/Web of Science/Embase may add further coverage."
        ),
        "papers": deduped,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {JSON_PATH} and {DB_PATH}")


if __name__ == "__main__":
    main()
