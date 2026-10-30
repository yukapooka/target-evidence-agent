"""PubMed wrappers (NCBI E-utilities): literature search, abstracts, and PMID existence checks.

Objective:
    Give the agents literature evidence for the "Literature support" and "Mechanistic plausibility" rubric
    criteria, and let the critic check that every cited PMID is real.

Functions:
    search_pubmed()    search text -> list of PMIDs, most relevant first        (esearch, JSON)
    fetch_abstracts()  PMIDs -> Article objects with title, abstract, journal, year  (efetch, XML)
    check_pmids()      PMIDs -> {pmid: True/False} for whether each one exists  (esummary, JSON)

Rationale:
    - Uses get_json/get_text from http.py, so answers are cached and the NCBI rate limit is respected.
    - api_key/tool/email are sent with every request (NCBI asks for them); the cache strips them before saving.
    - efetch has no JSON option, so abstracts are parsed from XML with the standard library (no new dependency).
    - A fake PMID still returns HTTP 200 from esummary, with an "error" field: that is how check_pmids spots it.
"""

import os
import xml.etree.ElementTree as ET

from pydantic import BaseModel

from agent.tools.http import get_json, get_text

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# Parameters NCBI wants on every request. Empty values are dropped (e.g. no API key -> no api_key param).
# Out: {"db": "pubmed", "tool": "target-evidence-agent", "email": "...", "api_key": "..."}
def _base_params() -> dict:
    params = {
        "db": "pubmed",
        "tool": "target-evidence-agent",
        "email": os.getenv("NCBI_EMAIL"),
        "api_key": os.getenv("NCBI_API_KEY"),
    }
    return {k: v for k, v in params.items() if v}


# --- Step 7a: literature search ---
# PMIDs matching a search, most relevant first.
# In:  search_pubmed("IL4R atopic dermatitis", max_results=5)
# Out: ["38186219", "39080841", "33483337", "31505066", "26454361"]

def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed and return PMIDs, most relevant first."""
    data = get_json(EUTILS + "esearch.fcgi", {**_base_params(), "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"})
    return data["esearchresult"]["idlist"]


# --- Step 7b: PMID existence check (used by the critic on Day 3) ---
# esummary answers HTTP 200 even for fake PMIDs, but marks them with an "error" key.
# In:  check_pmids(["26454361", "99999999999"])
# Out: {"26454361": True, "99999999999": False}

def check_pmids(pmids: list[str]) -> dict[str, bool]:
    """For each PMID, True if PubMed has a record for it, False otherwise."""
    data = get_json(EUTILS + "esummary.fcgi", {**_base_params(), "id": ",".join(pmids), "retmode": "json"})
    results = {}
    for pmid in pmids:
        record = data["result"].get(pmid, {})
        results[pmid] = "error" not in record
    return results


# --- Step 7c: abstracts ---
# Title, abstract, journal and year for each PMID, parsed from efetch XML.
# Structured abstracts keep their section labels ("BACKGROUND: ... METHODS: ..."); no abstract -> "".
# In:  fetch_abstracts(["26454361", "31505066"])
# Out: [Article(pmid="26454361", title="Efficacy and safety of dupilumab ...", journal="Lancet (London, England)",
#               year="2016", abstract="BACKGROUND: Data from early-stage studies ..."), Article(pmid="31505066", ...)]

class Article(BaseModel):
    pmid : str
    title : str
    abstract : str
    journal : str
    year : str | None = None

def fetch_abstracts(pmids: list[str]) -> list[Article]:
    """Fetch and parse PubMed records for the given PMIDs (one batched request)."""
    xml_text = get_text(EUTILS + "efetch.fcgi", {**_base_params(), "id": ",".join(pmids),  "rettype": "abstract", "retmode": "xml"})
    root = ET.fromstring(xml_text)

    articles = []
    for item in root.findall("PubmedArticle"):
        pmid = item.findtext("MedlineCitation/PMID")
        title = "".join(item.find("MedlineCitation/Article/ArticleTitle").itertext())
        journal = item.findtext("MedlineCitation/Article/Journal/Title")
        year =  item.findtext("MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
        sections = []
        for part in item.findall("MedlineCitation/Article/Abstract/AbstractText"):
            text = "".join(part.itertext())
            label = part.get("Label")
            sections.append(f"{label}: {text}" if label else text)
        articles.append(Article(pmid=pmid, title=title, abstract=" ".join(sections), journal=journal, year=year))
    return articles

