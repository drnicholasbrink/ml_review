"""PubMed E-utilities helpers extracted for Flask use."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import os
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CANONICAL_COLUMNS = ["RecordID", "PMID", "Title", "Abstract", "Authors", "Date", "Journal", "DOI"]


class PubMedResponseError(RuntimeError):
    """Raised when E-utilities responds successfully with an unusable payload."""


def _session() -> requests.Session:
    """Build a retrying E-utilities session for transient failures and rate limits."""

    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _base_params() -> dict[str, str]:
    params = {"tool": "ml_review"}
    if os.environ.get("NCBI_EMAIL", "").strip():
        params["email"] = os.environ["NCBI_EMAIL"].strip()
    return params


def count_pubmed(query: str, api_key: str = "", mindate: str | None = None, maxdate: str | None = None) -> int:
    """Return PubMed result count for a query and optional publication date range."""

    params = {**_base_params(), "db": "pubmed", "term": query, "rettype": "count", "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    if mindate and maxdate:
        params.update({"datetype": "pdat", "mindate": mindate, "maxdate": maxdate})
    response = _session().get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=60)
    response.raise_for_status()
    try:
        return int(response.json()["esearchresult"]["count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PubMedResponseError("PubMed returned an unexpected count response") from exc


def search_pubmed_ids(query: str, api_key: str = "", retmax: int = 500, mindate: str | None = None, maxdate: str | None = None) -> list[str]:
    """Return PubMed IDs for a query."""

    params = {**_base_params(), "db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}
    if api_key:
        params["api_key"] = api_key
    if mindate and maxdate:
        params.update({"datetype": "pdat", "mindate": mindate, "maxdate": maxdate})
    response = _session().get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=60)
    response.raise_for_status()
    try:
        return [str(item) for item in response.json()["esearchresult"].get("idlist", [])]
    except (KeyError, TypeError) as exc:
        raise PubMedResponseError("PubMed returned an unexpected search response") from exc


def fetch_pubmed_xml(pmids: list[str], api_key: str = "") -> str:
    """Fetch PubMed XML for a list of PMIDs."""

    if not pmids:
        return "<PubmedArticleSet />"
    params = {**_base_params(), "db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    response = _session().get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=120)
    response.raise_for_status()
    return response.text


def parse_pubmed_xml(xml_text: str) -> pd.DataFrame:
    """Parse PubMed XML into canonical review columns."""

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PubMedResponseError("PubMed returned malformed record XML") from exc
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = "".join(article.find(".//ArticleTitle").itertext()).strip() if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = ["".join(node.itertext()).strip() for node in article.findall(".//AbstractText")]
        journal = article.findtext(".//Journal/Title") or ""
        year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or ""
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            collective = author.findtext("CollectiveName") or ""
            name = collective or " ".join(part for part in [fore, last] if part)
            if name:
                authors.append(name)
        doi = ""
        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text or ""
                break
        records.append({"RecordID": pmid, "PMID": pmid, "Title": title, "Abstract": " ".join(abstract_parts), "Authors": "; ".join(authors), "Date": year, "Journal": journal, "DOI": doi})
    return pd.DataFrame(records, columns=CANONICAL_COLUMNS)


def fetch_pubmed_records(query: str, output_csv: Path, api_key: str = "", retmax: int = 500, mindate: str | None = None, maxdate: str | None = None) -> pd.DataFrame:
    """Search and fetch PubMed records into a CSV."""

    pmids = search_pubmed_ids(query, api_key=api_key, retmax=retmax, mindate=mindate, maxdate=maxdate)
    frames = []
    for start in range(0, len(pmids), 200):
        xml_text = fetch_pubmed_xml(pmids[start : start + 200], api_key=api_key)
        frames.append(parse_pubmed_xml(xml_text))
        progress = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
        progress.to_csv(output_csv, index=False)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
    result.to_csv(output_csv, index=False)
    return result
