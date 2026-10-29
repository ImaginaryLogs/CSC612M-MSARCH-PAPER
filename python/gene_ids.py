"""!
@brief Resolves HGNC gene symbols to Entrez Gene IDs via NCBI, cached to disk so every collector agrees on the same IDs.
@see docs/modules/gene-ids.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "data" / "raw" / "gene_ids_cache.json"

EUTILS_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_RATE_LIMIT_SECONDS = 0.35  # ~3 req/s, polite for the unauthenticated eutils limit


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolve_symbol(symbol: str, cache: dict | None = None) -> str | None:
    """!
    @brief One gene symbol -> its Entrez Gene ID, or None if NCBI found nothing.
    @param symbol HGNC gene symbol, e.g. "APOE".
    @param cache Shared cache dict to read/write; loads and saves its own if omitted.
    @return Entrez Gene ID as a string, or None if unresolved.
    @see docs/modules/gene-ids.md
    """
    own_cache = cache is None
    if own_cache:
        cache = _load_cache()

    key = symbol.upper()
    if key in cache:
        return cache[key]

    params = {"db": "gene", "term": f"{symbol}[sym] AND Homo sapiens[orgn]", "retmode": "json"}
    resp = requests.get(EUTILS_ESEARCH, params=params, timeout=10)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    entrez_id = ids[0] if ids else None
    cache[key] = entrez_id
    time.sleep(NCBI_RATE_LIMIT_SECONDS)

    if own_cache:
        _save_cache(cache)
    return entrez_id


def symbol_of(entrez_id: str, cache: dict | None = None) -> str | None:
    """!
    @brief Reverse lookup: Entrez ID -> gene symbol, from the already-populated cache.
    @param entrez_id Entrez Gene ID to look up.
    @param cache Shared cache dict to search; loads the on-disk cache if omitted.
    @return The gene symbol, or None if entrez_id isn't in the cache.
    @see docs/modules/gene-ids.md
    """
    cache = cache if cache is not None else _load_cache()
    for symbol, resolved_id in cache.items():
        if resolved_id == entrez_id:
            return symbol
    return None


def resolve_symbols(symbols: list[str]) -> dict[str, str | None]:
    """!
    @brief Batch version of resolve_symbol(), sharing one cache load/save across all lookups.
    @param symbols HGNC gene symbols to resolve.
    @return Mapping of each uppercased symbol to its Entrez Gene ID (None where unresolved).
    @see docs/modules/gene-ids.md
    """
    cache = _load_cache()
    result = {}
    for symbol in symbols:
        result[symbol.upper()] = resolve_symbol(symbol, cache=cache)
    _save_cache(cache)
    return result
