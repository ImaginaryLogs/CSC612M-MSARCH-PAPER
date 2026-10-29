# `python/gene_ids.py`

Resolves HGNC gene symbols (e.g. "APOE") to Entrez Gene IDs (e.g. "348") via NCBI's eutils esearch. Every Instance's drug_ids/targets/AD module use Entrez IDs (`data_architecture.tex`'s `targets.csv` schema), and this is the one place that does the symbol -> ID lookup, so every collector agrees on the same IDs for the same gene.

Cached to disk (`data/raw/gene_ids_cache.json`) because NCBI is a shared, rate-limited public service: repeated test runs should hit the cache, not re-query. Unresolved symbols are recorded as `None`, not silently dropped, so a caller can see what failed to resolve.

## `resolve_symbol()`

One gene symbol -> its Entrez Gene ID, or `None` if NCBI found nothing for `"<symbol>[sym] AND Homo sapiens[orgn]"`.

## `symbol_of()`

Reverse lookup: Entrez ID -> gene symbol, from whatever is already in the cache (every ID in an `Instance` got there via `resolve_symbol(s)`, so this is always a cache hit, never a fresh API call).

## `resolve_symbols()`

Batch version: shares one cache load/save and one rate-limit delay sequence across all lookups, so this is cheap to call repeatedly from different collectors without re-touching disk per symbol.
