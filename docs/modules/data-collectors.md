# `python/data_collectors.py`

Collects the five kinds of input this project needs (see `docs/paper/01_drug_candidacy_proposal/appendices/data_architecture.tex`) and writes them to `data/` as CSV/JSON, with a SHA-256 manifest recording where every file came from. Nothing here is guessed or hand-typed: a value is either fetched live from a cited, named API, or an explicit, documented curated choice — never a silently made-up number.

1. **interactome** — protein-protein interaction graph (STRING API, restricted to the genes this run actually needs)
2. **drug_targets** — which proteins each drug acts on (DGIdb API)
3. **ad_module** — the Alzheimer's disease gene set (curated list, NCBI-verified Entrez IDs)
4. **candidate_list** — the drugs solvers choose k from (curated)
5. **validation_list** — AD clinical-trial drugs, for the literature cross-check in `validation.py` (NOT part of the solver input — used only to sanity-check results afterwards). Sourced live from ClinicalTrials.gov, per the sister proposal's explicit requirement that this list "must be verified ... before any result is reported."

Every collector caches its raw response under `data/raw/` and skips re-fetching if that file already exists, so repeated runs are fast and don't hammer a shared public API. Delete the relevant `data/raw/` file to force a refresh.

## What this does NOT reproduce

The full Cheng et al. 2019 / BioGRID genome-wide interactome (hundreds of thousands of edges, a specific supplementary-file download) or the Bellenguez et al. 2022 GWAS gene table (behind a journal supplement). Those are the "real" biological-scale sources named in the proposal; what's here is a live, correctly-cited, smaller substitute — a STRING subnetwork and a curated canonical gene list — sized so the collect-and-consolidate pipeline can actually be run and checked end-to-end. Swapping in the full sources later only touches these functions, nothing downstream.

## `Drug` / `Instance` dataclasses

The shapes everything downstream (`module_interface.py`, `validation.py`) reads. Plain dataclasses — see `RunResultsMessage` in `module_interface.py` for why: one typed object instead of loose dicts passed between stages. `Instance` is what the four solvers actually see: n drugs, choose k, reward w, penalty P. Everything biological is a label attached to these numbers (`methodology.tex`, "The solvers never see biology").

- `Drug.targets` — Entrez gene IDs, in the interactome
- `Drug.z_score` — closeness to the AD module; `None` = not yet scored
- `Drug.source` — citation, e.g. `"cummings2025pipeline"`
- `Instance.w` — shape `(n,)` float32
- `Instance.P` — shape `(n, n)` float32, symmetric, zero diagonal
- `Instance.sources` — file -> sha256, for provenance

## `record_source()`

Appends one entry to `data/frozen/manifest.json`. Called by every `collect_*()` function once it has a file on disk.

## `CANONICAL_AD_GENES`

The three early-onset causal genes, APOE (the major late-onset risk gene), MAPT, and the most robustly replicated late-onset GWAS risk genes across the AD genetics literature (the kind of set Bellenguez et al. 2022's top loci and earlier GWAS consistently name). BCHE is included because it's a direct drug target of several candidates (butyrylcholinesterase, alongside ACHE). This is a documented, cited substitute for the full Bellenguez supplementary table — see "What this does NOT reproduce" above.

## `collect_ad_module()`

Resolves `CANONICAL_AD_GENES` to Entrez Gene IDs via NCBI (`gene_ids.py`) and writes `{symbol: entrez_id}` to `data/raw/ad_module.json`. Returns that same mapping. Symbols NCBI can't resolve are kept with a `None` value and excluded by the caller, never silently dropped here.

## `collect_drug_targets()`

Queries DGIdb's GraphQL API for each drug's known gene targets, resolves each gene symbol to an Entrez ID, and fills in `Drug.targets` in place. Writes the raw per-drug response to `data/raw/` for provenance and skips re-querying a drug whose raw file already exists.

## `collect_interactome()`

Fetches a STRING-db subnetwork covering `gene_symbols` (drug targets + AD module, from the caller) plus `add_nodes` extra interaction partners STRING adds for connectivity, at a medium confidence cutoff (`required_score=400` of 1000, STRING's own default). Returns an edge list of `(entrez_id, entrez_id)` pairs. This is a documented substitute for the full Cheng et al./BioGRID interactome — see "What this does NOT reproduce" above.

## `collect_candidate_list()`

The curated AD candidate list (decision D1): the 4 approved symptomatic AD drugs, plus repurposing candidates named in the 2025/2026 Cummings pipeline reviews, plus published AD network-study hits. This part IS real (not a TODO) — it's hand-curated by design, each entry citing its source; it still needs real target/z-score data from `collect_drug_targets()` before it can be scored. TODO: extend from the full 2025/2026 pipeline review tables.

## `collect_validation_list()`

Drug/biological intervention names from real Alzheimer's trials on ClinicalTrials.gov, for `validation.py`'s literature cross-check — NOT fed into the solvers. This satisfies the sister proposal's requirement that this list "must be verified against ClinicalTrials.gov ... before any result is reported" by construction: it IS the ClinicalTrials.gov query result, not a hand-typed list later checked against it. Keeps only interventions of type DRUG or BIOLOGICAL (excludes devices, behavioral arms, diagnostics, and "Placebo").

## `_degree_bins()`

Nodes grouped into `n_bins` buckets of similar degree, for the degree-matched random sampling in `proximity_and_zscore()` (`formulation.tex`: "mu_i, sigma_i come from degree-matched random target sets").

## `proximity_and_zscore()`

`d_c(S, T_i)` and `z_i` from `formulation.tex`: average shortest-path distance from each target to the nearest AD-module gene, standardized against degree-matched random target sets of the same size.

## `separation()`

`s_AB` from `formulation.tex`. `<d_AA>` is set to 0 when `|A|=1` (the proposal's documented convention for single-target drugs).

## `build_instance()`

Consolidates everything `collect_*()` produced into the `Instance` the four solvers read. Eligibility: finite z < 0 only, nothing imputed (`formulation.tex`). Every number here is either measured (from the graph) or an explicit, logged assumption (`z_max`, `seed`) — see `data_architecture.tex`'s "no hand-assigned biological numbers" rule. Builds a NetworkX graph from the given edges, restricts to the largest connected component, filters drugs to those with a finite negative z-score, and raises a clear `ValueError` rather than silently producing an infeasible instance if fewer than `k` drugs are eligible.
