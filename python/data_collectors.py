"""!
@brief Collects and consolidates the five inputs the solvers need (interactome, drug targets, AD module, candidate list, validation list), each from a live, cited, named source.
@see docs/modules/data-collectors.md
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import requests

from gene_ids import resolve_symbols

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
FROZEN_DIR = REPO_ROOT / "data" / "frozen"
MANIFEST_PATH = FROZEN_DIR / "manifest.json"


## @brief Typed shapes passed between collection, consolidation, and the solvers.
#  @see docs/modules/data-collectors.md

@dataclass
class Drug:
    drug_id: str
    name: str
    targets: list[str] = field(default_factory=list)  ##< Entrez gene IDs, in the interactome
    z_score: float | None = None                      ##< closeness to the AD module; None = not yet scored
    source: str = ""                                   ##< citation, e.g. "cummings2025pipeline"


@dataclass
class Instance:
    """!
    @brief What the four solvers actually see: n drugs, choose k, reward w, penalty P.
    @see docs/modules/data-collectors.md
    """
    id: str
    n: int
    k: int
    lam: float
    w: np.ndarray              ##< shape (n,) float32
    P: np.ndarray               ##< shape (n, n) float32, symmetric, zero diagonal
    drug_ids: list[str] = field(default_factory=list)
    sources: dict = field(default_factory=dict)  ##< file -> sha256, for provenance


def sha256_of(path: Path) -> str:
    """!
    @brief Hashes a file's contents.
    @param path File to hash.
    @return Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_source(name: str, path: Path, url: str, license_note: str,
                   manifest_path: Path = MANIFEST_PATH) -> dict:
    """!
    @brief Appends one entry to data/frozen/manifest.json.
    @param name Short, unique source identifier (e.g. "dgidb_donepezil").
    @param path Local file the source was written to.
    @param url Where the file was fetched from.
    @param license_note Attribution/license text for this source.
    @param manifest_path Manifest file to update.
    @return The manifest entry that was recorded.
    @see docs/modules/data-collectors.md
    """
    entry = {
        "name": name,
        "path": str(path.relative_to(REPO_ROOT)),
        "url": url,
        "sha256": sha256_of(path) if path.exists() else None,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license_note": license_note,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = [e for e in manifest["sources"] if e["name"] != name] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return entry


## @brief Curated substitute for the Bellenguez et al. 2022 GWAS gene table.
#  @see docs/modules/data-collectors.md
CANONICAL_AD_GENES = [
    "APP", "PSEN1", "PSEN2", "APOE", "MAPT",
    "TREM2", "CLU", "CR1", "PICALM", "BIN1",
    "ABCA7", "SORL1", "CD33", "MS4A6A", "EPHA1", "BCHE",
]


def collect_ad_module(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """!
    @brief Resolves CANONICAL_AD_GENES to Entrez Gene IDs via NCBI and caches the mapping.
    @param raw_dir Directory to cache the raw response in.
    @return Mapping of gene symbol to Entrez Gene ID (None where unresolved).
    @see docs/modules/data-collectors.md
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "ad_module.json"
    if not dest.exists():
        mapping = resolve_symbols(CANONICAL_AD_GENES)
        dest.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    mapping = json.loads(dest.read_text())
    record_source("ad_module", dest, url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                   license_note="NCBI Gene, public domain. Curated gene list -- see docs/modules/data-collectors.md.")
    return mapping


def collect_drug_targets(drugs: list[Drug], raw_dir: Path = RAW_DIR) -> list[Drug]:
    """!
    @brief Queries DGIdb for each drug's gene targets and fills in Drug.targets as Entrez IDs, in place.
    @param drugs Drugs to query; each one's .targets is mutated in place.
    @param raw_dir Directory to cache each drug's raw response in.
    @return The same drugs list, for chaining.
    @see docs/modules/data-collectors.md
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = "https://dgidb.org/api/graphql"

    for drug in drugs:
        dest = raw_dir / f"dgidb_{drug.name.lower()}.json"
        if not dest.exists():
            query = {"query": "query($name: String!) { drugs(names: [$name]) "
                               "{ nodes { name interactions { gene { name } } } } }",
                      "variables": {"name": drug.name}}
            resp = requests.post(url, json=query, timeout=15)
            resp.raise_for_status()
            dest.write_text(json.dumps(resp.json(), indent=2))

        payload = json.loads(dest.read_text())
        gene_symbols = sorted({
            interaction["gene"]["name"]
            for node in payload.get("data", {}).get("drugs", {}).get("nodes", [])
            for interaction in node.get("interactions", [])
        })
        symbol_to_id = resolve_symbols(gene_symbols)
        drug.targets = [entrez_id for entrez_id in symbol_to_id.values() if entrez_id]
        record_source(f"dgidb_{drug.name.lower()}", dest, url=url,
                       license_note="DGIdb, CC0 / public domain.")
    return drugs


def collect_interactome(gene_symbols: list[str], raw_dir: Path = RAW_DIR,
                         add_nodes: int = 10, required_score: int = 400) -> list[tuple[str, str]]:
    """!
    @brief Fetches a STRING-db subnetwork around gene_symbols and returns an (entrez_id, entrez_id) edge list.
    @param gene_symbols Seed genes (drug targets + AD module) the subnetwork is built around.
    @param raw_dir Directory to cache the raw STRING response in.
    @param add_nodes Extra interaction partners STRING adds for connectivity.
    @param required_score Minimum STRING confidence (0-1000) for an edge to be included.
    @return List of (entrez_id, entrez_id) edges.
    @see docs/modules/data-collectors.md
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "interactome_string.tsv"
    url = "https://string-db.org/api/tsv/network"
    if not dest.exists():
        params = {
            "identifiers": "\r".join(sorted(set(gene_symbols))),
            "species": 9606,  # Homo sapiens
            "add_nodes": add_nodes,
            "required_score": required_score,
        }
        resp = requests.post(url, data=params, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text)

    lines = dest.read_text().strip().splitlines()
    header = lines[0].split("\t")
    col_a, col_b = header.index("preferredName_A"), header.index("preferredName_B")
    symbol_pairs = [(row.split("\t")[col_a], row.split("\t")[col_b]) for row in lines[1:]]

    all_symbols = sorted({s for pair in symbol_pairs for s in pair})
    symbol_to_id = resolve_symbols(all_symbols)
    edges = [(symbol_to_id[a], symbol_to_id[b]) for a, b in symbol_pairs
             if symbol_to_id.get(a) and symbol_to_id.get(b)]

    record_source("interactome_string", dest, url=url,
                   license_note="STRING, CC BY 4.0. Subnetwork around the requested genes -- see docs/modules/data-collectors.md.")
    return edges


def collect_candidate_list() -> list[Drug]:
    """!
    @brief The curated AD candidate list (decision D1), each entry citing its source.
    @return Drugs with no targets/z_score yet -- fill those in with collect_drug_targets() and build_instance().
    @see docs/modules/data-collectors.md
    """
    return [
        Drug("DB00843", "donepezil", source="approved AD drug"),
        Drug("DB00989", "rivastigmine", source="approved AD drug"),
        Drug("DB00674", "galantamine", source="approved AD drug"),
        Drug("DB01043", "memantine", source="approved AD drug"),
        Drug("DB00820", "sildenafil", source="fang2021sildenafil (network-based repurposing hit)"),
    ]
    # TODO: extend from the full 2025/2026 pipeline review tables.


def collect_validation_list(raw_dir: Path = RAW_DIR, max_studies: int = 500) -> list[str]:
    """!
    @brief Live DRUG/BIOLOGICAL intervention names from ClinicalTrials.gov AD trials, for validation.py's literature cross-check.
    @param raw_dir Directory to cache the raw query results in.
    @param max_studies Roughly how many studies to page through before stopping.
    @return Sorted, deduplicated intervention names (Placebo excluded).
    @see docs/modules/data-collectors.md
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "clinicaltrials_ad_interventions.json"
    url = "https://clinicaltrials.gov/api/v2/studies"

    if not dest.exists():
        names: set[str] = set()
        page_token = None
        fetched = 0
        while fetched < max_studies:
            params = {"query.cond": "Alzheimer Disease", "pageSize": 100,
                      "fields": "NCTId,InterventionName,InterventionType"}
            if page_token:
                params["pageToken"] = page_token
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            for study in payload.get("studies", []):
                for interv in study.get("protocolSection", {}).get(
                        "armsInterventionsModule", {}).get("interventions", []):
                    if interv.get("type") in ("DRUG", "BIOLOGICAL") and interv.get("name"):
                        name = interv["name"].strip()
                        if name.lower() != "placebo":
                            names.add(name)
            fetched += len(payload.get("studies", []))
            page_token = payload.get("nextPageToken")
            if not page_token or not payload.get("studies"):
                break
        dest.write_text(json.dumps(sorted(names), indent=2))

    names = json.loads(dest.read_text())
    record_source("clinicaltrials_ad_interventions", dest, url=url,
                   license_note="ClinicalTrials.gov, US government work, public domain.")
    return names


def _degree_bins(graph: nx.Graph, n_bins: int = 5) -> dict[str, list[str]]:
    """!
    @brief Buckets graph nodes by degree, for the degree-matched random sampling in proximity_and_zscore().
    @param graph Interactome to bin nodes from.
    @param n_bins Number of degree buckets.
    @return Mapping of bin label to the nodes in it.
    @see docs/modules/data-collectors.md
    """
    nodes_by_degree = sorted(graph.nodes(), key=lambda v: graph.degree(v))
    bins: dict[str, list[str]] = {}
    chunk = max(1, len(nodes_by_degree) // n_bins)
    for i, node in enumerate(nodes_by_degree):
        bins.setdefault(str(i // chunk), []).append(node)
    return bins


def proximity_and_zscore(graph: nx.Graph, targets: list[str], ad_module: list[str],
                          rng: np.random.Generator, n_random: int = 200) -> tuple[float, float]:
    """!
    @brief Computes d_c(S, T_i) and z_i (formulation.tex) for one drug's targets against the AD module.
    @param graph Interactome (largest connected component).
    @param targets The drug's target genes (Entrez IDs), already restricted to nodes in graph.
    @param ad_module AD module genes (Entrez IDs), already restricted to nodes in graph.
    @param rng Random generator for the degree-matched null sample.
    @param n_random Number of random target sets to sample for mu/sigma.
    @return (d_c, z) -- the raw proximity and its standardized z-score.
    @see docs/modules/data-collectors.md
    """
    dist_from_ad = nx.multi_source_dijkstra_path_length(graph, ad_module)
    d_c = float(np.mean([dist_from_ad.get(t, np.nan) for t in targets]))

    bins = _degree_bins(graph)
    node_to_bin = {node: b for b, nodes in bins.items() for node in nodes}
    samples = []
    for _ in range(n_random):
        fake_targets = []
        for t in targets:
            pool = bins.get(node_to_bin.get(t), list(graph.nodes()))
            fake_targets.append(pool[rng.integers(len(pool))])
        samples.append(np.mean([dist_from_ad.get(t, np.nan) for t in fake_targets]))
    mu, sigma = float(np.mean(samples)), float(np.std(samples))
    z = (d_c - mu) / sigma if sigma > 0 else float("nan")
    return d_c, z


def separation(graph: nx.Graph, targets_a: list[str], targets_b: list[str]) -> float:
    """!
    @brief Computes s_AB (formulation.tex), the network separation between two drugs' target sets.
    @param graph Interactome (largest connected component).
    @param targets_a First drug's target genes (Entrez IDs).
    @param targets_b Second drug's target genes (Entrez IDs).
    @return s_AB; negative means the two target sets overlap in the network.
    @see docs/modules/data-collectors.md
    """
    def mean_min_within(group: list[str]) -> float:
        if len(group) <= 1:
            return 0.0
        dists = []
        for a in group:
            lengths = nx.single_source_shortest_path_length(graph, a)
            others = [lengths[b] for b in group if b != a and b in lengths]
            if others:
                dists.append(min(others))
        return float(np.mean(dists)) if dists else 0.0

    def mean_between(a_group: list[str], b_group: list[str]) -> float:
        total, count = 0.0, 0
        for a in a_group:
            lengths = nx.single_source_shortest_path_length(graph, a)
            reach = [lengths[b] for b in b_group if b in lengths]
            if reach:
                total += min(reach)
                count += 1
        for b in b_group:
            lengths = nx.single_source_shortest_path_length(graph, b)
            reach = [lengths[a] for a in a_group if a in lengths]
            if reach:
                total += min(reach)
                count += 1
        return total / count if count else float("nan")

    d_aa, d_bb = mean_min_within(targets_a), mean_min_within(targets_b)
    d_ab = mean_between(targets_a, targets_b)
    return d_ab - (d_aa + d_bb) / 2


def build_instance(drugs: list[Drug], graph_edges: list[tuple[str, str]], ad_module: list[str],
                    k: int, lam: float = 1.0, seed: int = 0, z_max: float = 10.0) -> Instance:
    """!
    @brief Consolidates everything collect_*() produced into the solver-ready Instance.
    @param drugs Candidate drugs, each with .targets filled in (collect_drug_targets()).
    @param graph_edges Interactome edges, (entrez_id, entrez_id) pairs (collect_interactome()).
    @param ad_module AD module genes, Entrez IDs (collect_ad_module()).
    @param k Combination size the solvers will choose.
    @param lam Overlap penalty weight (lambda in the objective).
    @param seed Random seed for the z-score null sampling.
    @param z_max Cap on an individual drug's reward before normalization.
    @return A solver-ready Instance, or raises ValueError if the result would be infeasible.
    @see docs/modules/data-collectors.md
    """
    rng = np.random.default_rng(seed)

    graph = nx.Graph()
    graph.add_edges_from(graph_edges)
    if graph.number_of_nodes() == 0:
        raise ValueError("empty interactome -- nothing to build an Instance from")
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()

    ad_module_in_graph = [g for g in ad_module if g in graph]
    if not ad_module_in_graph:
        raise ValueError("none of the AD module genes are in the interactome's largest component")

    eligible: list[Drug] = []
    z_scores: dict[str, float] = {}
    for drug in drugs:
        targets_in_graph = [t for t in drug.targets if t in graph]
        if not targets_in_graph:
            continue  # NOT in the solver input, per "no targets" eligibility rule
        _, z = proximity_and_zscore(graph, targets_in_graph, ad_module_in_graph, rng)
        if np.isfinite(z) and z < 0:
            drug.targets = targets_in_graph
            drug.z_score = z
            eligible.append(drug)
            z_scores[drug.drug_id] = z

    if not eligible:
        raise ValueError("no eligible drugs (need finite z < 0) -- check targets/interactome overlap")
    if len(eligible) < k:
        raise ValueError(f"only {len(eligible)} eligible drug(s) but k={k} -- no valid combination "
                          f"exists; loosen the interactome query or reduce k")

    n = len(eligible)
    raw_w = np.array([min(-z_scores[d.drug_id], z_max) for d in eligible], dtype=np.float64)
    w = (raw_w / raw_w.max()).astype(np.float32)

    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = separation(graph, eligible[i].targets, eligible[j].targets)
    P_raw = np.maximum(0.0, -S)
    P = (P_raw / P_raw.max() if P_raw.max() > 0 else P_raw).astype(np.float32)
    np.fill_diagonal(P, 0.0)

    return Instance(
        id=f"ad_combo_n{n}_k{k}_seed{seed}",
        n=n, k=k, lam=lam, w=w, P=P,
        drug_ids=[d.drug_id for d in eligible],
        sources={"ad_module_size": len(ad_module_in_graph), "graph_nodes": graph.number_of_nodes(),
                  "graph_edges": graph.number_of_edges(), "z_max": z_max, "seed": seed},
    )
