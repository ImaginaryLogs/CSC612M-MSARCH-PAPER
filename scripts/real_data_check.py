"""!
@brief Builds a real AD drug-combination instance with more drugs than the 5-drug pipeline list,
       checks scalar SA against the exact solvers on it, and compares the top-ranked combinations
       with combinations known from the literature and from ClinicalTrials.gov.

The pipeline's candidate list (data_collectors.collect_candidate_list) gives only n = 3 eligible
drugs. This script starts from 38 AD-relevant drugs (approved, agitation, and repurposing-trial
drugs), runs them through the SAME collectors and build_instance() as the pipeline, and keeps its
own cache and provenance manifest so it never touches the pipeline's data:

    data/raw/extended/                      DGIdb / NCBI / ClinicalTrials.gov downloads (git-ignored)
    data/raw/extended/<variant>/            STRING network + built instance for this target set
    data/frozen/manifest_extended.json      provenance (url, sha256, fetched_at) of every download
    data/real_runs.jsonl                    one record per solver run (git-ignored)
    data/real_data_report.md                the report printed at the end (git-ignored)

The first run downloads everything (a few minutes, mostly NCBI's 3 requests/second limit) and
builds the instance (a few minutes, network separation for every drug pair). Later runs reuse the
cache and take seconds.

Usage (from the repository root, after scripts/build_native.sh):
    uv run python scripts/real_data_check.py
    uv run python scripts/real_data_check.py --drop-pk-genes   # ignore CYP/transporter "targets"

@see docs/real-data-check.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import python.benchmark_analysis as ba                                              # noqa: E402
import python.data_collectors as dc                                                 # noqa: E402
from python.gene_ids import _load_cache as load_gene_cache                          # noqa: E402
from python.module_interface import ModuleConfig, RunResultsMessage, run_module     # noqa: E402
from python.validation import Validator, judge                                      # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "extended"
MANIFEST = REPO_ROOT / "data" / "frozen" / "manifest_extended.json"
RUNS_PATH = REPO_ROOT / "data" / "real_runs.jsonl"
REPORT_PATH = REPO_ROOT / "data" / "real_data_report.md"

## (DGIdb query name, mechanism class, why it is a candidate). drug_id is the name: these are
#  labels only, targets are always looked up by name.
CANDIDATES = [
    ("donepezil", "cholinesterase inhibitor", "approved AD drug"),
    ("rivastigmine", "cholinesterase inhibitor", "approved AD drug"),
    ("galantamine", "cholinesterase inhibitor", "approved AD drug"),
    ("tacrine", "cholinesterase inhibitor", "first approved AD drug, withdrawn (hepatotoxicity)"),
    ("huperzine", "cholinesterase inhibitor", "huperzine A, AD trials"),
    ("memantine", "NMDA antagonist", "approved AD drug"),
    ("brexpiprazole", "atypical antipsychotic", "approved for AD agitation (2023)"),
    ("citalopram", "SSRI", "AD agitation trials (CitAD)"),
    ("escitalopram", "SSRI", "AD agitation trials (S-CitAD)"),
    ("dextromethorphan", "NMDA / sigma-1", "AD agitation (AXS-05 component)"),
    ("bupropion", "NDRI / CYP2D6 inhibitor", "AD agitation (AXS-05 component)"),
    ("sildenafil", "PDE5 inhibitor", "network-medicine repurposing hit (Fang et al. 2021)"),
    ("metformin", "biguanide", "repurposing trials (MAP)"),
    ("semaglutide", "GLP-1 agonist", "phase 3 AD trials (evoke)"),
    ("liraglutide", "GLP-1 agonist", "AD trials (ELAD)"),
    ("pioglitazone", "PPAR-gamma agonist", "AD prevention trial (TOMMORROW)"),
    ("rosiglitazone", "PPAR-gamma agonist", "AD trials (REFLECT)"),
    ("losartan", "angiotensin receptor blocker", "vascular risk trials (rrAD, RADAR)"),
    ("telmisartan", "angiotensin receptor blocker", "AD repurposing trials"),
    ("valsartan", "angiotensin receptor blocker", "AD repurposing candidate"),
    ("amlodipine", "calcium channel blocker", "vascular risk trials (rrAD)"),
    ("nilvadipine", "calcium channel blocker", "AD phase 3 (NILVAD)"),
    ("atorvastatin", "statin", "vascular risk trials (rrAD)"),
    ("simvastatin", "statin", "AD trials (CLASP)"),
    ("bumetanide", "loop diuretic", "APOE4 transcriptomic repurposing hit (Taubes et al. 2021)"),
    ("levetiracetam", "antiepileptic", "AD hyperexcitability trials"),
    ("valproic", "antiepileptic / HDAC inhibitor", "AD agitation/prevention trials"),
    ("lithium", "mood stabilizer / GSK3 inhibitor", "AD and MCI trials"),
    ("nilotinib", "tyrosine kinase inhibitor", "AD phase 2"),
    ("dasatinib", "tyrosine kinase inhibitor (senolytic)", "senolytic trial with quercetin (SToMP-AD)"),
    ("quercetin", "flavonoid (senolytic)", "senolytic trial with dasatinib (SToMP-AD)"),
    ("masitinib", "tyrosine kinase inhibitor", "phase 3 add-on to ChEI/memantine (AB09004)"),
    ("baricitinib", "JAK inhibitor", "AD neuroinflammation trials"),
    ("ibuprofen", "NSAID", "AD prevention trials"),
    ("naproxen", "NSAID", "AD prevention trials (ADAPT, INTREPAD)"),
    ("celecoxib", "NSAID (COX-2)", "AD prevention trial (ADAPT)"),
    ("minocycline", "tetracycline antibiotic", "AD trial (MADE)"),
    ("doxycycline", "tetracycline antibiotic", "AD trial (with rifampin)"),
]
MECHANISM = {name: cls for name, cls, _ in CANDIDATES}

## Combinations with a documented clinical or trial history, and the expected direction:
#  "plausible" = used together or tested together on purpose; "redundant" = same mechanism, used
#  here as a negative control (the model should NOT favour them).
KNOWN_COMBOS = [
    (
        ("donepezil", "memantine"), 
        "plausible",
        "FDA-approved fixed-dose combination (Namzaric, Dec 2014) for moderate-severe AD"
    ), (("rivastigmine", "memantine"), 
        "plausible",
     "ChEI + memantine dual therapy; EFNS-ENS/EAN 2015 guideline, mixed evidence by outcome"),
    (("galantamine", "memantine"), "plausible",
     "ChEI + memantine dual therapy; EFNS-ENS/EAN 2015 guideline, mixed evidence by outcome"),
    (("dextromethorphan", "bupropion"), "plausible",
     "AXS-05 for AD agitation (ADVANCE-1, ACCORD); bupropion mainly boosts dextromethorphan via CYP2D6"),
    (("dasatinib", "quercetin"), "plausible",
     "senolytic combination, phase 1 feasibility trial in mild AD (Gonzales et al., Nat Med 2023; NCT04063124)"),
    (("losartan", "amlodipine", "atorvastatin"), "plausible",
     "intensive vascular risk reduction arm of rrAD (NCT02913664); no published efficacy result found"),
    (("masitinib", "donepezil"), "plausible",
     "masitinib tested as add-on to ChEI and/or memantine (AB09004, Dubois et al. 2023)"),
    (("masitinib", "memantine"), "plausible",
     "masitinib tested as add-on to ChEI and/or memantine (AB09004, Dubois et al. 2023)"),
    (("donepezil", "rivastigmine"), "redundant", "two cholinesterase inhibitors: same mechanism"),
    (("donepezil", "galantamine"), "redundant", "two cholinesterase inhibitors: same mechanism"),
    (("rivastigmine", "galantamine"), "redundant", "two cholinesterase inhibitors: same mechanism"),
    (("donepezil", "tacrine"), "redundant", "two cholinesterase inhibitors: same mechanism"),
    (("losartan", "telmisartan"), "redundant", "two angiotensin receptor blockers: same mechanism"),
    (("atorvastatin", "simvastatin"), "redundant", "two statins: same mechanism"),
]

## Brand / code names in ClinicalTrials.gov intervention names that stand for our drugs.
TRIAL_ALIASES = {
    "namzaric": ["donepezil", "memantine"],
    "aricept": ["donepezil"], "exelon": ["rivastigmine"], "razadyne": ["galantamine"],
    "reminyl": ["galantamine"], "namenda": ["memantine"], "ebixa": ["memantine"],
    "axs-05": ["dextromethorphan", "bupropion"], "auvelity": ["dextromethorphan", "bupropion"],
    "rexulti": ["brexpiprazole"], "rybelsus": ["semaglutide"], "ozempic": ["semaglutide"],
    "victoza": ["liraglutide"], "actos": ["pioglitazone"], "avandia": ["rosiglitazone"],
    "lipitor": ["atorvastatin"], "zocor": ["simvastatin"], "keppra": ["levetiracetam"],
    "depakote": ["valproic"], "valproate": ["valproic"], "divalproex": ["valproic"],
    "tasigna": ["nilotinib"], "sprycel": ["dasatinib"], "celebrex": ["celecoxib"],
    "viagra": ["sildenafil"], "revatio": ["sildenafil"],
}

## Genes DGIdb lists as "interactions" that are about drug handling, not mechanism.
PK_GENE = re.compile(r"^(CYP\d|ABC[A-G]\d|SLC\d|SLCO\d|UGT\d|FMO\d|NAT[12]$|ALB$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-data SA check and literature comparison")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4],
                        help="team sizes to solve and report (n // 3 is always added for the timing check)")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234, help="z-score sampling and SA seed")
    parser.add_argument("--drop-pk-genes", action="store_true",
                        help="remove CYP/ABC/SLC/UGT/FMO/NAT/ALB genes from every drug's targets")
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=0.01)
    parser.add_argument("--sa-seeds", type=int, default=20, help="SA runs per sweep budget (timing check)")
    parser.add_argument("--sweeps", type=int, nargs="+", default=[10, 30, 100, 300, 1000])
    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# Data: drugs -> targets -> network -> Instance
# --------------------------------------------------------------------------------------------

def gene_symbols_by_id() -> dict[str, str]:
    return {entrez: symbol for symbol, entrez in load_gene_cache().items() if entrez}


def collect(args: argparse.Namespace) -> tuple[list[dc.Drug], dict[str, dict]]:
    """!
    @brief DGIdb targets for every candidate (Entrez IDs), optionally without PK genes.
    @return The drugs, and per-drug notes (target counts) for the eligibility table.
    """
    drugs = [dc.Drug(drug_id=name, name=name, source=why) for name, _, why in CANDIDATES]
    dc.collect_drug_targets(drugs, raw_dir=RAW_DIR, manifest_path=MANIFEST)
    symbols = gene_symbols_by_id()
    notes = {}
    for drug in drugs:
        before = len(drug.targets)
        if args.drop_pk_genes:
            drug.targets = [t for t in drug.targets if not PK_GENE.match(symbols.get(t, ""))]
        notes[drug.name] = {"dgidb_targets": before, "targets_used": len(drug.targets)}
    return drugs, notes


def build(args: argparse.Namespace, drugs: list[dc.Drug]) -> tuple[dc.Instance, Path]:
    """!
    @brief Network + Instance for this target set, cached under data/raw/extended/<variant>/.
    @details The STRING download depends on the seed genes, and collect_interactome() reuses its
             cache file whatever genes are asked for, so each target set gets its own folder and
             the seed list is stored next to it and checked.
    """
    variant = "no_pk_genes" if args.drop_pk_genes else "all_targets"
    folder = RAW_DIR / variant
    folder.mkdir(parents=True, exist_ok=True)

    ad_module = dc.collect_ad_module(raw_dir=RAW_DIR, manifest_path=MANIFEST)
    ad_ids = [entrez for entrez in ad_module.values() if entrez]
    symbols = gene_symbols_by_id()
    target_symbols = {symbols[t] for d in drugs for t in d.targets if t in symbols}
    seeds = sorted(set(dc.CANONICAL_AD_GENES) | target_symbols)

    # Compare Entrez IDs, not symbols: the shared gene-ID cache can hold several symbols (aliases)
    # for one ID, so the symbol picked for an ID may change as other runs add to the cache.
    seed_ids = sorted({t for d in drugs for t in d.targets} | set(ad_ids))
    seed_file = folder / "interactome_seed_ids.json"
    if seed_file.exists() and json.loads(seed_file.read_text()) != seed_ids:
        raise RuntimeError(f"{folder} was built for a different gene set; delete the folder to rebuild")
    seed_file.write_text(json.dumps(seed_ids))

    cached = folder / f"instance_seed{args.seed}.npz"
    if cached.exists():
        data = np.load(cached, allow_pickle=False)
        names = [str(x) for x in data["names"]]
        z = dict(zip(names, data["z"].tolist()))
        for d in drugs:
            d.z_score = z.get(d.name)
        inst = dc.Instance(id="", n=len(names), k=0, lam=args.lam, w=data["w"], P=data["P"],
                           drug_ids=names, sources=json.loads(str(data["sources"])))
        return inst, folder

    edges = dc.collect_interactome(seeds, raw_dir=folder, manifest_path=MANIFEST)
    started = time.perf_counter()
    inst = dc.build_instance(drugs, edges, ad_ids, k=2, lam=args.lam, seed=args.seed)
    inst.sources["build_seconds"] = round(time.perf_counter() - started, 1)
    z = [next(d.z_score for d in drugs if d.drug_id == name) for name in inst.drug_ids]
    np.savez(cached, names=np.array(inst.drug_ids), w=inst.w, P=inst.P, z=np.array(z),
             sources=json.dumps(inst.sources))
    return inst, folder


def with_k(base: dc.Instance, k: int, seed: int) -> dc.Instance:
    return dc.Instance(id=f"real_extended_n{base.n}_k{k}_seed{seed}", n=base.n, k=k, lam=base.lam,
                       w=base.w, P=base.P, drug_ids=base.drug_ids, sources=base.sources)


# --------------------------------------------------------------------------------------------
# Solving: SA vs exact
# --------------------------------------------------------------------------------------------

def solve_all(base: dc.Instance, args: argparse.Namespace) -> tuple[list[dict], pd.DataFrame]:
    """!
    @brief For every k: exact_cpu, Python brute force (when small) and scalar SA, graded by judge().
    @return One summary row per k, and every run record (also appended to data/real_runs.jsonl).
    """
    k_timing = max(2, base.n // 3)
    ks = sorted(set(args.ks) | {k_timing})
    records: list[RunResultsMessage] = []
    summary = []
    for k in ks:
        inst = with_k(base, k, args.seed)
        exact = run_module(inst, ModuleConfig("exact_cpu"))
        e_opt = exact.e_solver
        python_ok = None
        if math.comb(base.n, k) <= 200_000:
            python_ok = abs(Validator().brute_force(inst).e_solver - e_opt) < 1e-6
        sa = run_module(inst, ModuleConfig("scalar", seed=args.seed, chains=4, sweeps=1000,
                                           T0=args.t0, T1=args.t1))
        for r in (exact, sa):
            Validator().grade(inst, r)
            r.extra.update({"n": base.n, "k": k, "e_optimal": e_opt})
            records.append(r)
        if k == k_timing:  # repeated SA runs at several budgets, for time to answer
            for sweeps in args.sweeps:
                for s in range(args.sa_seeds):
                    r = run_module(inst, ModuleConfig("scalar", seed=s, chains=1, sweeps=sweeps,
                                                      T0=args.t0, T1=args.t1))
                    Validator().grade(inst, r)
                    r.extra.update({"n": base.n, "k": k, "e_optimal": e_opt})
                    records.append(r)
        summary.append({
            "k": k, "teams": math.comb(base.n, k),
            "exact_best": [base.drug_ids[i] for i in exact.best], "e_exact": e_opt,
            "sa_best": [base.drug_ids[i] for i in sa.best], "e_sa": sa.e_judge,
            "sa_matches": abs(sa.e_judge - e_opt) < 1e-6, "python_matches": python_ok,
            "t_exact": exact.t_kernel, "t_sa": sa.t_kernel, "sa_valid": sa.valid,
        })
    for r in records:
        ba.append_run(r, RUNS_PATH)
    return summary, pd.DataFrame([asdict(r) for r in records])


# --------------------------------------------------------------------------------------------
# Literature: ClinicalTrials.gov co-occurrence and curated combinations
# --------------------------------------------------------------------------------------------

def trial_studies() -> list[dict]:
    """!
    @brief Every AD study on ClinicalTrials.gov with its intervention names (cached).
    """
    dest = RAW_DIR / "clinicaltrials_ad_studies.json"
    url = "https://clinicaltrials.gov/api/v2/studies"
    
    if not dest.exists():
        studies, token = [], None
        while True:
            params = {"query.cond": "Alzheimer Disease", "pageSize": 1000,
                      "fields": "NCTId,BriefTitle,InterventionName,InterventionType,Phase,OverallStatus"}
            if token:
                params["pageToken"] = token
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            for s in payload.get("studies", []):
                ps = s.get("protocolSection", {})
                studies.append({
                    "nct": ps.get("identificationModule", {}).get("nctId"),
                    "title": ps.get("identificationModule", {}).get("briefTitle", ""),
                    "phases": ps.get("designModule", {}).get("phases", []),
                    "interventions": [i.get("name", "") for i in
                                      ps.get("armsInterventionsModule", {}).get("interventions", [])
                                      if i.get("type") in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT",
                                                           "DIETARY_SUPPLEMENT")],
                })
            token = payload.get("nextPageToken")
            if not token:
                break
        dest.write_text(json.dumps(studies, indent=1))
    dc.record_source("clinicaltrials_ad_studies", dest, url=url, license_note="ClinicalTrials.gov, US government work, public domain.", manifest_path=MANIFEST)
    return json.loads(dest.read_text())


def drugs_in_study(study: dict, names: list[str]) -> set[str]:
    text = " | ".join(study["interventions"]).lower()
    found = {n for n in names if re.search(rf"\b{re.escape(n)}", text)}
    for alias, targets in TRIAL_ALIASES.items():
        if alias in text:
            found.update(t for t in targets if t in names)
    return found


def pair_trials(studies: list[dict], names: list[str]) -> dict[tuple[str, str], list[str]]:
    """!
    @brief NCT IDs of the AD studies whose interventions include both drugs of a pair.
    """
    out: dict[tuple[str, str], list[str]] = {}
    for s in studies:
        present = sorted(drugs_in_study(s, names))
        for a, b in itertools.combinations(present, 2):
            out.setdefault((a, b), []).append(s["nct"])
    return out


def ranked_teams(inst: dc.Instance, k: int) -> pd.DataFrame:
    """!
    @brief Every k-team with its energy, reward and penalty, best first (rank 1).
    """
    rows = []
    for team in itertools.combinations(range(inst.n), k):
        reward = float(sum(inst.w[i] for i in team))
        penalty = float(sum(inst.P[a, b] for a, b in itertools.combinations(team, 2)))
        rows.append({"team": tuple(inst.drug_ids[i] for i in team), "energy": -reward + inst.lam * penalty,
                     "reward": reward, "penalty": penalty})
    df = pd.DataFrame(rows).sort_values("energy", kind="stable").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["percentile"] = 100.0 * df["rank"] / len(df)  # 1 % = among the best 1 %
    return df


def auc_better(ranks_a: list[int], ranks_b: list[int]) -> float:
    """!
    @brief Probability that a random member of a ranks better (lower) than a random member of b.
    """
    if not ranks_a or not ranks_b:
        return math.nan
    wins = sum((ra < rb) + 0.5 * (ra == rb) for ra in ranks_a for rb in ranks_b)
    return wins / (len(ranks_a) * len(ranks_b))


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

def markdown_table(df: pd.DataFrame) -> str:
    """!
    @brief A DataFrame as a GitHub markdown table (floats to 4 significant digits, NaN blank).
    """
    if df.empty:
        return "(none)"

    def cell(value) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.4g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main() -> None:
    args = parse_args()
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    def table(df: pd.DataFrame) -> None:
        out(markdown_table(df))
        out()

    drugs, notes = collect(args)
    base, folder = build(args, drugs)
    eligible = set(base.drug_ids)

    out(f"# Real-data check ({'without' if args.drop_pk_genes else 'with'} PK genes, seed {args.seed})")
    out()
    out(f"Network: {base.sources.get('graph_nodes')} genes, {base.sources.get('graph_edges')} interactions "
        f"(STRING, confidence >= 400, largest component); AD module genes in it: "
        f"{base.sources.get('ad_module_size')} of {len(dc.CANONICAL_AD_GENES)}.")
    out()
    out(f"## Eligible drugs: n = {base.n} of {len(CANDIDATES)}")
    out()
    rows = []
    for d in drugs:
        if d.name in eligible:
            status = "eligible"
        elif notes[d.name]["targets_used"] == 0:
            status = "no targets"
        elif d.z_score is None:
            status = "not closer to AD module than random (z >= 0) or no targets in network"
        else:
            status = "excluded"
        idx = base.drug_ids.index(d.name) if d.name in eligible else None
        rows.append({"drug": d.name, "class": MECHANISM[d.name], "DGIdb targets": notes[d.name]["dgidb_targets"],
                     "targets used": notes[d.name]["targets_used"],
                     "z": d.z_score if d.name in eligible else math.nan,
                     "w": float(base.w[idx]) if idx is not None else math.nan, "status": status})
    table(pd.DataFrame(rows).sort_values(["status", "w"], ascending=[True, False]))

    summary, runs = solve_all(base, args)
    out("## SA vs exact")
    out()
    s = pd.DataFrame(summary)
    s["t_exact (ms)"] = s.pop("t_exact") * 1e3
    s["t_sa (ms)"] = s.pop("t_sa") * 1e3
    s["exact_best"] = s["exact_best"].map(" + ".join)
    s["sa_best"] = s["sa_best"].map(" + ".join)
    table(s)
    k_timing = max(2, base.n // 3)
    timed = runs[runs["instance_id"].str.contains(f"_k{k_timing}_")]
    out(f"Time to answer with 99% probability at k = {k_timing} ({args.sa_seeds} single-chain SA runs "
        f"per sweep budget; exact_cpu is one run):")
    out()
    table(ba.time_to_answer(timed))
    out("Speedup over exact_cpu (> 1: faster than exact search):")
    out()
    table(ba.speedup_tts(timed, baseline="exact_cpu"))

    studies = trial_studies()
    co = pair_trials(studies, base.drug_ids)
    pairs = ranked_teams(base, 2)
    pairs["trials"] = pairs["team"].map(lambda t: len(co.get(tuple(sorted(t)), [])))

    out("## Known combinations from the literature")
    out()
    rows = []
    for team, expected, why in KNOWN_COMBOS:
        missing = [d for d in team if d not in eligible]
        if missing:
            rows.append({"combination": " + ".join(team), "expected": expected, "rank": "",
                         "percentile": math.nan, "penalty": math.nan, "note": f"not eligible: {', '.join(missing)}",
                         "source": why})
            continue
        ranked = ranked_teams(base, len(team))
        row = ranked[ranked["team"].map(lambda t: set(t) == set(team))].iloc[0]
        rows.append({"combination": " + ".join(team), "expected": expected,
                     "rank": f"{row['rank']} / {len(ranked)}", "percentile": row["percentile"],
                     "penalty": row["penalty"], "note": "", "source": why})
    known = pd.DataFrame(rows)
    table(known)
    plaus = known[(known["expected"] == "plausible") & known["percentile"].notna()]["percentile"]
    redun = known[(known["expected"] == "redundant") & known["percentile"].notna()]["percentile"]
    out(f"Median percentile: plausible {plaus.median():.1f} (n={len(plaus)}), "
        f"redundant {redun.median():.1f} (n={len(redun)}). Lower is better; 50 = no better than random.")
    out()
    out("Sensitivity to lambda (the overlap penalty weight). Percentiles are among all teams of the "
        "same size; the best team of each size is shown:")
    out()
    rows = []
    for lam in (0.5, 1.0, 2.0, 3.0, 5.0):
        scaled = dc.Instance(id="", n=base.n, k=2, lam=lam, w=base.w, P=base.P, drug_ids=base.drug_ids)
        pct = {"plausible": [], "redundant": []}
        for team, expected, _ in KNOWN_COMBOS:
            if all(d in eligible for d in team):
                ranked = ranked_teams(scaled, len(team))
                pct[expected].append(float(ranked[ranked["team"].map(lambda t: set(t) == set(team))]
                                           ["percentile"].iloc[0]))
        rows.append({"lambda": lam,
                     "plausible median percentile": float(np.median(pct["plausible"])) if pct["plausible"] else math.nan,
                     "redundant median percentile": float(np.median(pct["redundant"])) if pct["redundant"] else math.nan,
                     "best pair": " + ".join(ranked_teams(scaled, 2)["team"].iloc[0]),
                     "best triple": " + ".join(ranked_teams(scaled, 3)["team"].iloc[0])})
    table(pd.DataFrame(rows))

    tested = pairs[pairs["trials"] > 0]["rank"].tolist()
    untested = pairs[pairs["trials"] == 0]["rank"].tolist()
    out("## ClinicalTrials.gov: pairs tested together in an AD study")
    out()
    out("Caution: a pair counts as co-tested when both drugs are listed among one study's interventions. "
        "That includes head-to-head comparisons (e.g. donepezil vs rivastigmine) and background therapy, "
        "not only deliberate combinations, so this is a weak signal.")
    out()
    out(f"{len(studies)} AD studies scanned; {len(tested)} of {len(pairs)} eligible pairs appear together "
        f"in at least one study's interventions. Probability that a co-tested pair ranks better than an "
        f"untested pair: {auc_better(tested, untested):.2f} (0.5 = no relation).")
    out()
    shown = pairs[pairs["trials"] > 0].copy()
    shown["team"] = shown["team"].map(" + ".join)
    shown["studies"] = [", ".join(co[tuple(sorted(t.split(' + ')))][:3]) +
                        (" ..." if len(co[tuple(sorted(t.split(' + ')))]) > 3 else "")
                        for t in shown["team"]]
    table(shown[["rank", "percentile", "team", "energy", "penalty", "trials", "studies"]])

    out("## Top-ranked combinations")
    out()
    for k in (2, 3):
        top = ranked_teams(base, k).head(10).copy()
        top["classes"] = top["team"].map(lambda t: " + ".join(MECHANISM[d] for d in t))
        if k == 2:
            top["trials"] = top["team"].map(lambda t: len(co.get(tuple(sorted(t)), [])))
        top["team"] = top["team"].map(" + ".join)
        out(f"k = {k}:")
        out()
        table(top.drop(columns=["percentile"]))

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"report written to {REPORT_PATH.relative_to(REPO_ROOT)}; runs appended to "
          f"{RUNS_PATH.relative_to(REPO_ROOT)}; cache in {folder.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
