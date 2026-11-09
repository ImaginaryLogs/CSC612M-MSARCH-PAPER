"""!
@brief Runs data collection, then each solver module, then validation, then benchmark analysis, in sequence.
@see docs/modules/orchestrator.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

import data_collectors as dc
import benchmark_analysis as ba
from gene_ids import symbol_of
from module_interface import ModuleConfig, RunResultsMessage, run_module
from validation import Validator


@dataclass
class OrchestratorConfig:
    platforms: list[str] = field(default_factory=lambda: ["scalar", "avx2", "cuda", "qaoa"])
    k: int = 3
    lam: float = 1.0
    seed: int = 1234
    chains: int = 4
    sweeps: int = 1000
    T0: float | None = None  ##< start temperature; required by the "scalar" and "avx2" platforms
    T1: float | None = None  ##< end temperature; T0 >= T1 > 0
    brute_force_check: bool = False
    synthetic_n: int | None = None  ##< solve a random instance of this many drugs instead of the real data


class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        """!
        @brief Builds the coordinator for one pipeline run.
        @param config Platforms, instance size, and solver settings for this run.
        """
        self.config = config
        self.validator = Validator(brute_force=config.brute_force_check)

    def collect_data(self) -> tuple[list[dc.Drug], list[tuple[str, str]], list[str]]:
        """!
        @brief Stage 1: gathers inputs; returns (drugs, interactome_edges, ad_module_entrez_ids).
        @return Drugs, interactome edges, and AD module Entrez IDs for build_instance().
        @see docs/modules/orchestrator.md
        """
        drugs = dc.collect_candidate_list()
        dc.collect_drug_targets(drugs)
        ad_module = dc.collect_ad_module()
        ad_module_ids = [entrez_id for entrez_id in ad_module.values() if entrez_id]

        target_ids = sorted({t for d in drugs for t in d.targets})
        target_symbols = [s for s in (symbol_of(tid) for tid in target_ids) if s]
        seed_genes = sorted(set(dc.CANONICAL_AD_GENES) | set(target_symbols))
        edges = dc.collect_interactome(seed_genes)

        return drugs, edges, ad_module_ids

    def build_instance(self, drugs: list[dc.Drug], edges: list[tuple[str, str]],
                        ad_module_ids: list[str]) -> dc.Instance:
        """!
        @brief Stage 2: consolidates collected data into the problem the solvers see.
        @param drugs Candidate drugs with targets filled in (from collect_data()).
        @param edges Interactome edges (from collect_data()).
        @param ad_module_ids AD module Entrez IDs (from collect_data()).
        @return A solver-ready Instance.
        @see docs/modules/orchestrator.md
        """
        return dc.build_instance(drugs, edges, ad_module_ids, self.config.k, self.config.lam,
                                  seed=self.config.seed)

    def run_modules(self, instance: dc.Instance) -> list[RunResultsMessage]:
        """!
        @brief Stage 3: runs every requested platform on the same instance.
        @param instance Problem instance every platform solves.
        @return One RunResultsMessage per platform in self.config.platforms.
        @see docs/modules/orchestrator.md
        """
        results = []
        for platform in self.config.platforms:
            module_config = ModuleConfig(platform=platform, seed=self.config.seed,
                                          chains=self.config.chains, sweeps=self.config.sweeps,
                                          T0=self.config.T0, T1=self.config.T1)
            result = run_module(instance, module_config)
            results.append(result)
        return results

    def validate(self, instance: dc.Instance,
                 results: list[RunResultsMessage]) -> list[RunResultsMessage]:
        """!
        @brief Stage 4: grades every result against the independent judge (and the brute-force oracle, if enabled).
        @param instance Problem instance the results were produced from.
        @param results Raw results from run_modules().
        @return The same results, with e_judge/valid (and extra["optimal"] if enabled) filled in.
        @see docs/modules/orchestrator.md
        """
        graded = [self.validator.grade(instance, r) for r in results]
        for r in graded:
            ba.append_run(r)
        return graded

    def analyze(self):
        """!
        @brief Stage 5: loads everything written so far and summarizes timing.
        @return The speedup table of every run recorded so far (benchmark_analysis.speedup_table(),
                scalar as the baseline).
        @see docs/modules/orchestrator.md
        """
        return ba.speedup_table(ba.load_runs())

    def run(self) -> list[RunResultsMessage]:
        """!
        @brief Runs the full pipeline: collect, consolidate, solve, validate. With
               config.synthetic_n set, stages 1-2 are replaced by a random instance of that
               size (data_collectors.synthetic_instance(); no download).
        @return Graded RunResultsMessages, one per requested platform.
        @see docs/modules/orchestrator.md
        """
        if self.config.synthetic_n is not None:
            instance = dc.synthetic_instance(self.config.synthetic_n, self.config.k,
                                             self.config.lam, seed=self.config.seed)
        else:
            drugs, edges, ad_module_ids = self.collect_data()
            instance = self.build_instance(drugs, edges, ad_module_ids)
        results = self.run_modules(instance)
        return self.validate(instance, results)
