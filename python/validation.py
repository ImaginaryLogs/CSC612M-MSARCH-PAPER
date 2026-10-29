"""!
@brief Grades solver output: judge() for every run, Validator(brute_force=True) for the true optimum, literature_crosscheck() for exploratory context.
@see docs/modules/validation.md
"""
from __future__ import annotations

from itertools import combinations

from data_collectors import Instance, collect_validation_list
from module_interface import RunResultsMessage


def judge(instance: Instance, selection: list[int]) -> float:
    """!
    @brief E(selection) = -sum(w_i) + lambda * sum_{i<j in selection} P_ij.
    @param instance Problem instance selection is drawn from.
    @param selection Drug indices (into instance.w/instance.P) to score.
    @return The energy of that selection, independent of any solver.
    @see docs/modules/validation.md
    """
    sel = list(selection)
    reward = sum(float(instance.w[i]) for i in sel)
    penalty = sum(float(instance.P[i, j]) for a, i in enumerate(sel) for j in sel[a + 1:])
    return -reward + instance.lam * penalty


class Validator:
    """!
    @brief Grades RunResultsMessages; pass brute_force=True to also compute the true optimum.
    @see docs/modules/validation.md
    """

    def __init__(self, brute_force: bool = False):
        """!
        @brief Configures how results get graded.
        @param brute_force Also compute the true optimum via brute_force() for comparison.
        """
        self.brute_force_enabled = brute_force

    def grade(self, instance: Instance, result: RunResultsMessage,
              tolerance: float = 1e-3) -> RunResultsMessage:
        """!
        @brief Fills in e_judge/valid on result -- a module never grades its own answer.
        @param instance Problem instance result was produced from.
        @param result Solver output to grade; mutated and returned.
        @param tolerance Max allowed gap between e_judge and e_solver to count as valid.
        @return The same result, with e_judge/valid (and extra["optimal"] if enabled) filled in.
        @see docs/modules/validation.md
        """
        e_judge = judge(instance, result.best)
        result.e_judge = e_judge
        result.valid = abs(e_judge - result.e_solver) <= tolerance
        if self.brute_force_enabled:
            best = self.brute_force(instance)
            result.extra["optimal"] = best.best
            result.extra["e_optimal"] = best.e_solver
            result.extra["is_optimal"] = sorted(result.best) == sorted(best.best)
        return result

    def brute_force(self, instance: Instance) -> RunResultsMessage:
        """!
        @brief The exact solver: tries every k-combination, keeps the best.
        @param instance Problem instance to solve exactly.
        @return The optimal RunResultsMessage (platform="exact_python"); raises if n < k.
        @see docs/modules/validation.md
        """
        if instance.n < instance.k:
            raise ValueError(f"infeasible instance: n={instance.n} < k={instance.k}, "
                              "no k-combination exists")
        best_energy, best_sel, ties = None, None, 0
        for sel in combinations(range(instance.n), instance.k):
            e = judge(instance, list(sel))
            if best_energy is None or e < best_energy:
                best_energy, best_sel, ties = e, list(sel), 1
            elif e == best_energy:
                ties += 1
        return RunResultsMessage(
            instance_id=instance.id, platform="exact_python", seed=0, chains=0, sweeps=0,
            best=best_sel, e_solver=best_energy, e_judge=best_energy, valid=True, ties=ties,
        )


def literature_crosscheck(selected_drug_names: list[str]) -> dict:
    """!
    @brief Flags which selected drugs are already known from AD clinical trials. Exploratory only.
    @param selected_drug_names Drug names (not IDs) to check, e.g. from a chosen combination.
    @return Dict with "known", "novel", and an explanatory "note".
    @see docs/modules/validation.md
    """
    trial_drugs = {name.lower() for name in collect_validation_list()}  # TODO: raises until collected
    known = [name for name in selected_drug_names if name.lower() in trial_drugs]
    novel = [name for name in selected_drug_names if name.lower() not in trial_drugs]
    return {"known": known, "novel": novel,
            "note": "Exploratory only -- see limitations.tex."}
