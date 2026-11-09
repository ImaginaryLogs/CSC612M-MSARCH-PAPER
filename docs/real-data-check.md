# Real-data check: more drugs, SA vs exact, and a literature comparison

The pipeline's own candidate list (5 drugs) leaves only n = 3 eligible drugs, which is too small to test simulated annealing. `scripts/real_data_check.py` starts from **38 AD-relevant drugs**, runs them through the same collectors and `build_instance()` as the pipeline, checks `scalar` SA against the exact solvers, and compares the model's rankings with combinations known from the literature and from ClinicalTrials.gov.

Results below are from 2026-11-09 (seed 1234, λ = 1, `T0 = 1`, `T1 = 0.01`). Timings are from one machine.

## Running it

```sh
bash scripts/build_native.sh                                   # once
uv run python scripts/real_data_check.py                       # all DGIdb targets (same as the pipeline)
uv run python scripts/real_data_check.py --drop-pk-genes       # without drug-handling genes
uv run pytest -q tests/test_real_data.py                       # SA == exact on the cached instances
```

### Inputs

| Flag | Default | Meaning |
|---|---|---|
| `--ks` | `2 3 4` | team sizes to solve and report; `n // 3` is always added for the timing check |
| `--lam` | `1.0` | λ, the overlap penalty weight (the report also shows λ = 0.5–5) |
| `--seed` | `1234` | seeds the z-score sampling that builds the instance, and SA |
| `--drop-pk-genes` | off | remove genes about drug handling rather than mechanism from every drug's targets: `CYP*`, `ABC*`, `SLC*`, `SLCO*`, `UGT*`, `FMO*`, `NAT1/2`, `ALB` |
| `--t0`, `--t1` | `1.0`, `0.01` | SA temperatures |
| `--sa-seeds`, `--sweeps` | `20`, `10 30 100 300 1000` | repeated SA runs for the time-to-answer check |

The 38 candidates, each with its mechanism class and why it is included, are in `CANDIDATES` at the top of the script. The literature combinations are in `KNOWN_COMBOS`, and brand/code names used to match trial interventions (Namzaric, AXS-05, …) are in `TRIAL_ALIASES`.

Data sources, downloaded once and then cached: DGIdb (drug targets), NCBI (gene IDs), STRING (protein network, confidence ≥ 400), and ClinicalTrials.gov (all 4,339 AD studies with their intervention lists). The first run takes several minutes, mostly for NCBI's 3 requests/second limit and for computing network separation for every drug pair. Later runs reuse the cache and take seconds.

### Files

| File | Tracked? | Content |
|---|---|---|
| `data/raw/extended/` | no | DGIdb, NCBI and ClinicalTrials.gov downloads |
| `data/raw/extended/<all_targets\|no_pk_genes>/` | no | STRING network, the seed gene IDs it was built from, and the built instance (`instance_seed1234.npz`: drug names, `w`, `P`, z-scores) |
| `data/frozen/manifest_extended.json` | yes (new) | provenance of every download: url, sha256, fetched_at. The pipeline's `manifest.json` is not touched |
| `data/real_runs.jsonl` | no | one record per solver run (same format as `data/runs.jsonl`) |
| `data/real_data_report.md` | no | the full report, overwritten each run |

The terminal output is the same report. It has sections for: eligible drugs (with z-score, reward `w`, and why others were excluded), SA vs exact for each k, time to answer and speedup over `exact_cpu`, known combinations with their rank, λ sensitivity, ClinicalTrials.gov co-occurrence, and the top 10 pairs and triples.

## Results: SA vs exact

| Variant | Network | Eligible drugs | k checked | SA = exact? | At k = n/3: `exact_cpu` | SA time to answer (99%) | SA speedup over exact |
|---|---|---|---|---|---|---|---|
| all targets | 688 genes | **n = 16** | 2, 3, 4, 5 | yes, every k (also = Python brute force) | 47 µs (4,368 teams) | 31 µs (30 sweeps) | ≈ 1.5 |
| `--drop-pk-genes` | 625 genes | **n = 21** | 2, 3, 4, 7 | yes, every k (also = Python brute force) | 2.6 ms (116,280 teams) | 35 µs (30 sweeps) | ≈ 74 |

On real data, SA finds the exact optimum every time. At n = 16 the two solvers are about even (this is the crossover). At n = 21, SA reaches the answer about 74× faster. For small k (2–4), exact search is still far faster, because there are only a few hundred to a few thousand teams.

## Results: does it make sense medically?

**Short answer: partly.** The top combinations follow a sensible pattern. But the model can't yet reproduce the one well-established combination, and its scores rest on several data artifacts. Details, using the `--drop-pk-genes` variant (n = 21) unless noted:

### What looks right

- **Best combinations pair a cholinesterase inhibitor with a drug from a different class.**
  - Best pair: rivastigmine + simvastatin. Best triple: rivastigmine + simvastatin + levetiracetam.
  - The rest of the top 10 adds GLP-1 agonists, baricitinib, metformin or bupropion to a cholinesterase inhibitor.
  - This matches how AD trials are actually designed: a new candidate given on top of standard cholinesterase-inhibitor therapy. Levetiracetam (network hyperexcitability), statins, GLP-1 agonists and metformin are all being studied in AD.
- **The overlap penalty does flag same-mechanism pairs.**
  - Donepezil + rivastigmine gets the maximum penalty (1.0). Rivastigmine + galantamine gets 0.70, and atorvastatin + simvastatin 0.39.
  - With λ ≥ 2, these redundant pairs fall to the 98th percentile (worst 2%) while the top-ranked combinations stay the same.

### What doesn't

1. **The approved combination is not favoured.**
   - Donepezil + memantine is FDA-approved as Namzaric (2014), but it ranks only 89th of 210 pairs (42nd percentile).
   - Memantine barely qualifies (reward 0.03, the lowest of the 21 eligible drugs). With all targets it is excluded entirely: its targets are on average exactly as close to the AD genes as random genes are (z ≈ 0).
   - Other literature combinations can't be scored at all because one partner is excluded: dextromethorphan + bupropion (AXS-05), dasatinib + quercetin, losartan + amlodipine + atorvastatin (rrAD), and masitinib add-on.
2. **At λ = 1 the negative control fails.** Two cholinesterase inhibitors together rank around the 45th percentile, no worse than a random pair, because the reward outweighs the penalty. λ ≈ 2–3 fixes this (table in the report).
3. **The network is too small to measure closeness well.**
   - STRING is queried only around the drugs' own targets plus the 16 AD genes, so the "interactome" is ~650 genes in which every gene is on average 1.5 steps from the AD module.
   - Closeness z-scores therefore have little room to separate drugs. This is why 17–22 of the 38 drugs come out "no closer than random", including dasatinib, losartan and most NSAIDs.
   - Published network-medicine work (e.g. Fang et al. 2021, cited in the proposal) uses the full human interactome (~15k genes).
4. **DGIdb "targets" are not all targets.**
   - DGIdb lists APOE, an AD-module gene, for donepezil, rivastigmine and galantamine. That is a pharmacogenetic association, not a drug target, but it puts these drugs at distance 0 from the AD module. It largely explains why rivastigmine gets the top reward (w = 1).
   - DGIdb also lists metabolism and transport genes (CYP3A4, ABCB1, …) shared by many unrelated drugs. Removing them (`--drop-pk-genes`) adds memantine, atorvastatin, nilotinib and quercetin to the eligible set and changes the rankings. The data needs this kind of filtering.
5. **Drugs with one target are never penalized.**
   - `separation()` treats a single-target drug's internal distance as 0, so huperzine (target: ACHE only) gets zero penalty with every other cholinesterase inhibitor. Rivastigmine + huperzine ranks 9th of 120 pairs in the all-targets variant.
   - In the opposite direction, escitalopram (2 targets) gets penalized with 11 other drugs in that variant, most with unrelated mechanisms.
6. **ClinicalTrials.gov co-occurrence is weak evidence.**
   - 7 of 210 pairs appear together in at least one AD study. Co-tested pairs rank no better than others (probability 0.47; 0.5 = no relation).
   - Most co-tested pairs are head-to-head comparisons (donepezil vs rivastigmine) or background therapy, not deliberate combinations. Only donepezil + memantine (11 studies) and rivastigmine/galantamine + memantine are real combination uses.

### Recommendations, in order of impact

1. Use the **full** STRING human network (or another whole-interactome resource) instead of a subnetwork around the candidates.
2. Restrict DGIdb interactions to mechanism targets: drop pharmacogenomic associations and drug-handling genes (`--drop-pk-genes` is a first step), or take targets from a curated source such as DrugBank or ChEMBL.
3. Raise λ to about 2–3, or calibrate it so that redundant pairs (the negative controls in `KNOWN_COMBOS`) rank in the bottom half.
4. Handle single-target drugs in `separation()`, e.g. by not assigning internal distance 0.

These are modelling choices for the paper, so none of them is applied to the pipeline. The script only adds the `--drop-pk-genes` option for this check.

## Literature sources used in `KNOWN_COMBOS`

- Donepezil + memantine (Namzaric), FDA approval December 2014: [Drugs.com approval history](https://Drugs.com/history/namzaric.html), [FDA label (2014)](https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/206439lbl.pdf)
- ChEI + memantine dual therapy, EFNS-ENS/EAN guideline (2015): [guideline PDF](https://orbilu.uni.lu/bitstream/10993/61069/1/EFNS%e2%80%90ENS%20EAN%20Guideline%20on%20concomitant%20use%20of%20cholinesterase%20inhibitors%20and%20memantine.pdf)
- Dasatinib + quercetin, phase 1 feasibility trial in mild AD (Gonzales et al., *Nat Med* 2023; NCT04063124): [PubMed 37679434](https://pubmed.ncbi.nlm.nih.gov/37679434/)
- Losartan + amlodipine + atorvastatin, rrAD (NCT02913664): [ClinicalTrials.gov record](https://classic.clinicaltrials.gov/ct2/show/NCT02913664). No published efficacy result found.
- Dextromethorphan + bupropion (AXS-05) for AD agitation: [NeurologyLive, FDA approval report](https://www.neurologylive.com/view/fda-approves-axs-05-new-treatment-alzheimer-disease-agitation), [BioSpace on phase 3 data](https://www.biospace.com/axsome-caps-big-year-with-solid-phase-iii-data-in-alzheimer-s-agitation)
- Masitinib as add-on to ChEI and/or memantine (AB09004; Dubois et al., *Alzheimer's Res Ther* 2023): [ESTHER entry](https://bioweb.supagro.inrae.fr/ESTHER/paper/Dubois_2023_Alzheimers.Res.Ther_15_39)
- Two drugs of the same class (two ChEIs, two ARBs, two statins) are used as **negative controls by reasoning** (same mechanism), not because a guideline was found that explicitly rules them out.
