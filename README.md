# Mutation Picker

VariantScout studies how well model scores rank experimentally measured single substitutions within an assay. The interactive app also displays scores for unmeasured substitutions.

**Retrospective result across 20 held-out proteins.** The average Ridge top ten contained 89.0% functional hits (random expectation 63.71%) and 16.0% top-decile hits (random expectation 13.475%). Mean Spearman correlation was 0.4614.

---

## The problem

An enzyme of ~300 residues has about 5,700 possible single mutations (300 positions × 19 substitutions). A lab round can only test roughly 20–100 variants, and most random mutations damage the protein.

**Which 20–100 mutations should be tested first?**

## The approach

Four signals are computed per mutation and will be combined with a small model:

| Signal | Source | Idea |
|---|---|---|
| **Evolutionary log-odds** | Multiple sequence alignment (MSA) | Positions evolution keeps unchanged are usually important |
| **RSA (burial)** | AlphaFold structure | Changing a buried residue is riskier than a surface one |
| **BLOSUM62** | Substitution matrix | How chemically conservative is the change? |
| **ESM-2 masked marginal** | Protein language model | How natural does the new amino acid look at this position? |

Evaluation is **retrospective** on [ProteinGym](https://proteingym.org) deep-mutational-scanning data: the tool ranks mutations without seeing the measured fitness, and we check how many of its top picks were genuinely functional.

## Research question

> When does a protein language model (ESM-2) beat a plain MSA conservation score, and when does it lose?

The plan is to test whether the answer depends on MSA depth (`Neff/L`), assay type and residue burial.

---

## Formulas

Every formula here corresponds directly to code in `src/`.

### 1. Evolutionary log-odds (`src/msa.py`)

For each MSA column (position) `i` and amino acid `a`, we compute a pseudocount frequency:

```
f(i, a) = (count(i, a) + 0.5) / (count_of_standard_residues(i) + 10)
```

Each of the 20 amino acid types receives a fixed pseudocount of 0.5. Gaps are excluded from the denominator.

The **log-odds score** for a mutation `wt → mt` at position `i` is:

```
evo_score(i, wt→mt) = ln( f(i, mt) / f(i, wt) )
```

A positive score means the mutant amino acid appears more often than the wild-type at that position across homologous sequences.

**Site entropy** (used as a separate signal):

```
H(i) = -Σ_a f(i,a) · ln( f(i,a) )    (over 20 amino acids)
```

Entropy is measured in nats. Low entropy → the position is conserved → changes are more likely to be damaging.

### 2. Relative Solvent Accessibility — RSA (`src/structure.py`)

Absolute solvent-accessible area (ASA) is computed with the Shrake–Rupley algorithm via Biopython. It is normalised by the maximum ASA for each amino acid type (Miller et al. 1987 values):

```
RSA(residue) = ASA(residue) / ASA_max(amino_acid_type)
```

RSA ≈ 0 means buried; RSA ≈ 1 means fully surface-exposed. Mutations at buried positions (low RSA) tend to damage the fold. pLDDT is read from the B-factor column of the AlphaFold PDB file.

### 3. BLOSUM62 (`src/baseline.py`)

The Biopython `substitution_matrices.load("BLOSUM62")` table gives an empirical log-odds score for each amino-acid pair derived from aligned protein blocks. Higher scores mean the substitution was observed frequently in related proteins:

```
blosum_score(wt→mt) = BLOSUM62[wt][mt]
```

### 4. ESM-2 masked-marginal log-odds (`src/esm.py`)

We use ESM-2 35M (`facebook/esm2_t12_35M_UR50D`). For each position `i`, the full sequence is fed to the model with position `i` masked (`<mask>`). The model predicts a probability distribution over all amino acids at that position:

```
esm_score(i, wt→mt) = log P(mt | sequence_masked_at_i)
                     - log P(wt | sequence_masked_at_i)
```

`+1` token offset accounts for ESM-2's `<cls>` start token.

ESM matrices are stored in `data/processed/esm_cache/`. Existing entries lack complete model revision and input provenance.

Reference comparisons use a named score column and report coverage and both correlations on matched measured single substitutions.

### 5. Combined model (`src/evaluate.py`)

Raw scores and feature scales differ across assays, so every feature and the target are **rank-normalised within each assay** before combining:

```
f_norm(i, assay) = rank(f_i in assay) / |assay|
```

A Ridge regression (α = 1) is then fit to predict the rank-normalised DMS score from the five rank-normalised features:

```
score(mutation) = w₀ + w₁·evo_norm + w₂·entropy_norm + w₃·blosum_norm
                      + w₄·rsa_norm + w₅·esm_norm
```

Each fold learns weights from the other proteins. Results are reported for both Ridge and LightGBM.

**Bootstrap CIs.** 95 % confidence intervals on the mean per-protein Spearman are computed by resampling the 20 per-protein values with replacement (n = 1000 bootstrap draws, percentile method).

---

## Results (20 proteins)

Spearman ρ between each signal and measured DMS fitness (full table in `results/tables/baseline.csv`):

| Assay | Neff/L | evo | RSA | ESM-2 | Best |
|---|---:|---:|---:|---:|---|
| GFP_AEQVI | 0.06 | 0.457 | 0.400 | 0.076 | evo |
| P53_HUMAN | 0.37 | **0.579** | 0.434 | 0.108 | evo |
| R1AB_SARS2 | 1.14 | **0.565** | 0.534 | -0.026 | evo |
| PTEN_HUMAN | 3.72 | 0.412 | **0.443** | 0.407 | RSA |
| TPMT_HUMAN | 26.71 | 0.405 | **0.443** | 0.433 | RSA |
| RNC_ECOLI | 70.70 | 0.548 | 0.324 | **0.557** | ESM-2 |
| BLAT (TEM-1) | 178.0 | 0.452 | **0.604** | 0.557 | RSA |

The three shallowest alignments had weak ESM-2 correlations. A near-zero correlation also occurred at Neff/L 168.82. These observations do not establish a depth cutoff for a new protein.

### Combined model on held-out measured substitutions

| Model | Mean Spearman | Functional P@10 | Random functional | Top-decile P@10 | Random top-decile |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.4614 | 89.0% | 63.71% | 16.0% | 13.475% |
| LightGBM | 0.4631 | 89.0% | 63.71% | 24.0% | 13.475% |

Each assay contributes one precision value. The denominator is the number of measured picks, up to ten. Functional hits use the assay's binary label. Top-decile hits include ties at its 90th-percentile score. The random expectations are the corresponding frequencies among all measured singles in that assay. The saved table has one assay per protein. New evaluation runs average assays within each protein for bootstrap intervals and paired gains.

The filtered app shortlist, fallback model, 650M model and double-mutant heuristic have no matching held-out precision estimate in these tables. Percentile tiers describe relative rank.

A separate GFP reporter check uses published ESM2_35M scores, fits Ridge on measured CALM1 and TPMT substitutions, and holds GFP out of training. Of 1,084 measured GFP singles, 1,051 passed the structural filter. With a cap of two picks per position, 20 of 20 picks met the functional threshold and 5 of 20 reached the assay-wide top decile. Across 2,000 random orderings under identical filters and caps, the corresponding mean rates were 91.59% and 9.69%. This single-reporter estimate uses measured candidates and a two-protein training cohort. The deployed app ranks unmeasured candidates and uses a different training cohort.

Run `python -m src.gfp_shortlist_check` to reproduce the aggregate check. It uses the three assays' ProteinGym zero-shot CSVs under `data/raw/zero_shot/` in addition to the documented benchmark data. Its report under `results/runs/gfp-shortlist/` records source, dependency, and input versions. It verifies report contents on reuse and reads no legacy scoring caches.

### Ablation (Ridge, drop one feature at a time)

| Dropped feature | Mean Spearman | Δ |
|---|---:|---:|
| none (full model) | 0.461 | — |
| RSA | 0.391 | **−0.070** |
| evo | 0.446 | −0.015 |
| blosum | 0.447 | −0.015 |
| entropy | 0.461 | −0.001 |
| ESM-2 | 0.459 | −0.002 |

Removing RSA produced the largest decrease in mean Spearman in the saved Ridge ablation.

## Evaluation design

- **Leave-one-protein-out** cross-validation, grouped by **UniProt ID** (not assay name — some proteins have multiple assays).
- Every feature and the target **rank-normalised within each assay** to a [0, 1] percentile.
- Two models compared: `Ridge(alpha=1.0)` and `LightGBMRegressor`. Ridge is reported as the primary because with 20 proteins and 5 features, boosted trees can overfit.
- Metrics: Spearman ρ, NDCG@10, NDCG@50, Precision@10 (binary hit) and Precision@10 (true top-10%), each with a random baseline. Bootstrap 95 % CIs (n=1000) on per-protein Spearman values.

## Limitations

- Single mutations only; no epistasis.
- ProteinGym assays differ in what they measure (growth, activity, abundance).
- Burial computed from a single chain, ignoring binding partners and ligands.
- No wet-lab validation.

## Project layout

```
data/raw/         ProteinGym files (not tracked in git)
data/processed/   ESM-2 score cache (not tracked in git)
src/
  data.py         assay loader, numbering checks, STARTER_SET
  msa.py          MSA parser, evo log-odds, site entropy
  structure.py    RSA and pLDDT from AlphaFold PDB
  esm.py          ESM-2 masked-marginal scoring and cache
  baseline.py     per-signal Spearman across all proteins
  evaluate.py     LOO cross-validation, ablation, Neff/L plot
  plots.py        heatmap generation
app/
  app.py          Streamlit app
doc/
  mutation-picker-roadmap.md
results/
  tables/         baseline.csv, loo_metrics.csv, ablation.csv
  figures/        per-protein heatmaps, neff_vs_rho.png
Makefile
```

See the full plan in [`doc/mutation-picker-roadmap.md`](doc/mutation-picker-roadmap.md).

### Interactive Web App Features (`app/app.py`)

- **Interactive 3D Structure Viewer (`py3Dmol`)**: Render AlphaFold PDB structures directly in Streamlit with dynamic coloring:
  - **Candidate & Catalytic Sites**: Highlights top-10 recommended mutations in hot pink spheres and active site catalytic residues in yellow sticks.
  - **pLDDT Confidence**: Colors by AlphaFold per-residue prediction confidence (Very High >90, Confident 70–90, Low 50–70, Very Low <50).
  - **RSA Burial & Mean Tolerance**: Visualizes solvent exposure and average mutational tolerance across the backbone.
- **ESM-2 Model Toggle**: Switch between `ESM-2 35M` (fast, lightweight) and `ESM-2 650M`, with persistent disk and memory caching.
- **Combinatorial (Double) Mutant Scoring**: Predicts pairwise combinations using additive scoring with **3D C-alpha distance constraints**:
  - Flags steric clash / direct contact risks ($d_{C\alpha} < 5$ Å).
  - Highlights structurally independent pairs ($d_{C\alpha} > 10$ Å) under the unvalidated additive heuristic.
- **Scientific Correctness Refinements**:
  - **Sequence-reweighted $N_{eff}/L$ ($\theta = 0.8$)**: Computed from the submitted alignment. For large, nonidentical alignments the estimate is withheld.
  - **Disordered Residue Filter (pLDDT $\ge$ 70 by default)**: Prevents floppy signal peptide tails from dominating top picks.
  - **Position Diversity Cap**: Enforces library diversity by capping substitutions per site (default: max 2).
  - **Rank Tiers**: Categorizes candidates into 🟢 Tier 1 (Top 1%), 🟡 Tier 2 (Top 1%–5%), and ⚪ Tier 3 (5%–15%).
  - **UniProt Accession Resolver**: Supports both UniProt Entry Names (`BLAT_ECOLX`) and primary accessions (`P62593`) via UniProt REST API.
  - **Full Landscape Export**: Download both top recommendations and the full ~5,000 mutation landscape CSVs.

---

## Setup and offline tests

Use Python 3.11 or 3.12. From the repository root, create an environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash.

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

The default suite uses synthetic fixtures and needs no benchmark downloads or pretrained weights. To list the data-dependent tests, run `python -m pytest --collect-only -m integration`. To run them, use `python -m pytest -m integration` after provisioning the reference CSV, assay CSV, MSA and structure files used by those tests and the 35M tokenizer and model weights. Some data-dependent tests skip when their files are absent.

Launch the app with `python -m streamlit run app/app.py`. The benchmark commands `python -m src.baseline` and `python -m src.evaluate` preflight the starter cohort and write to unique directories under `results/runs/`. Each run records its input hashes and completion state in `manifest.json`. The published CSVs stay in `results/tables/`.

For Docker, build with `docker build -t variantscout .`. Local datasets, uploads and environments are excluded from the image. From the project root in PowerShell or Bash:

```sh
docker run --rm -p 8501:8501 -v "${PWD}/data/raw:/app/data/raw:ro" -v variantscout-processed:/app/data/processed variantscout
```

### Data

The ProteinGym data is several GB and is **not** included in this repository. Download the substitution benchmark from [proteingym.org](https://proteingym.org) and place it as:

```
data/raw/DMS_substitutions.csv
data/raw/dms/DMS_ProteinGym_substitutions/
data/raw/msa/DMS_msa_files/
data/raw/weights/DMS_msa_weights/
data/raw/structures/ProteinGym_AF2_structures/
```

## References

- Notin et al., *ProteinGym*, NeurIPS 2023
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2), Science 2023
- Stiffler, Hekstra and Ranganathan, *Evolvability as a function of purifying selection in TEM-1 beta-lactamase*, Cell 2015
- Miller et al., *Interior and surface of monomeric proteins*, Journal of Molecular Biology 1987 (RSA max ASA values)
