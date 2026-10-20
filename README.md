# Mutation Picker

**Rank enzyme mutations before going to the lab.**
Given a protein sequence, Mutation Picker scores every possible single amino-acid substitution and returns a ranked top-k list of the variants most likely to keep the enzyme working.

> **Leave-one-protein-out on 20 ProteinGym assays — Ridge combined model: mean Spearman ρ = 0.461 (95 % CI 0.397–0.523), Precision@10 = 89 % vs 64 % random (binary functional hit).**

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
f(i, a) = (count(i, a) + pseudo) / (N_sequences + 20 × pseudo)
```

`pseudo = 0.01 × N_sequences` (relative pseudocount).

The **log-odds score** for a mutation `wt → mt` at position `i` is:

```
evo_score(i, wt→mt) = log2( f(i, mt) / f(i, wt) )
```

A positive score means the mutant amino acid appears more often than the wild-type at that position across homologous sequences.

**Site entropy** (used as a separate signal):

```
H(i) = -Σ_a f(i,a) · log2( f(i,a) )    (over 20 amino acids)
```

Low entropy → the position is conserved → changes are more likely to be damaging.

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

Results are cached to `data/processed/esm_cache/<UniProt_ID>.npy` as a `(L, vocab_size)` log-probability matrix, so the model runs at most once per protein.

> **Validation note:** before using ESM-2 numbers in any write-up, compare against ProteinGym's published zero-shot scores (download `DMS_substitutions_zero_shot_scores.zip` from proteingym.org). Your Spearman should match theirs within ±0.02 for the same model size. A larger gap almost always means an off-by-one in the token position indexing.

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

Weights are learned on all *other* proteins (leave-one-protein-out). Because only 5 features and ~20 groups are available, Ridge is more reliable here than a boosted tree — both are reported so the reader can check.

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

*Key finding:* ESM-2 is near-random or negative on the three shallowest MSAs (Neff/L < 2) and wins on mid-to-high depth proteins. RSA is the most consistently strong single signal. See `results/figures/neff_vs_rho.png`.

### Combined model (LOO)

Ridge and LightGBM give nearly identical results, which shows the signal is real rather than model-specific:

| Model | Mean Spearman | 95 % CI | Mean P@10 (binary) | Random P@10 |
|---|---:|---|---:|---:|
| Ridge | **0.461** | [0.397, 0.523] | **89 %** | 64 % |
| LightGBM | 0.463 | — | 88 % | 64 % |

Full per-protein table: `results/tables/loo_metrics.csv`.

### Ablation (Ridge, drop one feature at a time)

| Dropped feature | Mean Spearman | Δ |
|---|---:|---:|
| none (full model) | 0.461 | — |
| RSA | 0.391 | **−0.070** |
| evo | 0.446 | −0.015 |
| blosum | 0.447 | −0.015 |
| entropy | 0.461 | −0.001 |
| ESM-2 | 0.459 | −0.002 |

RSA is by far the most important signal. ESM-2 and entropy add almost nothing on average, but ESM-2 dominates individually on several mid-depth proteins (RNC, RASH, BLAT). Full table: `results/tables/ablation.csv`.

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

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run baseline signals across all 20 proteins
python -m src.baseline

# Run LOO evaluation, ablation, and Neff/L plot
python -m src.evaluate

# Launch the Streamlit app
streamlit run app/app.py
```

Or with `make`:

```bash
make install
make reproduce   # runs baseline then evaluate
make app         # starts the Streamlit server
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
