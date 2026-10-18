# Mutation Picker

**Rank enzyme mutations before going to the lab.**
Given a protein sequence, Mutation Picker scores every possible single amino-acid substitution and returns a ranked top-k list of the variants most likely to keep the enzyme working.

> **Status:** work in progress. Evolutionary, structure, BLOSUM and ESM-2 baselines are working across 10 proteins. Combined model and app still to come. No headline results are claimed yet.

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

---

## Early results (10 proteins, unvalidated)

Spearman ρ between each signal and measured DMS fitness:

| Assay | Neff/L | evo | BLOSUM | RSA | ESM-2 | Best |
|---|---:|---:|---:|---:|---:|---|
| R1AB_SARS2 | 1.1 | **0.565** | 0.207 | 0.534 | -0.026 | evo |
| PTEN_HUMAN | 3.7 | 0.412 | 0.182 | **0.443** | 0.407 | RSA |
| TPMT_HUMAN | 26.7 | 0.405 | 0.240 | **0.443** | 0.433 | RSA |
| TPK1_HUMAN | 29.9 | 0.221 | **0.241** | 0.126 | 0.199 | BLOSUM |
| RNC_ECOLI | 70.7 | 0.548 | 0.228 | 0.324 | **0.557** | ESM-2 |
| DYR_ECOLI | 77.5 | **0.392** | 0.151 | 0.341 | 0.349 | evo |
| AMIE_PSEAE | 89.1 | 0.213 | 0.277 | **0.500** | 0.408 | RSA |
| BLAT (TEM-1) | 178.0 | 0.452 | 0.346 | **0.604** | 0.557 | RSA |
| OXDA_RHOTO | 271.3 | 0.140 | 0.205 | **0.352** | 0.219 | RSA |
| KKA2_KLEPN | 298.7 | 0.275 | 0.226 | **0.532** | 0.261 | RSA |

RSA is the best single signal on **6 of 10** proteins. ESM-2 is near-random (-0.026) on R1AB — the shallowest MSA — while the evolutionary score is the strongest (0.565). This needs validation against ProteinGym's published ESM-2 scores before interpretation.

No confidence intervals are reported yet. These are single-run point estimates.

## Evaluation design

- **Leave-one-protein-out** cross-validation (grouped by UniProt ID).
- Scores and features **rank-normalised within each assay**.
- Metrics: Spearman, NDCG@k and precision@10, with bootstrap confidence intervals against a random baseline.

## Limitations

- Single mutations only; no epistasis.
- ProteinGym assays differ in what they measure (growth, activity, abundance).
- Burial computed from a single chain, ignoring binding partners and ligands.
- No wet-lab validation.

## Project layout

```
data/raw/         ProteinGym files (not tracked in git)
data/processed/   ESM-2 score cache (not tracked in git)
src/              feature extraction and evaluation
  data.py         assay loader and numbering checks
  msa.py          MSA parser, evo log-odds, entropy
  structure.py    RSA and pLDDT from AlphaFold PDB
  esm.py          ESM-2 masked-marginal scoring and cache
  baseline.py     runs all signals across the protein set
  plots.py        heatmap generation
doc/              detailed roadmap
results/          tables and figures (tracked)
```

See the full plan in [`doc/mutation-picker-roadmap.md`](doc/mutation-picker-roadmap.md).

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.baseline
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
