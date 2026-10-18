# Mutation Picker

**Rank enzyme mutations before going to the lab.**
Given a protein sequence, Mutation Picker scores every possible single amino-acid substitution and returns a ranked top-k list of the variants most likely to keep the enzyme working.

> **Status:** work in progress. Data loading and the evolutionary baseline are working; the ESM-2 scoring, combined model, evaluation and app are still being built. No headline results are claimed yet.

---

## The problem

An enzyme of ~300 residues has about 5,700 possible single mutations (300 positions x 19 substitutions). A lab round can only test roughly 20 to 100 variants, and most random mutations damage the protein.

**Which 20 to 100 mutations should be tested first?**

## The approach

Three complementary signals are computed per mutation and combined with a small model:

| Signal | Source | Idea |
|---|---|---|
| **Evolution** | Multiple sequence alignment (MSA) | Positions that evolution keeps unchanged are usually important |
| **Structure** | AlphaFold structure (burial / RSA) | Changing a buried residue is riskier than changing a surface one |
| **Protein language model** | ESM-2 (masked-marginal log-odds) | How natural does the new amino acid look at this position? |

Evaluation is **retrospective** on [ProteinGym](https://proteingym.org) deep-mutational-scanning data: the tool ranks mutations without seeing the measured fitness, and we check how many of its top picks were genuinely functional.

## Research question

> When does a protein language model (ESM-2) beat a plain MSA conservation score, and when does it lose?

The plan is to test whether the answer depends on MSA depth (`Neff/L`), assay type and residue burial.

## Evaluation design

- **Leave-one-protein-out** cross-validation (grouped by UniProt ID, so assays of the same protein never leak into training).
- Scores and features **rank-normalized within each assay**.
- Metrics: Spearman, NDCG@k and precision@10, reported with bootstrap confidence intervals against a random baseline.

## Early sanity check

On the TEM-1 beta-lactamase assay (`BLAT_ECOLX_Stiffler_2015`), a simple MSA log-odds score reaches Spearman **0.45** with the true fitness (single run, unweighted MSA). This is only a pipeline check, not a final result.

## Limitations

- Single mutations only; no epistasis.
- ProteinGym assays mostly measure *tolerance* (is the variant still functional?), not improvement, and the assays differ in what they measure (growth, activity, abundance).
- Burial is computed from a single chain, ignoring binding partners and ligands.
- No wet-lab validation.

## Project layout

```
data/raw/        ProteinGym files (not tracked in git, see below)
src/             feature extraction, ESM-2 scoring, evaluation   (planned)
app/             Streamlit app                                    (planned)
doc/             detailed roadmap
```

See the full plan in [`doc/mutation-picker-roadmap.md`](doc/mutation-picker-roadmap.md).

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Data

The ProteinGym data is several GB and is **not** included in this repository. Download the substitution benchmark from [proteingym.org](https://proteingym.org) (or the [GitHub repo](https://github.com/OATML-Markslab/ProteinGym)) and place it as:

```
data/raw/DMS_substitutions.csv
data/raw/dms/DMS_ProteinGym_substitutions/
data/raw/msa/DMS_msa_files/
data/raw/weights/DMS_msa_weights/
data/raw/structures/ProteinGym_AF2_structures/
```

Then run `python main.py` to check that the files are found.

## References

- Notin et al., *ProteinGym*, NeurIPS 2023
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2), Science 2023
- Stiffler, Hekstra and Ranganathan, *Evolvability as a function of purifying selection in TEM-1 beta-lactamase*, Cell 2015
