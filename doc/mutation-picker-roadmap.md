# Mutation Picker: A Verified, Recruiter-Ready Roadmap

A protein-ML portfolio project for enzyme-design and ML internship applications.

> **Status of this document.** This is the corrected version of the original roadmap.
> It was checked against the ProteinGym files in `data/raw/`. Items marked
> **[verified]** were run on real data. Items marked **[untested]** are standard
> methods that have not yet been run here, so test them before trusting them.

---

## 0. What Changed and Why (Findings from Checking the Original Plan)

| # | Finding | Status | Fix in this version |
|---|---------|--------|---------------------|
| 1 | Removing lowercase letters from the MSA shifts every position. The TEM-1 query has **71 lowercase residues** (including the signal peptide), so 286 columns shrink to 215. `freq[pos-1]` then points at the wrong residue. `.` gap characters also make sequence lengths ragged. | verified | Uppercase everything, `.` -> `-`, keep all 286 columns, and assert the query matches the DMS wild type. |
| 2 | TEM-1 has 4 assays and DYR has 2. Grouping cross-validation by assay leaks the held-out protein into training. | reasoned | Group by `UniProt_ID`. |
| 3 | A "top 10% hit" in TEM-1 is a wild-type-like plateau: 28% of variants score >= -0.1 and the q90 threshold is about 0.0. A hit really means "not damaging". | verified | Report two hit definitions and say plainly that the benchmark measures tolerance, not improvement. |
| 4 | Conservation (negative entropy) and RSA have the "wrong" sign against fitness (conservation rho = -0.37 on TEM-1). | verified | Define every feature so that higher means more tolerated. |
| 5 | The shipped weights file for TEM-1 has 208,923 entries but the MSA has 209,644 sequences. | verified | Check lengths before using weights. Never plug them in blindly. |
| 6 | Pure Python loops over 200k sequences are slow. | verified | Vectorize with numpy. |
| 7 | The Streamlit app cannot make an MSA from a FASTA file alone. | reasoned | Three options in Section 9, plus a fallback model. |
| 8 | Several assays are not catalytic activity: TPMT = protein abundance, PTEN = yeast-growth surrogate, TEM-1/DYR/KKA2 = organism growth. | verified | An assay-type column goes into every results table. |
| 9 | Docker "reproduce everything" is unrealistic because the data runs to several GB. | reasoned | Mount data as a volume and document a download script. |
| 10 | The 4-week plan packs too much into week 4. | reasoned | Six-week plan with a clear "must-have vs stretch" split. |

**Sanity result [verified, single assay, one run].**
`BLAT_ECOLX_Stiffler_2015`, site-independent MSA score with the corrected parser
(no sequence weights, because the weights file did not match the MSA):

| Signal | Spearman vs `DMS_score` |
|---|---|
| Evolutionary log-odds score | **0.452** |
| Conservation (neg. entropy, raw sign) | -0.374 (flip the sign to read it as +0.374) |

Top-10 picks by the evolutionary score: **3/10** are in the true top 10% by value,
but **10/10** are labelled functional by `DMS_score_bin`. That gap is the reason
for Section 7.

---

## 1. The Problem and the Pitch

### Problem

An enzyme has about 300 positions, each with 19 possible substitutions, giving about
5,700 single mutants. A lab round tests only about 20 to 100 variants, and most
random mutations hurt the enzyme.

**Which 20 to 100 mutations should be tested first?**

### The sharper research question (this is the headline)

Do not pitch "a tool that ranks mutations." Many people build that. Pitch a question
you can answer with data:

> **When does a protein language model (ESM-2) beat a plain MSA conservation score,
> and when does it lose?  Can a small model learn to combine them per enzyme?**

Mutation Picker is then the tool that implements the answer.

Candidate hypotheses to test, each with a clear result either way:

- ESM-2 wins when the MSA is shallow (low `MSA_Neff_L`, e.g. PTEN at 3.7).
- The MSA wins when it is deep (e.g. TEM-1 at 178).
- ESM-2 wins on abundance/stability assays and loses on specific-activity assays.
- Buried positions favor the MSA, and surface positions favor ESM-2.

### Signals

- **Evolution:** per-position amino-acid frequencies from the MSA.
- **Structure:** burial (RSA) from the AlphaFold structure.
- **Protein language model:** ESM-2 masked-marginal log-odds.

### How it is judged

Retrospectively, with ProteinGym's measured fitness. Features never see the answers.
The combining model is trained on *other proteins* only.

---

## 2. Intuition (Short Version)

- A **mutation** such as `A24C` means alanine at position 24 becomes cysteine.
- **Deep mutational scanning (DMS)** makes nearly every single mutation, selects for
  function, and sequences before and after. Each variant gets a fitness score.
  In ProteinGym, higher `DMS_score` means fitter.
- **Conservation:** positions evolution rarely changes are usually important.
- **Burial:** changing a buried residue often breaks folding; surface residues are
  more tolerant.
- **ESM-2:** a protein language model. Mask a position and it tells you which amino
  acids look natural there. (It was trained on roughly 65M UniRef50 clusters.)

### Pipeline

```
ProteinGym assay (mutations + fitness)
   + MSA (evolution)   + AF2 structure (burial)   + ESM-2
                         |
        features per mutation (rank-normalized within each assay)
                         |
     simple model (ridge first, LightGBM second) -> score
                         |
  Leave-one-PROTEIN-out evaluation: Spearman, NDCG@k, top-10 precision
                         |
   Error analysis (the question in Section 1) -> Streamlit app -> README
```

### Reading (overview only)

- ProteinGym (Notin et al., NeurIPS 2023)
- ESM-2 (Lin et al., Science 2023)
- TEM-1 DMS (Stiffler, Hekstra, Ranganathan, Cell 2015)

---

## 3. Setup and Repository Layout

A laptop CPU is enough for the 35M ESM-2 model. Use free Colab for bigger models.

```bash
pip install biopython pandas numpy scipy scikit-learn lightgbm torch transformers matplotlib seaborn streamlit py3Dmol
```

> Note: Biopython was **not** installed in the current environment. Install it
> before running the structure and substitution-matrix code below.

```
variantScout/
  data/raw/          ProteinGym: dms/, msa/, weights/, structures/  (already downloaded)
  data/processed/    feature tables (parquet/csv)
  notebooks/
  src/
    data.py          load reference + assay, numbering checks
    msa.py           MSA loading, frequencies, evo score
    structure.py     RSA, pLDDT
    esm.py           masked-marginal scoring
    evaluate.py      metrics + leave-one-protein-out
  app/app.py
  results/           tables/, figures/
  README.md
```

Create the Git repository on day one. Commit small and often with readable messages.

---

## 4. Data

### 4A. What you already have [verified]

| Folder | Contents |
|---|---|
| `data/raw/DMS_substitutions.csv` | 217 assays, 46 columns |
| `data/raw/dms/DMS_ProteinGym_substitutions/` | 217 assay CSVs: `mutant`, `mutated_sequence`, `DMS_score`, `DMS_score_bin` |
| `data/raw/msa/DMS_msa_files/` | 101 MSAs (shared across assays of one protein) |
| `data/raw/weights/DMS_msa_weights/` | `.npy` sequence weights |
| `data/raw/structures/ProteinGym_AF2_structures/` | AlphaFold2 PDBs (pLDDT is in the B-factor column) |

You do **not** need to download PDB 1BTL: ProteinGym already ships structures
whose numbering matches the target sequence. (`pdb_range` was `1-286` for TEM-1.)

### 4B. Choose 6 to 10 proteins

Selection criteria: `seq_len < 500`, at least ~2,000 single mutants, and a **mix of
assay types** so that your "when does ESM-2 lose" analysis has contrast.

Starting set (all present and checked in the reference file):

| Assay | Length | Singles | Assay type | MSA Neff/L |
|---|---|---|---|---|
| `BLAT_ECOLX_Stiffler_2015` | 286 | 4,996 | Organism fitness (ampicillin growth) | 178.0 (deep) |
| `DYR_ECOLI_Thompson_2019` | 159 | 2,363 | Organism fitness (growth) | 77.5 |
| `KKA2_KLEPN_Melnikov_2014` | 264 | 4,960 | Organism fitness (kanamycin growth) | 298.7 (deep) |
| `AMIE_PSEAE_Wrenbeck_2017` | 346 | 6,227 | Activity ("enzyme function") | 89.1 |
| `PTEN_HUMAN_Mighell_2018` | 403 | 7,260 | Activity (yeast growth surrogate) | 3.7 (shallow) |
| `TPMT_HUMAN_Matreyek_2018` | 245 | 3,648 | Expression (protein abundance) | 26.7 |

Add 2 to 4 more from the reference file to reach 8 to 10. Use a script to filter:

```python
import pandas as pd
ref = pd.read_csv("data/raw/DMS_substitutions.csv")
cand = ref[(ref.seq_len < 500) & (ref.DMS_number_single_mutants >= 2000)]
print(cand[["DMS_id", "seq_len", "DMS_number_single_mutants",
            "MSA_Neff_L", "coarse_selection_type", "selection_assay"]]
      .sort_values("MSA_Neff_L").to_string())
```

Rules:

- **One assay per protein** for the main benchmark (or group by protein, see Section 7).
  Keep the other assays of the same protein as an extra "same protein, new assay" test.
- Record `selection_assay` for each protein in a table in the README. Assays do not
  all measure the same thing.
- Mix `coarse_selection_type` values (Activity, Expression, OrganismalFitness,
  Stability) so the analysis has contrast.

### 4C. Numbering checks (do not skip)

In the six checked assays `MSA_start = 1`, `pdb_range` starts at 1 and
`raw_mut_offset` is empty, so numbering is direct. **Do not assume this for the extra
proteins.** Assert it in code for every protein:

```python
import numpy as np, pandas as pd

def check_numbering(row, df, msa_query):
    """Fail loudly if DMS, MSA and target sequence disagree."""
    tgt = row.target_seq
    pos = df.mutant.str[1:-1].astype(int).values
    wt  = df.mutant.str[0].values
    assert all(tgt[p - 1] == a for p, a in zip(pos, wt)), "DMS wt != target_seq"
    assert msa_query.upper() == tgt[int(row.MSA_start) - 1:int(row.MSA_end)], \
        "MSA query != target region (check MSA_start/MSA_end)"
```

Also check the structure: extract the residue sequence from the PDB and align it
to `target_seq` (Biopython `PairwiseAligner`). Build one mapping table:

`dms_pos -> msa_col -> pdb_resid`

If `MSA_start != 1`, the MSA column is `pos - MSA_start`, not `pos - 1`.

### 4D. First look at the data

```python
import pandas as pd
df = pd.read_csv("data/raw/dms/DMS_ProteinGym_substitutions/BLAT_ECOLX_Stiffler_2015.csv")
df = df[~df.mutant.str.contains(":")]           # singles only
df["wt"] = df.mutant.str[0]
df["pos"] = df.mutant.str[1:-1].astype(int)
df["mt"] = df.mutant.str[-1]
sens = df.groupby("pos").DMS_score.mean()        # per-position sensitivity
```

First figure: position x amino-acid heatmap of `DMS_score`.
Always print the score distribution too. TEM-1 has a large wild-type-like plateau
(28% of variants have `DMS_score >= -0.1`).

---

## 5. Baseline Signals

### 5A. Evolution from the MSA [verified]

Corrected loader: keep all columns, uppercase, convert `.` to `-`, vectorize.

```python
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"

def load_msa(path):
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            else:
                cur.append(line)
    seqs.append("".join(cur))
    seqs = [s.upper().replace(".", "-") for s in seqs]    # keep ALL columns
    assert len({len(s) for s in seqs}) == 1, "ragged MSA"
    M = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), -1)
    return seqs[0], M

def frequencies(M, weights=None):
    w = np.ones(len(M)) if weights is None else weights
    f = np.zeros((M.shape[1], 20))
    for j, a in enumerate(AA):
        f[:, j] = ((M == a.encode()) * w[:, None]).sum(0)
    return (f + 0.5) / (f.sum(1, keepdims=True) + 10)     # pseudocount

def evo_score(f, pos, wt, mt):                             # pos is 1-based DMS position
    return np.log(f[pos - 1, AA.index(mt)]) - np.log(f[pos - 1, AA.index(wt)])

def conservation(f):                                       # higher = more conserved
    return (f * np.log(f)).sum(1)                          # negative entropy
```

**Sequence weights.** Weighting down near-duplicate sequences (for example 80%
identity, `theta = 0.2`) is better in principle, but:

- Only use the shipped `.npy` weights if `len(w) == number of MSA sequences`.
  The TEM-1 file did **not** match (208,923 vs 209,644). Likely the weights were
  computed on a filtered MSA. The exact filter is unknown, so do not guess.
- If lengths do not match, either use unweighted frequencies (acceptable for a
  baseline and what the sanity result used) or subsample to about 20k sequences and
  compute your own weights. Computing weights for 200k sequences is O(N^2)
  and too slow.

**Sign convention.** Define every feature so that **higher = more tolerated**, so
all Spearman values read in the same direction:

```python
cons_tol = -conservation(f)         # high entropy = tolerant
```

### 5B. Burial from the structure [untested here]

ProteinGym ships AlphaFold2 models, so use those. pLDDT is stored in the B-factor
column.

```python
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1

MAXASA = dict(A=129, R=274, N=195, D=193, C=167, Q=225, E=223, G=104, H=224, I=197,
              L=201, K=236, M=224, F=240, P=159, S=155, T=172, W=285, Y=263, V=174)

s = PDBParser(QUIET=True).get_structure("x", "data/raw/structures/ProteinGym_AF2_structures/BLAT_ECOLX.pdb")
ShrakeRupley().compute(s[0], level="R")
chain = next(s[0].get_chains())                  # do not hard-code "A"
rsa, plddt = {}, {}
for r in chain:
    if r.id[0] == " ":                           # skip waters and ligands
        rsa[r.id[1]] = r.sasa / MAXASA[seq1(r.resname)]
        plddt[r.id[1]] = r["CA"].get_bfactor()
```

- Small RSA = buried, large RSA = exposed. Use `rsa` directly as "tolerance".
- Positions with pLDDT below about 70 are unreliable: record them and use them in the
  error analysis.
- RSA from a single chain ignores partners and ligands. Say so in the README.

### 5C. BLOSUM62

```python
from Bio.Align import substitution_matrices
B62 = substitution_matrices.load("BLOSUM62")
blosum = B62[wt][mt]
```

### 5D. The "before" table

One row per signal per assay: Spearman against `DMS_score`.

```python
from scipy.stats import spearmanr
rho = spearmanr(df.evo, df.DMS_score).correlation
```

Rows: BLOSUM62, conservation, RSA, evolutionary score (and later ESM-2).
Add a column with the assay type and one with `MSA_Neff_L`.

---

## 6. ESM-2 Scoring [untested here]

Start with `facebook/esm2_t12_35M_UR50D` (CPU). Scale up to 150M or 650M on Colab
if time allows. Report the scaling curve (35M -> 150M -> 650M), which is itself
an interesting result.

```python
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

name = "facebook/esm2_t12_35M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
model = EsmForMaskedLM.from_pretrained(name).eval()

@torch.no_grad()
def esm_logprobs(seq, batch=32):
    assert len(seq) <= 1022, "sequence too long for ESM-2; window or truncate"
    ids = tok(seq, return_tensors="pt")["input_ids"]          # [1, L+2]
    L = len(seq)
    out = torch.zeros(L, model.config.vocab_size)
    for start in range(0, L, batch):
        idx = list(range(start, min(start + batch, L)))
        m = ids.repeat(len(idx), 1)
        for row, i in enumerate(idx):
            m[row, i + 1] = tok.mask_token_id                 # +1 for the start token
        logits = model(m).logits
        for row, i in enumerate(idx):
            out[i] = torch.log_softmax(logits[row, i + 1], -1)
    return out

# mutation score = lp[pos-1, tok.convert_tokens_to_ids(mt)] - lp[pos-1, tok.convert_tokens_to_ids(wt)]
```

Intuition: hide one position and ask which amino acid fits. If the mutant is more
probable than the wild type, the mutation looks safe.

Cache the log-probability matrix per protein (`.npy`) so you compute each protein
only once.

---

## 7. Combine the Signals and Evaluate Properly

### 7A. Features

`esm_score`, `evo_score`, `blosum`, `rsa`, `cons_tol`, hydrophobicity change.

Extra features worth trying:

- `plddt` at the position.
- `mt` and `wt` one-hot or physico-chemical properties (volume, charge).
- Whether the mutant is proline or glycine (these are special).
- **Position-level vs substitution-level split:** a position's average evolutionary
  or ESM score captures "is this site sensitive", while the per-mutation
  difference captures "which substitution is the least bad". Reporting both is an
  insight many projects miss.

### 7B. The non-negotiable evaluation rules

1. **Normalize inside each assay.** Raw scores and raw feature scales differ across
   assays. Rank-normalize the target *and* every feature within each assay.

   ```python
   df["y"] = df.groupby("uniprot")["DMS_score"].rank(pct=True)
   for c in FEATURES:
       df[c] = df.groupby("uniprot")[c].rank(pct=True)
   ```

2. **Leave-one-PROTEIN-out cross-validation.** Group by `UniProt_ID`, not by assay
   name. Otherwise a protein with several assays (TEM-1 has four, DYR has two)
   leaks into training. A random split is also wrong, because neighbouring
   mutations at the same position look alike.

   ```python
   from sklearn.model_selection import LeaveOneGroupOut
   from sklearn.linear_model import Ridge
   import lightgbm as lgb

   def loo(df, X_cols, make_model):
       df = df.copy(); df["pred"] = np.nan
       for tr, te in LeaveOneGroupOut().split(df, groups=df.uniprot):
           m = make_model().fit(df.iloc[tr][X_cols], df.iloc[tr].y)
           df.iloc[te, df.columns.get_loc("pred")] = m.predict(df.iloc[te][X_cols])
       return df

   ridge = lambda: Ridge(alpha=1.0)
   gbm   = lambda: lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15)
   ```

   With only 6 to 10 groups and about 6 features, a boosted-tree model can easily
   overfit. **Report ridge and LightGBM both.** If ridge matches the boosted model,
   say so; it shows judgement.

### 7C. Metrics (report all, per protein and averaged)

| Metric | Why |
|---|---|
| Spearman | Standard in ProteinGym, so comparable to published numbers |
| NDCG@k (k = 10, 50) | Rewards putting the best variants first |
| **Precision@10 using `DMS_score_bin`** | "Of my 10 picks, how many are functional?" |
| **Precision@10 using true top-10%** | Stricter: "are they among the best?" |
| Random baseline | Expected precision@10 equals the base rate; compute it, do not assume |

```python
from sklearn.metrics import ndcg_score

def precision_at_k(d, col, k=10, hit="bin"):
    top = d.nlargest(k, col)
    if hit == "bin":
        return top.DMS_score_bin.mean()
    thr = d.DMS_score.quantile(0.9)
    return (top.DMS_score >= thr).mean()

def ndcg_at_k(d, col, k=10):
    return ndcg_score([d.y.values], [d[col].values], k=k)    # y = rank percentile (>= 0)
```

**Be honest about what a "hit" is.** In these assays most top-ranked variants are
wild-type-like. The benchmark therefore tests *finding non-damaging mutations*,
not *finding improvements*. State this in the README. It is a strength, not a
weakness, to name it.

**Uncertainty.** Add bootstrap confidence intervals over positions or mutations
for every headline number. With 6 to 10 proteins, a single mean without an interval
is not credible.

---

## 8. What Makes Recruiters Notice

Ordered by impact per hour of work:

1. **A real research question with an honest answer** (Section 1). For example:
   "ESM-2 35M beats conservation only when `MSA_Neff_L` is below X; on deep MSAs the
   plain MSA score wins." Plot Spearman difference against `MSA_Neff_L`.
2. **Correct evaluation:** leave-one-protein-out, per-assay normalization, two hit
   definitions, confidence intervals, random baseline.
3. **Ablation table:** drop one feature at a time and report the change.
4. **Error analysis:** where does it fail? Check shallow MSAs (PTEN), low-pLDDT
   regions, proline/glycine substitutions, active-site positions, and positions
   where ESM-2 and the MSA disagree most. Show 2 to 3 concrete failing examples.
5. **ESM-2 scaling result:** 35M vs 150M vs 650M on your assays.
6. **Honest comparison with ProteinGym:** look up the current leaderboard yourself
   and place your numbers next to it. Say plainly where you stand and why.
   You are not expected to beat it, and the site-independent MSA model plus ESM-2
   are already baselines there.
7. **A demo on a non-benchmark enzyme** (e.g. a lipase), labelled "no wet-lab
   validation".
8. **A clean repository:** typed functions, a `Makefile` or a short script per step,
   a small test that checks numbering (Section 4C), and a pinned `requirements.txt`.
9. **Reproducibility:** `make reproduce` that downloads data and regenerates the
   tables. Docker is a stretch goal, not a requirement.

---

## 9. Build the Tool

A Streamlit app. The user gives a protein sequence (FASTA) and gets a ranked top-20
table, a position-by-amino-acid heatmap and a 3D view (`py3Dmol`) with sensitive
positions coloured.

### The MSA problem

A FASTA alone cannot produce an MSA: a homology search over UniRef is far too heavy
for a laptop. Choose one of:

| Option | Effort | Notes |
|---|---|---|
| **A. Optional `.a2m` upload** | Low | Simplest. Document how to generate one. |
| **B. ColabFold MMseqs2 server** | Medium | Needs internet and polite rate limits; check its usage terms. [untested] |
| **C. ESM-2 + structure only** | Low | Fallback model with no MSA features. |

**Train the fallback separately.** A model trained with MSA features gets
out-of-distribution inputs if you simply zero them. Train model A (all features)
and model C (ESM-2 + structure + BLOSUM) and let the app choose.

### Structure source

- User PDB upload, or
- AlphaFold DB by UniProt ID, or
- ESMFold (heavier).

### Uncertainty flags in the app

- Few MSA sequences (use `Neff / L`, not raw count).
- Low pLDDT regions.
- Sequence longer than the ESM-2 limit.

### Packaging

Docker is optional. If you do it, mount `data/` as a volume and keep model weights
out of the image (download on first run).

---

## 10. Presentation

**README order:**

1. One figure (heatmap or 3D view) and **one result line**, e.g.
   "Leave-one-protein-out on N assays: precision@10 = X% vs Y% random."
2. The research question and the finding.
3. Results table (Spearman, NDCG@10, precision@10 both ways, with CIs).
4. Method, with a diagram.
5. Limitations.
6. How to run.

**Limitations to state (all of these are real):**

- Single mutants only, no epistasis.
- Assays are tolerance screens, not improvement screens; assay types differ.
- Features from one structure ignore binding partners and ligands.
- Small number of proteins in the held-out evaluation.
- No wet-lab validation. The benchmark is not an industrial setting.

**Extras:**

- Two-page write-up (PDF).
- LinkedIn post: the question, one figure, the finding, the repo link.

**Resume bullet (fill in real numbers only):**

> Built a mutation-ranking pipeline combining ESM-2, MSA conservation and AlphaFold burial; showed with leave-one-protein-out validation on [N] ProteinGym assays that [finding about MSA depth]; precision@10 of [X]% vs [Y]% for random, with bootstrap CIs.

---

## 11. Common Mistakes

1. **Dropping lowercase MSA letters.** Numbering silently shifts (verified on TEM-1).
2. Assuming MSA, DMS and PDB numbering agree. Assert it (Section 4C).
3. Leave-one-*assay*-out instead of leave-one-*protein*-out.
4. Random train/test splits.
5. Mixing raw scores or raw feature scales across assays.
6. Mixed-up signs (conservation and RSA need flipping).
7. Plugging in sequence weights without checking the length.
8. Feeding ESM-2 a sequence over about 1022 residues.
9. Calling a "not damaging" hit an "improvement".
10. Reporting a single mean with no confidence interval.
11. Hiding limitations.

---

## 12. Interview Questions

- **Spearman vs top-k precision?** Spearman scores the whole ranking, top-k scores
  only the part that goes to the lab. They can disagree (3/10 vs 10/10 above).
- **Why leave-one-protein-out and not leave-one-assay-out?** Several assays share
  a protein, so the held-out protein would leak.
- **Why normalize within each assay?** Scales differ, so raw scores cannot be pooled.
- **When does ESM-2 lose to conservation?** Answer from your own plot against
  `MSA_Neff_L`.
- **What does a "hit" mean in your benchmark?** Not damaging, rather than improved.
- **Why ridge as well as LightGBM?** Few groups and few features invite overfitting.
- **How did you handle MSA numbering?** Describe the lowercase-column trap and the
  assertion you wrote.

---

## 13. Six-Week Plan (about 2 to 3 hours per day)

| Week | Must-have | Stretch |
|---|---|---|
| 1 | Setup, repo, load data, numbering checks, assay table, first heatmap | Read ProteinGym and ESM-2 intros |
| 2 | MSA loader, conservation, evo score, BLOSUM, RSA/pLDDT, "before" table | Own sequence weights on a subsampled MSA |
| 3 | ESM-2 35M on all proteins (cached), ridge leave-one-protein-out | 150M and 650M scaling |
| 4 | LightGBM, metrics with CIs, ablation, **Spearman-vs-Neff/L analysis** | Position vs substitution split |
| 5 | Error analysis, fallback model, Streamlit app (upload `.a2m` or ESM-only) | Demo on a lipase |
| 6 | README, figures, write-up PDF, tests, `make reproduce`, LinkedIn post | Docker, ColabFold integration |

If time runs short, cut the stretch column. Never cut the evaluation.

---

## Appendix A: The Simple Starter (TEM-1 Tolerance Map)

A 10-day learning path: TEM-1 data (`BLAT_ECOLX_Stiffler_2015`) -> MSA conservation ->
burial -> compare against real data -> optional ESM-2 -> figure and 2-page summary.

Excellent for learning. Alone it looks like a tutorial because TEM-1 is the most
common example in the field. It becomes recruiter-worthy once it has a research
question, held-out evaluation across proteins, error analysis and a clean repository,
which is what this document adds.

## Appendix B: Other Project Ideas

1. **"When does ESM-2 fail?" benchmark study:** this is now the headline of Mutation
   Picker, not a separate project.
2. **Edge deployment of an existing CV project (useful for NVIDIA or Samsung):**
   ONNX or TensorRT export, INT8 quantization, latency-vs-accuracy table.
3. **Computer vision meets biology:** colony counting or inhibition-zone
   measurement. Verify the dataset first (for example AGAR) with a 1 to 2 day
   feasibility check.
4. **A merged pull request to an open-source library** (Biopython, scikit-bio or
   ProteinGym tooling): recruiters can verify it.

Suggested order: for edge/CV roles do idea 2 first (1 to 2 weeks), then Mutation
Picker. For enzyme-design internships, make Mutation Picker the flagship.
