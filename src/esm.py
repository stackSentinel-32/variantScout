"""ESM-2 masked-marginal scoring.

For each position in the sequence, the model sees every other residue and
predicts which amino acid belongs there. The mutation score is:

    log P(mutant) − log P(wild-type)   (at the masked position)

Results are cached to data/processed/esm_cache/ so the model runs at most
once per protein.

Validation (do this before trusting any numbers):
  Download the ProteinGym zero-shot scores zip from proteingym.org and place
  the ESM2 prediction CSV for your assay in data/raw/zero_shot/. Then run
  compare_to_proteingym() to check that your Spearman matches theirs within ~0.02.
  A larger gap means an off-by-one in the token positions.
"""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "esm_cache"
MAX_LEN = 1022

_model_cache: dict = {}


def _load_model():
    if "tok" not in _model_cache:
        _model_cache["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model_cache["model"] = EsmForMaskedLM.from_pretrained(MODEL_NAME).eval()
    return _model_cache["tok"], _model_cache["model"]


def score_sequence(seq: str, uniprot_id: str, batch_size: int = 32) -> np.ndarray:
    """Return (L, vocab) log-probability matrix, loading from cache if available.

    Rows are sequence positions (0-based). Columns are model vocabulary tokens.
    Call token_to_aa() to convert column indices to amino-acid letters.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{uniprot_id}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    assert len(seq) <= MAX_LEN, f"{uniprot_id} has {len(seq)} residues, ESM-2 limit is {MAX_LEN}"

    tok, model = _load_model()
    ids = tok(seq, return_tensors="pt")["input_ids"]  # shape [1, L+2]
    L = len(seq)
    logprobs = np.zeros((L, tok.vocab_size), dtype=np.float32)

    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        batch = ids.repeat(end - start, 1)
        for i, pos in enumerate(range(start, end)):
            batch[i, pos + 1] = tok.mask_token_id   # +1 skips the <cls> start token

        with torch.no_grad():
            logits = model(batch).logits             # [batch, L+2, vocab]

        for i, pos in enumerate(range(start, end)):
            logprobs[pos] = torch.log_softmax(logits[i, pos + 1], dim=-1).numpy()

    np.save(cache_path, logprobs)
    print(f"  cached ESM-2 scores for {uniprot_id}")
    return logprobs


def esm_mutation_scores(logprobs: np.ndarray, tok, pos, wt, mt) -> np.ndarray:
    """Log-odds score for each mutation given the (L, vocab) log-prob matrix.

    pos: 1-based DMS positions
    wt, mt: wild-type and mutant amino-acid series
    """
    wt_ids = tok.convert_tokens_to_ids(list(wt))
    mt_ids = tok.convert_tokens_to_ids(list(mt))
    cols = pos - 1                                   # 0-based row index
    return logprobs[cols, mt_ids] - logprobs[cols, wt_ids]


def compare_to_proteingym(df, zero_shot_csv: Path) -> float:
    """Compare your ESM-2 Spearman to ProteinGym's published value.

    zero_shot_csv: path to the ESM2 prediction CSV from ProteinGym's zero-shot
                   scores zip (data/raw/zero_shot/<assay_id>.csv).

    Returns the Spearman of ProteinGym's scores against DMS_score.
    If the difference from your own Spearman is > 0.05, check for an
    off-by-one in token positions (the most common ESM-2 scoring bug).
    """
    from scipy.stats import spearmanr
    ref = __import__("pandas").read_csv(zero_shot_csv)
    merged = df.merge(ref[["mutant", "score"]], on="mutant", how="inner")
    rho = spearmanr(merged["score"], merged["DMS_score"]).correlation
    print(f"ProteinGym ESM2 Spearman on this assay: {rho:.3f}")
    return rho
