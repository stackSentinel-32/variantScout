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
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

MODEL_NAME: str = "facebook/esm2_t12_35M_UR50D"
AVAILABLE_MODELS: Dict[str, str] = {
    "35M": "facebook/esm2_t12_35M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}
CACHE_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "processed" / "esm_cache"
MAX_LEN: int = 1022

_model_cache: Dict[str, Any] = {}


def _load_model(model_name: str = MODEL_NAME) -> Tuple[AutoTokenizer, EsmForMaskedLM]:
    tok_key = f"{model_name}_tok"
    model_key = f"{model_name}_model"
    if tok_key not in _model_cache or model_key not in _model_cache:
        print(f"Loading ESM-2 model weights ({model_name}) ...")
        _model_cache[tok_key] = AutoTokenizer.from_pretrained(model_name)
        _model_cache[model_key] = EsmForMaskedLM.from_pretrained(model_name).eval()
    return _model_cache[tok_key], _model_cache[model_key]


def score_sequence(
    seq: str,
    uniprot_id: str,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Return (L, vocab) log-probability matrix, loading from cache if available.

    Rows are sequence positions (0-based). Columns are model vocabulary tokens.
    Call token_to_aa() to convert column indices to amino-acid letters.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = "650M" if "650M" in model_name else "35M"
    cache_path = CACHE_DIR / f"{uniprot_id}_{model_tag}.npy"
    legacy_path = CACHE_DIR / f"{uniprot_id}.npy"

    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(seq):
            return arr
    if model_tag == "35M" and legacy_path.exists():
        arr = np.load(legacy_path)
        if arr.shape[0] == len(seq):
            return arr

    assert len(seq) <= MAX_LEN, f"{uniprot_id} has {len(seq)} residues, ESM-2 limit is {MAX_LEN}"

    tok, model = _load_model(model_name)
    ids = tok(seq, return_tensors="pt")["input_ids"]  # shape [1, L+2]
    L = len(seq)
    logprobs = np.zeros((L, tok.vocab_size), dtype=np.float32)

    # For larger models like 650M on CPU, reduce batch_size to save memory
    effective_batch = 8 if "650M" in model_name else batch_size

    for start in range(0, L, effective_batch):
        end = min(start + effective_batch, L)
        batch = ids.repeat(end - start, 1)
        for i, pos in enumerate(range(start, end)):
            batch[i, pos + 1] = tok.mask_token_id   # +1 skips the <cls> start token

        with torch.no_grad():
            logits = model(batch).logits             # [batch, L+2, vocab]

        for i, pos in enumerate(range(start, end)):
            logprobs[pos] = torch.log_softmax(logits[i, pos + 1], dim=-1).numpy()

    np.save(cache_path, logprobs)
    if model_tag == "35M" and not legacy_path.exists():
        np.save(legacy_path, logprobs)
    print(f"  cached ESM-2 ({model_tag}) scores for {uniprot_id}")
    return logprobs


def esm_mutation_scores(
    logprobs: np.ndarray,
    tok: AutoTokenizer,
    pos: Union[Sequence[int], np.ndarray],
    wt: Union[Sequence[str], np.ndarray],
    mt: Union[Sequence[str], np.ndarray],
) -> np.ndarray:
    """Log-odds score for each mutation given the (L, vocab) log-prob matrix.

    pos: 1-based DMS positions
    wt, mt: wild-type and mutant amino-acid series
    """
    wt_ids = tok.convert_tokens_to_ids(list(wt))
    mt_ids = tok.convert_tokens_to_ids(list(mt))
    cols = np.asarray(pos) - 1                       # 0-based row index
    return logprobs[cols, mt_ids] - logprobs[cols, wt_ids]


def compare_to_proteingym(df: pd.DataFrame, zero_shot_csv: Path) -> float:
    """Compare your ESM-2 Spearman to ProteinGym's published value.

    zero_shot_csv: path to the ESM2 prediction CSV from ProteinGym's zero-shot
                   scores zip (data/raw/zero_shot/<assay_id>.csv).

    Returns the Spearman of ProteinGym's scores against DMS_score.
    """
    from scipy.stats import spearmanr
    ref = pd.read_csv(zero_shot_csv)
    merged = df.merge(ref[["mutant", "score"]], on="mutant", how="inner")
    rho = float(spearmanr(merged["score"], merged["DMS_score"]).correlation)
    print(f"ProteinGym ESM2 Spearman on this assay: {rho:.3f}")
    return rho

