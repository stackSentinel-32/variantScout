import numpy as np
import pytest

from src.esm import (
    AVAILABLE_MODELS,
    MODEL_NAME,
    _load_model,
    _model_cache,
    esm_mutation_scores,
    score_sequence,
)


def test_available_models():
    """Verify available models and default configuration."""
    assert "35M" in AVAILABLE_MODELS
    assert "650M" in AVAILABLE_MODELS
    assert AVAILABLE_MODELS["35M"] == MODEL_NAME
    assert "facebook/esm2_t12_35M_UR50D" in AVAILABLE_MODELS["35M"]
    assert "facebook/esm2_t33_650M_UR50D" in AVAILABLE_MODELS["650M"]


def test_model_caching():
    """Verify model weights are cached per model key."""
    tok, model = _load_model(MODEL_NAME)
    assert tok is not None
    assert model is not None
    assert f"{MODEL_NAME}_tok" in _model_cache
    assert f"{MODEL_NAME}_model" in _model_cache


def test_esm_mutation_scores_computation():
    """Test log-odds mutation scoring calculation."""
    tok, _ = _load_model(MODEL_NAME)
    # Mock logprobs matrix: 5 residues, vocab_size
    L = 5
    vocab = tok.vocab_size
    mock_logprobs = np.zeros((L, vocab), dtype=np.float32)

    # Set specific logprobs for WT and MT tokens
    m_id = tok.convert_tokens_to_ids("M")
    t_id = tok.convert_tokens_to_ids("T")

    # Position 1 (0-based col index 0): mutant T has higher logprob than WT M
    mock_logprobs[0, m_id] = -2.5
    mock_logprobs[0, t_id] = -0.5

    scores = esm_mutation_scores(
        logprobs=mock_logprobs,
        tok=tok,
        pos=[1],
        wt=["M"],
        mt=["T"],
    )
    # Score should be log P(mt) - log P(wt) = -0.5 - (-2.5) = +2.0
    assert len(scores) == 1
    assert np.isclose(scores[0], 2.0, atol=1e-5)
