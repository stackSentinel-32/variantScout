from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

AA: str = "ACDEFGHIKLMNPQRSTVWY"


def load_msa(path: Union[str, Path]) -> Tuple[str, np.ndarray]:
    """Load an A2M/FASTA MSA into query string and numpy array of character bytes."""
    blocks = open(path).read().split(">")[1:]
    seqs = ["".join(b.splitlines()[1:]).upper().replace(".", "-") for b in blocks]
    msa = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), -1)
    return seqs[0], msa


def frequencies(msa: np.ndarray) -> np.ndarray:
    """Compute pseudocount-regularized amino-acid frequencies at each alignment column."""
    counts = np.stack([(msa == a.encode()).sum(0) for a in AA], axis=1)
    return (counts + 0.5) / (counts.sum(1, keepdims=True) + 10)


def evo_scores(
    freq: np.ndarray,
    cols: Union[Sequence[int], np.ndarray],
    wt: Sequence[str],
    mt: Sequence[str],
) -> np.ndarray:
    """Log-odds evolutionary substitution scores: log P(mutant) - log P(wildtype)."""
    wt_idx = [AA.index(a) for a in wt]
    mt_idx = [AA.index(a) for a in mt]
    return np.log(freq[cols, mt_idx]) - np.log(freq[cols, wt_idx])


def site_entropy(freq: np.ndarray) -> np.ndarray:
    """Calculate Shannon entropy for each column in the frequency matrix."""
    return -(freq * np.log(freq)).sum(1)


def compute_neff(msa: np.ndarray, theta: float = 0.8) -> float:
    """Compute sequence-reweighted number of effective sequences (Neff) at identity threshold theta.

    Weights each sequence inversely by the number of sequences within (1 - theta)
    identity distance. For large MSAs (N > 1500), a random subsample is used
    to estimate Neff efficiently without excessive memory usage.
    """
    N, L = msa.shape
    if N <= 1:
        return float(N)

    max_sample = 1500
    if N > max_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=max_sample, replace=False)
        sub = msa[idx]
        arr = np.ascontiguousarray(sub.view(np.uint8).reshape(max_sample, L))
        weights = np.zeros(max_sample, dtype=np.float64)
        min_matches = int(np.ceil(theta * L))
        for i in range(max_sample):
            matches = (arr[i] == arr).sum(axis=1)
            weights[i] = 1.0 / np.sum(matches >= min_matches)
        return float(weights.sum() * (N / max_sample))
    else:
        arr = np.ascontiguousarray(msa.view(np.uint8).reshape(N, L))
        weights = np.zeros(N, dtype=np.float64)
        min_matches = int(np.ceil(theta * L))
        for i in range(N):
            matches = (arr[i] == arr).sum(axis=1)
            weights[i] = 1.0 / np.sum(matches >= min_matches)
        return float(weights.sum())


compute_neff_score = compute_neff


