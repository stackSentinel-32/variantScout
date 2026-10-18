import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


def load_msa(path):
    # keep every column: lowercase letters in the query are real positions
    blocks = open(path).read().split(">")[1:]
    seqs = ["".join(b.splitlines()[1:]).upper().replace(".", "-") for b in blocks]
    msa = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), -1)
    return seqs[0], msa


def frequencies(msa):
    counts = np.stack([(msa == a.encode()).sum(0) for a in AA], axis=1)
    return (counts + 0.5) / (counts.sum(1, keepdims=True) + 10)


def evo_scores(freq, cols, wt, mt):
    wt_idx = [AA.index(a) for a in wt]
    mt_idx = [AA.index(a) for a in mt]
    return np.log(freq[cols, mt_idx]) - np.log(freq[cols, wt_idx])


def site_entropy(freq):
    return -(freq * np.log(freq)).sum(1)
