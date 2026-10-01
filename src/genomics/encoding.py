"""DNA sequence encoding: one-hot and k-mer frequency features."""
from itertools import product
from typing import List

import numpy as np

BASES = "ACGT"
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(sequence: str, seq_length: int = None) -> np.ndarray:
    """
    Encodes a DNA sequence as a (L, 4) one-hot array. Unknown bases (e.g. 'N')
    are encoded as all-zero rows. Sequences are padded/truncated to
    `seq_length` if provided.
    """
    seq = sequence.upper()
    if seq_length is not None:
        seq = seq[:seq_length].ljust(seq_length, "N")

    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in BASE_TO_IDX:
            arr[i, BASE_TO_IDX[base]] = 1.0
    return arr


def batch_one_hot_encode(sequences: List[str], seq_length: int) -> np.ndarray:
    return np.stack([one_hot_encode(s, seq_length) for s in sequences])


def _build_kmer_vocab(k: int) -> List[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]


def kmer_frequency_encode(sequence: str, k: int = 4) -> np.ndarray:
    """
    Encodes a sequence as a normalized k-mer frequency vector of length 4^k.
    """
    vocab = _build_kmer_vocab(k)
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float32)

    seq = sequence.upper()
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if kmer in vocab_index:
            counts[vocab_index[kmer]] += 1
            total += 1

    if total > 0:
        counts /= total
    return counts


def batch_kmer_encode(sequences: List[str], k: int = 4) -> np.ndarray:
    return np.stack([kmer_frequency_encode(s, k) for s in sequences])
