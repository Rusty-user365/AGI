"""Trivial baselines and linear probes — the benchmark-validity gate (Exp 0).

The purpose of this module is adversarial: it tries to solve the paper's tasks
without any of the architecture. If a linear probe on hashed surface features
matches the scope head, then the scope head is reading a token and the paper's
scope claim is unsupported. That is a cheap, decisive check and it should run
before any GPU time is spent.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict

import torch


# --------------------------------------------------------------------------
# Feature maps
# --------------------------------------------------------------------------

def hashed_features(texts: list[str], dim: int = 512, seed: int = 0) -> torch.Tensor:
    """Signed hashed bag-of-words. No learning, no tokenizer dependency."""
    X = torch.zeros(len(texts), dim)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            h = int(hashlib.md5(f"{seed}:{w}".encode()).hexdigest()[:8], 16)
            X[i, h % dim] += 1.0 if (h >> 9) & 1 else -1.0
    return X


def hashed_bigrams(texts: list[str], dim: int = 512, seed: int = 0) -> torch.Tensor:
    """Adds order information, which is what a scope marker exploits."""
    X = torch.zeros(len(texts), dim)
    for i, t in enumerate(texts):
        ws = t.lower().split()
        grams = ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]
        for w in grams:
            h = int(hashlib.md5(f"{seed}:{w}".encode()).hexdigest()[:8], 16)
            X[i, h % dim] += 1.0 if (h >> 9) & 1 else -1.0
    return X


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------

class MajorityBaseline:
    def __init__(self):
        self.mode = None

    def fit(self, y):
        self.mode = Counter(int(v) for v in y).most_common(1)[0][0]
        return self

    def predict(self, n: int):
        return [self.mode] * n

    def accuracy(self, y) -> float:
        return sum(1 for v in y if int(v) == self.mode) / max(len(y), 1)


class RidgeClassifier:
    """Closed-form ridge regression on one-hot targets. A linear probe."""

    def __init__(self, n_classes: int, lam: float = 1.0):
        self.n_classes = n_classes
        self.lam = lam
        self.W = None

    def fit(self, X: torch.Tensor, y) -> "RidgeClassifier":
        X = X.float()
        Y = torch.zeros(len(y), self.n_classes)
        for i, v in enumerate(y):
            Y[i, int(v)] = 1.0
        Xb = torch.cat([X, torch.ones(len(y), 1)], dim=1)
        d = Xb.shape[1]
        A = Xb.T @ Xb + self.lam * torch.eye(d)
        B = Xb.T @ Y
        self.W = torch.linalg.solve(A, B)
        return self

    def predict(self, X: torch.Tensor) -> list[int]:
        Xb = torch.cat([X.float(), torch.ones(X.shape[0], 1)], dim=1)
        return (Xb @ self.W).argmax(-1).tolist()

    def accuracy(self, X, y) -> float:
        pred = self.predict(X)
        return sum(1 for p, t in zip(pred, y) if p == int(t)) / max(len(y), 1)


class NgramLM:
    """Order-2 character/word model with backoff. Scores documents for fluency."""

    def __init__(self, order: int = 2):
        self.order = order
        self.counts = defaultdict(Counter)
        self.unigram = Counter()
        self.total = 0

    def fit(self, texts: list[str]) -> "NgramLM":
        for t in texts:
            ws = ["<s>"] * (self.order - 1) + t.lower().split() + ["</s>"]
            for i in range(self.order - 1, len(ws)):
                ctx = tuple(ws[i - self.order + 1:i])
                self.counts[ctx][ws[i]] += 1
                self.unigram[ws[i]] += 1
                self.total += 1
        return self

    def logprob(self, text: str) -> float:
        ws = ["<s>"] * (self.order - 1) + text.lower().split() + ["</s>"]
        lp = 0.0
        for i in range(self.order - 1, len(ws)):
            ctx = tuple(ws[i - self.order + 1:i])
            c = self.counts.get(ctx)
            w = ws[i]
            if c and c[w] > 0:
                lp += torch.log(torch.tensor(c[w] / sum(c.values()))).item()
            else:
                lp += torch.log(torch.tensor(
                    (self.unigram.get(w, 0) + 1) / (self.total + len(self.unigram) + 1))).item()
        return lp / max(len(ws), 1)


def probe_report(texts_train, y_train, texts_test, y_test, n_classes: int,
                 use_bigrams: bool = True) -> dict:
    """Fit a linear probe and a majority baseline; return both accuracies."""
    f = hashed_bigrams if use_bigrams else hashed_features
    Xtr, Xte = f(texts_train), f(texts_test)
    probe = RidgeClassifier(n_classes).fit(Xtr, y_train)
    maj = MajorityBaseline().fit(y_train)
    return {
        "probe_acc": probe.accuracy(Xte, y_test),
        "majority_acc": maj.accuracy(y_test),
        "n_test": len(y_test),
    }
