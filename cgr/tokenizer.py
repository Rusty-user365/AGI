"""Word-level tokenizer built from the corpus.

The draft specified a 32k vocabulary. With ~20 template sentences that is
absurd: 32k rows of embedding for a corpus whose entire type inventory is a
few hundred words, and it makes the "10M parameter" claim mostly embedding
table. We build the vocabulary from the corpus instead and report the real
count in the model card.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAD, MASK, BOS, EOS, SEP = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<mask>", "<bos>", "<eos>", "<sep>"]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Tokenizer:
    def __init__(self, words: list[str]):
        self.itos = list(SPECIALS) + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = PAD
        self.mask_id = MASK
        self.bos_id = BOS
        self.eos_id = EOS
        self.sep_id = SEP

    # ---------------------------------------------------------------- build
    @staticmethod
    def build(texts: list[str], min_count: int = 1) -> "Tokenizer":
        counts: dict[str, int] = {}
        for t in texts:
            for w in tokenize(t):
                counts[w] = counts.get(w, 0) + 1
        words = sorted(w for w, c in counts.items() if c >= min_count)
        return Tokenizer(words)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ---------------------------------------------------------------- codec
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(w, self.mask_id) for w in tokenize(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in (PAD, MASK, BOS, EOS) or i >= len(self.itos):
                continue
            if i == SEP:
                out.append(" |")
            else:
                out.append(self.itos[i])
        return " ".join(out).replace(" |", " |")

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.itos), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Tokenizer":
        words = json.loads(Path(path).read_text(encoding="utf-8"))
        return Tokenizer(words[len(SPECIALS):])
