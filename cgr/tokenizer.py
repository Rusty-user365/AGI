"""Tokenizers.

Two tokenizers, for two backbones:

  word   the corpus-derived word-level vocabulary used by the blank-canvas
         `MaskedDiffusionLM`. Kept because it makes the CPU tests fast and
         because the C1 result was established on it.
  gpt2   the GPT-2 byte-pair vocabulary that DiffuGPT-S was pre-trained with.
         A pre-trained backbone cannot be paired with a different vocabulary:
         the embedding rows would be unrelated to the weights. The adapter
         therefore delegates to the HF tokenizer and adds one pad token, so
         padded batches have a token that is not also a real document
         boundary.

The draft specified a 32k vocabulary. With ~20 template sentences that is
absurd for the word tokenizer: 32k rows of embedding for a corpus whose entire
type inventory is a few hundred words. The GPT-2 vocabulary is different: it
is fixed by the pre-trained checkpoint, and its size is reported rather than
chosen.
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
    def spec(self) -> dict:
        return {"kind": "word", "itos": list(self.itos)}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.spec()), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Tokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = raw["itos"] if isinstance(raw, dict) else raw
        return Tokenizer(itos[len(SPECIALS):])


class GPT2Tokenizer:
    """HF GPT-2 BPE tokenizer, adapted to the `Tokenizer` interface.

    `mask_id` is the checkpoint's mask token (``"------------"``, id 10541).
    `sep_id` is the newline token, used by `blocked_spans` between blocks.
    A dedicated pad token is added at the end of the vocabulary; the backbone
    resizes its embedding by one row. Using EOS as pad would make the document
    boundary indistinguishable from padding, and the attention mask would then
    hide the real BOS/EOS tokens.

    `decode` uses `clean_up_tokenization_spaces=False`: the template grammar is
    defined on exact spacing (``" , so"``, ``" ."``), and the default cleanup
    removes the space before punctuation, which would make every produced block
    fail the parser for a reason that has nothing to do with the model.
    """

    def __init__(self, path: str | Path):
        from transformers import AutoTokenizer
        self.path = str(path)
        self._tok = AutoTokenizer.from_pretrained(self.path)
        if self._tok.pad_token_id is None:
            # `add_special_tokens` appends to the vocabulary; the DiffuGPT
            # checkpoint has 50257 rows, so the backbone allocates one more.
            self._tok.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_id = int(self._tok.pad_token_id)
        self.mask_id = int(self._tok.mask_token_id)
        self.bos_id = int(self._tok.bos_token_id)
        self.eos_id = int(self._tok.eos_token_id)
        self.sep_id = int(self._tok.encode("\n", add_special_tokens=False)[0])

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def encode(self, text: str, add_bos: bool = True,
               add_eos: bool = True) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        keep = [int(i) for i in ids if int(i) != self.pad_id]
        return self._tok.decode(keep, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)

    # ---------------------------------------------------------------- io
    def spec(self) -> dict:
        return {"kind": "gpt2", "path": self.path}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.spec()), encoding="utf-8")


def make_tokenizer(cfg) -> "Tokenizer | GPT2Tokenizer | None":
    """Return the configured tokenizer, or None for the corpus-built one.

    None is not an error: the word tokenizer's vocabulary is derived from the
    training split, so it cannot be constructed until the documents exist.
    """
    kind = getattr(cfg.model, "tokenizer", "word")
    if kind == "gpt2":
        path = getattr(cfg.model, "backbone_path", None) or "DiffuGPT-S"
        return GPT2Tokenizer(path)
    if kind == "word":
        return None
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def tokenizer_from_spec(spec: dict) -> "Tokenizer | GPT2Tokenizer":
    if spec.get("kind") == "gpt2":
        return GPT2Tokenizer(spec["path"])
    itos = spec.get("itos") or spec
    if isinstance(itos, list) and itos and itos[0] == "<pad>":
        itos = itos[len(SPECIALS):]
    return Tokenizer(itos)


#: Both tokenizers implement the same codec, so callers should accept either.
#: The distinction that matters is vocabulary, not type: a pre-trained backbone
#: can only be paired with the vocabulary it was trained on.
TokenizerLike = Tokenizer | GPT2Tokenizer
