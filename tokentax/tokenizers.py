"""Tokenizer registry. To add one, add a line to TOKENIZERS: display name -> (kind, id)."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

# kind "tiktoken": id is an encoding name. kind "hf": id is a Hugging Face repo (only tokenizer files are fetched).
TOKENIZERS = {
    "GPT-4 (cl100k)": ("tiktoken", "cl100k_base"),
    "GPT-4o (o200k)": ("tiktoken", "o200k_base"),
    "Llama 3": ("hf", "unsloth/Llama-3.2-1B"),
    "Gemma 2": ("hf", "unsloth/gemma-2-2b"),
    "Qwen 2.5": ("hf", "Qwen/Qwen2.5-7B"),
    "DeepSeek-V3": ("hf", "deepseek-ai/DeepSeek-V3"),
    "Sarvam-1": ("hf", "sarvamai/sarvam-1"),
    "mBERT": ("hf", "bert-base-multilingual-cased"),
}


@dataclass
class Tokenizer:
    name: str
    vocab_size: int
    encode: Callable[[str], list[int]]
    unk_id: int | None = None  # id of the "unknown" token; None if the tokenizer can't lose text (byte-level)


@lru_cache(maxsize=None)
def get(name: str) -> Tokenizer:
    if name not in TOKENIZERS:
        raise KeyError(f"unknown tokenizer {name!r}; known: {', '.join(TOKENIZERS)}")
    kind, ident = TOKENIZERS[name]
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(ident)
        # disallowed_special=() so text that happens to contain "<|endoftext|>" is counted, not rejected
        return Tokenizer(name, enc.n_vocab, lambda s: enc.encode(s, disallowed_special=()))
    if kind == "hf":
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_pretrained(ident)  # fetches tokenizer.json only, no weights
        # no BOS/CLS/SEP: we count the text, not the model's wrapper tokens
        unk = getattr(tok.model, "unk_token", None)
        unk_id = tok.token_to_id(unk) if unk else None
        return Tokenizer(name, tok.get_vocab_size(), lambda s: tok.encode(s, add_special_tokens=False).ids, unk_id)
    raise ValueError(f"unknown tokenizer kind {kind!r}")
