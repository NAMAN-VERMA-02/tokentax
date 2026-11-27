"""The study: token tax of every language vs English, for every tokenizer, on FLORES-200 devtest.

    python -m tokentax.measure            # writes results/results.csv
"""

import csv
import random
import sys
from pathlib import Path

from . import data, tokenizers

CONTEXT = 128_000
BASELINE = "English"


def token_counts(encode, sentences: list[str]) -> list[int]:
    return [len(encode(s)) for s in sentences]


def unk_rate(encode, unk_id, sentences: list[str]) -> float:
    """Share of tokens that are "unknown". A low tax with a high unk rate means text was thrown away, not saved."""
    if unk_id is None:
        return 0.0
    ids = [i for s in sentences for i in encode(s)]
    return sum(i == unk_id for i in ids) / len(ids) if ids else 0.0


def tax(lang_counts: list[int], base_counts: list[int]) -> float:
    """Total tokens in the language / total tokens for the same sentences in the baseline."""
    if len(lang_counts) != len(base_counts):
        raise ValueError("token counts must be for the same parallel sentences")
    if not base_counts or sum(base_counts) == 0:
        raise ValueError("baseline has no tokens")
    return sum(lang_counts) / sum(base_counts)


def bootstrap_ci(lang_counts, base_counts, draws=1000, seed=0) -> tuple[float, float]:
    """95% CI of the tax by resampling sentence pairs (keeps each pair together)."""
    rng = random.Random(seed)
    n = len(base_counts)
    stats = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(base_counts[i] for i in idx)
        stats.append(sum(lang_counts[i] for i in idx) / b if b else float("inf"))
    stats.sort()
    return stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]


def utf8_bytes(sentences: list[str]) -> int:
    return sum(len(s.encode("utf-8")) for s in sentences)


def run(out: Path, tokenizer_names=None, languages=None) -> list[dict]:
    tokenizer_names = tokenizer_names or list(tokenizers.TOKENIZERS)
    languages = languages or list(data.LANGUAGES)
    texts = {lang: data.load(data.LANGUAGES[lang]) for lang in languages + [BASELINE]}
    base_bytes = utf8_bytes(texts[BASELINE])

    rows = []
    for tname in tokenizer_names:
        try:
            tok = tokenizers.get(tname)
        except Exception as e:  # a missing/renamed HF repo should not kill the whole study
            print(f"skip {tname}: {e}", file=sys.stderr)
            continue
        base = token_counts(tok.encode, texts[BASELINE])
        for lang in languages:
            counts = base if lang == BASELINE else token_counts(tok.encode, texts[lang])
            lo, hi = bootstrap_ci(counts, base)
            rows.append({
                "tokenizer": tname,
                "vocab_size": tok.vocab_size,
                "language": lang,
                "tokens": sum(counts),
                "tax": round(tax(counts, base), 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "unk_rate": round(unk_rate(tok.encode, tok.unk_id, texts[lang]), 4),
                "byte_premium": round(utf8_bytes(texts[lang]) / base_bytes, 4),
                "chars_per_token": round(sum(map(len, texts[lang])) / sum(counts), 3),
                "sentences_in_128k": int(CONTEXT / (sum(counts) / len(counts))),
            })
        print(f"done {tname}", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    run(Path("results/results.csv"))
