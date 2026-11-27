"""tokentax "मेरा नाम नमन है" --vs "my name is naman"
tokentax "some text" -t gpt-4o,sarvam
tokentax bench          # rerun the full FLORES study + charts
"""

import argparse
import sys
from pathlib import Path

from . import tokenizers


def resolve(query: str) -> str:
    """Exact name, or a unique case-insensitive substring of one ("sarvam" -> "Sarvam-1")."""
    if query in tokenizers.TOKENIZERS:
        return query
    hits = [n for n in tokenizers.TOKENIZERS if query.lower() in n.lower()]
    if len(hits) != 1:
        raise SystemExit(f"'{query}' matches {hits or 'nothing'}; known: {', '.join(tokenizers.TOKENIZERS)}")
    return hits[0]


def count(text: str, vs: str | None, names: list[str]) -> list[tuple[str, int, int | None]]:
    out = []
    for name in names:
        tok = tokenizers.get(name)
        out.append((name, len(tok.encode(text)), len(tok.encode(vs)) if vs else None))
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["bench"]:
        from . import measure, plots

        csv = Path("results/results.csv")
        measure.run(csv)
        plots.main(csv)
        print(f"wrote {csv}, charts in {csv.parent}/, docs/results.json")
        return

    p = argparse.ArgumentParser(prog="tokentax", description="Count tokens for your text across LLM tokenizers.")
    p.add_argument("text")
    p.add_argument("--vs", help="the same text in another language (e.g. English) to get the tax")
    p.add_argument("-t", "--tokenizers", help="comma-separated names or fragments (default: all)")
    a = p.parse_args(argv)
    names = [resolve(q.strip()) for q in a.tokenizers.split(",")] if a.tokenizers else list(tokenizers.TOKENIZERS)

    width = max(map(len, names))
    print(f"{'tokenizer':<{width}}  tokens" + ("  vs   tax" if a.vs else ""))
    for name, n, m in count(a.text, a.vs, names):
        line = f"{name:<{width}}  {n:>6}"
        if m is not None:
            line += f"  {m:>4}  {n / m:.2f}x" if m else f"  {m:>4}     -"
        print(line)


if __name__ == "__main__":
    main()
