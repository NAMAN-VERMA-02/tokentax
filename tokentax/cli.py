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


def read_file(path: Path, parser: argparse.ArgumentParser) -> str:
    try:
        # Keep line endings intact because they can affect token counts.
        with path.open(encoding="utf-8", newline="") as source:
            return source.read()
    except (OSError, UnicodeError) as exc:
        parser.error(f"could not read {path} as UTF-8: {exc}")


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
    original = p.add_mutually_exclusive_group(required=True)
    original.add_argument("text", nargs="?", help="literal text, or '-' to read UTF-8 from stdin")
    original.add_argument("-f", "--file", type=Path, help="read the text from a UTF-8 file")
    comparison = p.add_mutually_exclusive_group()
    comparison.add_argument("--vs", help="the same text in another language (e.g. English) to get the tax")
    comparison.add_argument("--vs-file", type=Path, help="read the comparison text from a UTF-8 file")
    p.add_argument("-t", "--tokenizers", help="comma-separated names or fragments (default: all)")
    a = p.parse_args(argv)
    names = [resolve(q.strip()) for q in a.tokenizers.split(",")] if a.tokenizers else list(tokenizers.TOKENIZERS)

    text = a.text
    if a.file is not None:
        text = read_file(a.file, p)
    elif text == "-":
        try:
            # Decode piped bytes independently of the terminal's locale.
            stream = getattr(sys.stdin, "buffer", None)
            text = stream.read().decode("utf-8") if stream is not None else sys.stdin.read()
        except (OSError, UnicodeError) as exc:
            p.error(f"could not read stdin as UTF-8: {exc}")
    vs = read_file(a.vs_file, p) if a.vs_file is not None else a.vs

    width = max(map(len, names))
    print(f"{'tokenizer':<{width}}  tokens" + ("  vs   tax" if vs else ""))
    for name, n, m in count(text, vs, names):
        line = f"{name:<{width}}  {n:>6}"
        if m is not None:
            line += f"  {m:>4}  {n / m:.2f}x" if m else f"  {m:>4}     -"
        print(line)


if __name__ == "__main__":
    main()
