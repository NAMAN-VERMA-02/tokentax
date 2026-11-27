# Contributing

Thanks for helping. Most useful contributions are small: one new tokenizer, one new language, one fixed number.

## Setup

```bash
git clone https://github.com/NAMAN-VERMA-02/tokentax
cd tokentax
pip install -e ".[dev]"
pytest -q            # no network needed
tokentax bench       # downloads FLORES-200 (25 MB) + tokenizer files, rewrites results/
```

## Adding a tokenizer

1. Add one line to `TOKENIZERS` in `tokentax/tokenizers.py`. Use `("tiktoken", "<encoding>")` or `("hf", "<repo id>")`. The HF repo must have a `tokenizer.json` and must not be gated, so anyone can rerun it. For gated models, look for an ungated mirror that has the same tokenizer files.
2. Check it loads: `tokentax "hello" -t "<your name>"`.
3. Run `tokentax bench` and commit the updated `results/` and `docs/results.json`.
4. If the findings in the README change, update the numbers there **from the new CSV**. Never type a number you did not get from a run.

## Adding a language

Add one line to `LANGUAGES` in `tokentax/data.py`, mapping a display name to a [FLORES-200 code](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200). Then follow steps 3–4 above.

## Rules

- Every number in the README must come from `results/results.csv` or from `pytest` output.
- Tests must not need the network, so they pass on forks and in CI.
- Keep functions small and code plain. A comment should explain *why* something is done, not *what* the line does.
- One topic per PR.

## Ideas

Look at issues labelled [`good first issue`](https://github.com/NAMAN-VERMA-02/tokentax/labels/good%20first%20issue).
