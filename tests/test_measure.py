import pytest

from tokentax import data, measure, tokenizers


def words(s):
    return s.split()


def test_baseline_tax_is_one():
    counts = measure.token_counts(words, ["a b c", "d e"])
    assert measure.tax(counts, counts) == 1.0


def test_tax_is_ratio_of_totals_not_mean_of_ratios():
    base = [1, 9]
    lang = [3, 9]  # mean of per-sentence ratios would be 2.0
    assert measure.tax(lang, base) == pytest.approx(12 / 10)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        measure.tax([1, 2], [1])


def test_empty_baseline_rejected():
    with pytest.raises(ValueError):
        measure.tax([], [])


def test_ci_brackets_point_estimate():
    base = [5, 7, 3, 9, 4, 6, 8, 2] * 20
    lang = [11, 13, 8, 20, 9, 12, 19, 5] * 20
    lo, hi = measure.bootstrap_ci(lang, base, draws=500)
    assert lo <= measure.tax(lang, base) <= hi
    assert lo < hi


def test_ci_is_deterministic():
    base, lang = [1, 2, 3, 4], [2, 3, 7, 4]
    assert measure.bootstrap_ci(lang, base, seed=1) == measure.bootstrap_ci(lang, base, seed=1)


def test_utf8_bytes_counts_devanagari_as_three():
    assert measure.utf8_bytes(["a"]) == 1
    assert measure.utf8_bytes(["क"]) == 3


def test_romanize_hindi_is_ascii_lowercase():
    out = data.romanize_hindi("नमस्ते दुनिया")
    assert out == "namaste duniya"


def test_unknown_tokenizer_names_the_known_ones():
    with pytest.raises(KeyError, match="GPT-4o"):
        tokenizers.get("nope")


def test_registry_entries_are_well_formed():
    for name, (kind, ident) in tokenizers.TOKENIZERS.items():
        assert kind in {"tiktoken", "hf"}, name
        assert ident, name
    assert data.LANGUAGES[measure.BASELINE] == "eng_Latn"


def test_unk_rate():
    enc = lambda s: [0 if w == "?" else 1 for w in s.split()]
    assert measure.unk_rate(enc, 0, ["a ? b ?"]) == 0.5
    assert measure.unk_rate(enc, None, ["a ? b ?"]) == 0.0


def test_cli_resolves_fragments():
    from tokentax.cli import resolve

    assert resolve("sarvam") == "Sarvam-1"
    assert resolve("GPT-4o (o200k)") == "GPT-4o (o200k)"
    with pytest.raises(SystemExit):
        resolve("gpt")  # ambiguous: GPT-4 and GPT-4o
