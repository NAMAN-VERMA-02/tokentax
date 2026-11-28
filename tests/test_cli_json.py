import json

import pytest

from tokentax import cli, tokenizers


@pytest.fixture(autouse=True)
def fake_tokenizers(monkeypatch):
    registry = {
        "Words": tokenizers.Tokenizer("Words", 100, str.split),
        "Chars": tokenizers.Tokenizer("Chars", 100, list),
    }
    monkeypatch.setattr(tokenizers, "TOKENIZERS", dict.fromkeys(registry, ("fake", "")))
    monkeypatch.setattr(tokenizers, "get", registry.__getitem__)


def test_json_without_comparison(capsys):
    cli.main(["a b", "--json"])
    assert json.loads(capsys.readouterr().out) == [
        {"tokenizer": "Words", "tokens": 2, "vs_tokens": None, "tax": None},
        {"tokenizer": "Chars", "tokens": 3, "vs_tokens": None, "tax": None},
    ]


def test_json_with_comparison(capsys):
    cli.main(["a b", "--vs", "x y z", "-t", "char,word", "--json"])
    assert json.loads(capsys.readouterr().out) == [
        {"tokenizer": "Chars", "tokens": 3, "vs_tokens": 5, "tax": 3 / 5},
        {"tokenizer": "Words", "tokens": 2, "vs_tokens": 3, "tax": 2 / 3},
    ]


@pytest.mark.parametrize("vs", ["", " "])
def test_json_with_zero_token_comparison(vs, capsys):
    cli.main(["a b", "--vs", vs, "-t", "word", "--json"])
    assert json.loads(capsys.readouterr().out) == [
        {"tokenizer": "Words", "tokens": 2, "vs_tokens": 0, "tax": None},
    ]


@pytest.mark.parametrize(("args", "expected"), [
    ([], "tokenizer  tokens\nWords       2\n"),
    (["--vs", "x y z"], "tokenizer  tokens  vs   tax\nWords       2     3  0.67x\n"),
    (["--vs", ""], "tokenizer  tokens  vs   tax\nWords       2     0     -\n"),
])
def test_default_table(args, expected, capsys):
    cli.main(["a b", "-t", "word", *args])
    assert capsys.readouterr().out == expected
