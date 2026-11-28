from io import BytesIO, TextIOWrapper
from types import SimpleNamespace

import pytest

from tokentax import cli


@pytest.fixture
def inputs(monkeypatch):
    texts = []

    def encode(text):
        texts.append(text)
        return list(text)

    def get_tokenizer(name):
        assert name == cli.resolve("gpt-4o")
        return SimpleNamespace(encode=encode)

    monkeypatch.setattr(cli.tokenizers, "get", get_tokenizer)
    return texts


def test_literal_input_and_comparison(inputs):
    cli.main(["வணக்கம்", "--vs", "hello", "-t", "gpt-4o"])
    assert inputs == ["வணக்கம்", "hello"]


def test_utf8_file_input_and_comparison(tmp_path, inputs):
    text = "வணக்கம்\r\n"
    translation = "héllo\n"
    primary = tmp_path / "original text.txt"
    baseline = tmp_path / "translation.txt"
    primary.write_bytes(text.encode("utf-8"))
    baseline.write_bytes(translation.encode("utf-8"))

    cli.main(["-f", str(primary), "--vs-file", str(baseline), "-t", "gpt-4o"])

    assert inputs == [text, translation]


def test_stdin_is_utf8_even_with_a_different_locale(monkeypatch, inputs):
    text = "ନମସ୍କାର\r\n"
    with TextIOWrapper(BytesIO(text.encode("utf-8")), encoding="cp1252") as stdin:
        monkeypatch.setattr(cli.sys, "stdin", stdin)
        cli.main(["-", "-t", "gpt-4o"])
    assert inputs == [text]


@pytest.mark.parametrize(
    "args",
    [
        ["literal", "--file", "original.txt"],
        ["literal", "--vs", "hello", "--vs-file", "translation.txt"],
    ],
)
def test_conflicting_input_sources(args, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_invalid_utf8_file_reports_the_source(tmp_path, capsys):
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_bytes(b"\xff")
    with pytest.raises(SystemExit) as error:
        cli.main(["-f", str(bad_file)])
    assert error.value.code == 2
    assert f"could not read {bad_file} as UTF-8" in capsys.readouterr().err
