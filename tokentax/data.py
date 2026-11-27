"""FLORES-200 loader. Every language file holds the same 1012 sentences, line by line."""

import os
import tarfile
import urllib.request
from pathlib import Path

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CACHE = Path(os.environ.get("TOKENTAX_CACHE", Path.home() / ".cache" / "tokentax"))

# display name -> FLORES code. "hin_Latn" is not in FLORES; we make it by transliterating hin_Deva.
LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Hindi (romanized)": "hin_Latn",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Urdu": "urd_Arab",
    "French": "fra_Latn",
    "Chinese": "zho_Hans",
}


def _dataset_dir() -> Path:
    root = CACHE / "flores200_dataset"
    if root.exists():
        return root
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "flores200_dataset.tar.gz"
    if not archive.exists():
        print(f"downloading FLORES-200 (25 MB) to {CACHE} ...")
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(FLORES_URL, tmp)
        tmp.rename(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(CACHE, filter="data")
    return root


def romanize_hindi(text: str) -> str:
    """Devanagari -> casual Latin ("namaste duniya"). Machine output, not how real people type Hinglish."""
    from indic_transliteration import sanscript

    return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


def load(code: str, split: str = "devtest") -> list[str]:
    if code == "hin_Latn":
        return [romanize_hindi(s) for s in load("hin_Deva", split)]
    path = _dataset_dir() / split / f"{code}.{split}"
    if not path.exists():
        raise FileNotFoundError(f"no FLORES file for {code!r} ({path})")
    return path.read_text(encoding="utf-8").splitlines()
