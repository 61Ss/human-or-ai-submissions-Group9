"""Hand-crafted stylometric features shared by the EDA and the style models."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ZERO_WIDTH_RE = re.compile("[​-‏⁠﻿]")
# A "word" mixing Latin with Cyrillic/Greek letters is a classic homoglyph attack.
HOMOGLYPH_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[Ͱ-ϿЀ-ӿ])\w+\b")
PUNCT = {
    "comma": ",", "semicolon": ";", "colon": ":", "exclam": "!", "question": "?",
    "dquote": '"', "squote": "'", "paren": "(", "hyphen": "-", "emdash": "—",
    "endash": "–", "ellipsis": "…", "curly_quote": "“”‘’",
}

# Features that only describe line layout. EDA: a blank line almost always means AI in train,
# which may be a collection artefact rather than a property of authorship.
LAYOUT_FEATURES = ["blank_lines", "n_lines", "avg_line_len", "markdown_marks"]
LINE_BREAK_RE = re.compile(r"\s*\n\s*")


def strip_layout(text):
    """Replace every line break (and blank line) with one space; used as a robustness check."""
    return LINE_BREAK_RE.sub(" ", text)


def style_features(text):
    words = WORD_RE.findall(text)
    n_words = max(len(words), 1)
    lower = [w.lower() for w in words]
    sents = [s for s in SENT_SPLIT_RE.split(text.strip()) if s]
    sent_lens = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    n_chars = max(len(text), 1)
    lines = text.split("\n")
    feats = {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "n_sentences": len(sents),
        "avg_sent_len": sent_lens.mean(),
        "std_sent_len": sent_lens.std(),
        "type_token_ratio": len(set(lower)) / n_words,
        "stopword_ratio": sum(w in ENGLISH_STOP_WORDS for w in lower) / n_words,
        "n_lines": len(lines),
        "blank_lines": sum(1 for l in lines if not l.strip()),
        "avg_line_len": np.mean([len(l) for l in lines]),
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "upper_ratio": sum(c.isupper() for c in text) / n_chars,
        "non_ascii_ratio": sum(ord(c) > 127 for c in text) / n_chars,
        "first_person": sum(w in {"i", "me", "my", "mine", "we", "our", "us"} for w in lower) / n_words,
        "ends_terminal_punct": float(text.rstrip()[-1:] in ".!?\"')"),
        "starts_lowercase": float(text[:1].islower()),
        "trailing_ws": float(text != text.rstrip()),
        "double_space": text.count("  "),
        "markdown_marks": len(re.findall(r"^\s*(?:[#>*-]|\d+\.)\s", text, flags=re.M)),
        "zero_width": len(ZERO_WIDTH_RE.findall(text)),
        "homoglyph_words": len(HOMOGLYPH_RE.findall(text)),
    }
    for name, chars in PUNCT.items():
        feats[f"p_{name}"] = sum(text.count(c) for c in chars) / n_words
    return feats


def build_features(df):
    return pd.DataFrame([style_features(t) for t in df.text], index=df.index)
