"""
Turns your corpus CSVs into the JSON + PNG files the website reads from /data.

Run this after rhyme_similarity.py / ngram_similarity.py so it can import their
(already-tested) feature-extraction and scoring functions directly -- this
script does not duplicate that logic, it only adds the richer per-occurrence
context (surrounding lines, line numbers) that the website's click-to-expand
panels need, which the console-oriented scripts don't bother with.

Produces:
    data/ngrams.json
    data/rhymes.json
    data/null_calibration.json
    data/rhyme_similarity_distribution.png   (histogram, same as the standalone script's plot)
    data/ngram_similarity_distribution.png   (histogram, same as the standalone script's plot)

Does NOT touch data/embeddings.json -- that one stays as bundled sample data
until you actually train a line-embedding model; see README.

Usage: put this file next to rhyme_similarity.py and ngram_similarity.py,
point INPUT_FOLDER at your corpus, adjust QUESTIONED_MATCHES / labels below,
then: python3 export_for_website.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import rhyme_similarity as rhyme_mod
import ngram_similarity as ngram_mod

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

INPUT_FOLDER = Path("csv_files")
TEXT_COLUMN = "text"

KNOWN_MATCH = "cronycke_van_brabant"
KNOWN_LABEL = "Cornicke"

# (filename substring, display label) for the questioned document
QUESTIONED_MATCHES = [
    ("voortzetting", "De Voortzetting"),
]

TOP_N_TERMS = 20              # shared terms exported per comparison, per method
MAX_OCCURRENCES_PER_TERM = 3  # example occurrences shown per term, per document
CONTEXT_WINDOW_LINES = 2      # lines of surrounding context before/after a match

OUTPUT_FOLDER = Path("data")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def line_window(lines: list[str], match_idx: int, window: int = CONTEXT_WINDOW_LINES) -> dict:
    """Builds the {line_no, lines, match_index} shape the website expects for
    one occurrence of a term, centered on the line at `match_idx`."""
    start = max(0, match_idx - window)
    end = min(len(lines), match_idx + window + 1)
    return {
        "line_no": match_idx + 1,          # 1-based line number of the MATCHED line
        "lines": lines[start:end],
        "match_index": match_idx - start,  # position of the matched line within the window
    }


def find_doc_and_label(doc_names: list[str], match: str) -> int:
    return rhyme_mod.find_doc(doc_names, match)


# --------------------------------------------------------------------------
# Rhyme export
# --------------------------------------------------------------------------

def rhyme_occurrences(doc_name: str, rhyme_word: str, lines_by_doc: dict, max_n: int) -> list[dict]:
    lines = lines_by_doc[doc_name]
    occs = []
    for i, line in enumerate(lines):
        if len(occs) >= max_n:
            break
        tokens = line.strip().split()
        if not tokens:
            continue
        last = rhyme_mod._PUNCT_RE.sub("", tokens[-1]).lower()
        if last == rhyme_word:
            occs.append(line_window(lines, i))
    return occs


def build_rhymes_json() -> dict:
    lines_by_doc, _ = rhyme_mod.load_corpus(INPUT_FOLDER, TEXT_COLUMN, rhyme_mod.AUTHOR_COLUMN)
    Z, counts, doc_names, feature_names, _ = rhyme_mod.build_rhyme_matrix(lines_by_doc)
    sim_matrix = cosine_similarity(Z)
    known_idx = find_doc_and_label(doc_names, KNOWN_MATCH)

    comparisons = []
    for match, label in QUESTIONED_MATCHES:
        q_idx = find_doc_and_label(doc_names, match)
        resp = rhyme_mod.responsible_rhymes(known_idx, q_idx, Z, counts, feature_names, top_n=TOP_N_TERMS)

        items = []
        for _, r in resp.iterrows():
            term = r["rhyme_word"]
            items.append({
                "term": term,
                "type": "rhyme",
                "score": float(r["contribution"]),
                "known": rhyme_occurrences(doc_names[known_idx], term, lines_by_doc, MAX_OCCURRENCES_PER_TERM),
                "questioned": rhyme_occurrences(doc_names[q_idx], term, lines_by_doc, MAX_OCCURRENCES_PER_TERM),
            })

        comparisons.append({
            "questioned_label": label,
            "similarity": float(sim_matrix[known_idx, q_idx]),
            "items": items,
        })

    return {"meta": {"known_label": KNOWN_LABEL}, "comparisons": comparisons}


# --------------------------------------------------------------------------
# N-gram export
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def build_token_stream(lines: list[str]) -> list[tuple[str, int]]:
    """(token, line_idx) pairs in reading order -- mirrors how the TF-IDF
    pseudo-document was built (lines joined with a single space) so n-gram
    positions line up with the vectorizer's own tokenization."""
    stream = []
    for i, line in enumerate(lines):
        for tok in _TOKEN_RE.findall(line.lower()):
            stream.append((tok, i))
    return stream


def ngram_occurrences(lines: list[str], token_stream: list[tuple[str, int]],
                       ngram_tokens: list[str], max_n: int) -> list[dict]:
    k = len(ngram_tokens)
    occs = []
    n = len(token_stream)
    for start in range(n - k + 1):
        if len(occs) >= max_n:
            break
        if all(token_stream[start + j][0] == ngram_tokens[j] for j in range(k)):
            match_line_idx = token_stream[start][1]
            occs.append(line_window(lines, match_line_idx))
    return occs


def build_ngrams_json() -> dict:
    doc_text = ngram_mod.load_corpus(INPUT_FOLDER, TEXT_COLUMN)
    # also keep the ORIGINAL per-line lists (not joined) for the 3 documents we need context for
    lines_by_doc, _ = rhyme_mod.load_corpus(INPUT_FOLDER, TEXT_COLUMN, rhyme_mod.AUTHOR_COLUMN)

    X, doc_names, feature_names = ngram_mod.build_ngram_matrix(doc_text, ngram_mod.EXCLUDE_WORDS)
    sim_matrix = cosine_similarity(X)
    known_idx = find_doc_and_label(doc_names, KNOWN_MATCH)

    # token streams are only needed for the documents we actually show context for
    token_streams = {}
    for match, _ in [(KNOWN_MATCH, KNOWN_LABEL)] + QUESTIONED_MATCHES:
        idx = find_doc_and_label(doc_names, match)
        token_streams[doc_names[idx]] = build_token_stream(lines_by_doc[doc_names[idx]])

    comparisons = []
    for match, label in QUESTIONED_MATCHES:
        q_idx = find_doc_and_label(doc_names, match)
        resp = ngram_mod.responsible_ngrams(known_idx, q_idx, X, feature_names, top_n=TOP_N_TERMS)

        items = []
        for _, r in resp.iterrows():
            term = r["ngram"]
            ngram_tokens = term.split()
            term_type = "bigram" if len(ngram_tokens) == 2 else "trigram" if len(ngram_tokens) == 3 else "n-gram"
            items.append({
                "term": term,
                "type": term_type,
                "score": float(r["contribution"]),
                "known": ngram_occurrences(lines_by_doc[doc_names[known_idx]], token_streams[doc_names[known_idx]],
                                            ngram_tokens, MAX_OCCURRENCES_PER_TERM),
                "questioned": ngram_occurrences(lines_by_doc[doc_names[q_idx]], token_streams[doc_names[q_idx]],
                                                 ngram_tokens, MAX_OCCURRENCES_PER_TERM),
            })

        comparisons.append({
            "questioned_label": label,
            "similarity": float(sim_matrix[known_idx, q_idx]),
            "items": items,
        })

    return {"meta": {"known_label": KNOWN_LABEL}, "comparisons": comparisons}


# --------------------------------------------------------------------------
# Calibration export (background = known vs every other single document)
# --------------------------------------------------------------------------

def calibration_block(sim_matrix: np.ndarray, doc_names: list[str], known_idx: int) -> dict:
    background = [{"document": name, "score": float(sim_matrix[known_idx, i])}
                  for i, name in enumerate(doc_names) if i != known_idx]
    scores = np.array([b["score"] for b in background])

    comparisons = []
    for match, label in QUESTIONED_MATCHES:
        q_idx = find_doc_and_label(doc_names, match)
        score = float(sim_matrix[known_idx, q_idx])
        percentile = float((scores < score).mean() * 100)
        comparisons.append({"questioned_label": label, "score": score, "percentile": round(percentile, 1)})

    return {"known_label": KNOWN_LABEL, "background": background, "comparisons": comparisons}


def build_calibration_json_and_plots() -> dict:
    lines_by_doc, _ = rhyme_mod.load_corpus(INPUT_FOLDER, TEXT_COLUMN, rhyme_mod.AUTHOR_COLUMN)
    Z, _, doc_names_r, _, _ = rhyme_mod.build_rhyme_matrix(lines_by_doc)
    rhyme_sim = cosine_similarity(Z)
    known_idx_r = find_doc_and_label(doc_names_r, KNOWN_MATCH)

    doc_text = ngram_mod.load_corpus(INPUT_FOLDER, TEXT_COLUMN)
    X, doc_names_n, _ = ngram_mod.build_ngram_matrix(doc_text, ngram_mod.EXCLUDE_WORDS)
    ngram_sim = cosine_similarity(X)
    known_idx_n = find_doc_and_label(doc_names_n, KNOWN_MATCH)

    # Histogram PNGs for the site -- same plot_distribution() used by the
    # standalone scripts, so this is exactly what running them yourself
    # produces. Only the real QUESTIONED_MATCHES comparisons are marked; there
    # is no automatic "best match" marker.
    rhyme_background = rhyme_mod.similarity_background(rhyme_sim)
    rhyme_markers = {f"{KNOWN_LABEL} vs. {label}  (sim={rhyme_sim[known_idx_r, find_doc_and_label(doc_names_r, match)]:.3f})": rhyme_sim[known_idx_r, find_doc_and_label(doc_names_r, match)] for match, label in QUESTIONED_MATCHES}
    rhyme_mod.plot_distribution(rhyme_background, rhyme_markers, OUTPUT_FOLDER / "rhyme_similarity_distribution.png")

    ngram_background = ngram_mod.similarity_background(ngram_sim)
    ngram_markers = {f"{KNOWN_LABEL} vs. {label}  (sim={ngram_sim[known_idx_n, find_doc_and_label(doc_names_n, match)]:.3f})": ngram_sim[known_idx_n, find_doc_and_label(doc_names_n, match)] for match, label in QUESTIONED_MATCHES}
    ngram_mod.plot_distribution(ngram_background, ngram_markers, OUTPUT_FOLDER / "ngram_similarity_distribution.png")

    return {
        "rhymes": calibration_block(rhyme_sim, doc_names_r, known_idx_r),
        "ngrams": calibration_block(ngram_sim, doc_names_n, known_idx_n),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run() -> None:
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    print("Building rhymes.json ...")
    rhymes = build_rhymes_json()
    (OUTPUT_FOLDER / "rhymes.json").write_text(json.dumps(rhymes, ensure_ascii=False, indent=1))

    print("Building ngrams.json ...")
    ngrams = build_ngrams_json()
    (OUTPUT_FOLDER / "ngrams.json").write_text(json.dumps(ngrams, ensure_ascii=False, indent=1))

    print("Building null_calibration.json and histogram plots ...")
    calibration = build_calibration_json_and_plots()
    (OUTPUT_FOLDER / "null_calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=1))

    print(f"Done. Wrote rhymes.json, ngrams.json, null_calibration.json, and both "
          f"*_similarity_distribution.png histograms to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    run()
