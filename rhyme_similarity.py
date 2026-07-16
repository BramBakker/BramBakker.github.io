from __future__ import annotations

import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

INPUT_FOLDER = Path("csv_files")
TEXT_COLUMN = "text"
AUTHOR_COLUMN = "author"
UNKNOWN_AUTHOR_LABEL = "Onbekend"

QUESTIONED_MATCH = "brabantsche_yeesten__6"    # <-- the candidate/disputed manuscript
KNOWN_MATCHES = ["cronycke_van_brabant"]       # <-- optional: any number of reference texts, each called out in the results

MAX_FEATURES = 3000
MIN_DOC_FREQ = 2          # a rhyme word must appear in >=2 documents to count as a shared feature at all

TOP_N_DOCS = 10           # how many nearest-by-rhyme documents to list
TOP_N_DETAIL = 3          # how many of those top matches get a full rhyme breakdown
TOP_N_RHYMES = 15         # how many rhyme words to show per breakdown
CONTEXT_LINES = 2         # example lines to show per rhyme word per document

SAVE_PLOT = True
OUTPUT_FOLDER = INPUT_FOLDER.parent


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_corpus(folder: Path, text_col: str = "text", author_col: str = "author"):
    """Returns (lines_by_doc, author_by_doc)."""
    lines_by_doc, author_by_doc = {}, {}
    for csv_file in sorted(folder.glob("*.csv")):
        if csv_file.stem == "corpus_summary":
            continue
        df = pd.read_csv(csv_file)
        lines = df[text_col].dropna().astype(str).tolist()
        if not lines:
            continue
        lines_by_doc[csv_file.stem] = lines
        if author_col in df.columns and len(df[author_col].dropna()) > 0:
            author_by_doc[csv_file.stem] = str(df[author_col].dropna().iloc[0])
        else:
            author_by_doc[csv_file.stem] = UNKNOWN_AUTHOR_LABEL
    return lines_by_doc, author_by_doc


def find_doc(doc_names: list[str], match: str) -> int:
    hits = [i for i, n in enumerate(doc_names) if match.lower() in n.lower()]
    if not hits:
        raise ValueError(f"No document name contains '{match}'. Available: {doc_names}")
    if len(hits) > 1:
        raise ValueError(f"Multiple documents match '{match}': {[doc_names[i] for i in hits]}")
    return hits[0]


# --------------------------------------------------------------------------
# Rhyme extraction (keeps the source line so we can show context later)
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[{}]+$".format(re.escape(string.punctuation)))


def extract_rhyme_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """Returns (rhyme_word, full_line) for every line that ends in a word."""
    pairs = []
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        last = _PUNCT_RE.sub("", tokens[-1]).lower()
        if last:
            pairs.append((last, line.strip()))
    return pairs


def build_rhyme_matrix(lines_by_doc: dict[str, list[str]]):
    """
    Builds a document x rhyme-word matrix.
    Returns:
        Z              -- z-scored frequency matrix (each column standardized across the corpus)
        counts          -- raw rhyme-word counts per document (for readable context)
        doc_names       -- list of document names, index-aligned with Z/counts rows
        feature_names   -- list of rhyme words, index-aligned with Z/counts columns
        rhyme_pairs_by_doc -- dict: doc name -> list of (rhyme_word, full_line), for context lookup
    """
    doc_names = list(lines_by_doc.keys())
    rhyme_pairs_by_doc = {d: extract_rhyme_pairs(lines_by_doc[d]) for d in doc_names}
    pseudo_docs = [" ".join(w for w, _ in rhyme_pairs_by_doc[d]) for d in doc_names]

    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b",
                                  max_features=MAX_FEATURES, min_df=MIN_DOC_FREQ)
    counts = vectorizer.fit_transform(pseudo_docs).toarray().astype(float)
    feature_names = list(vectorizer.get_feature_names_out())

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    freqs = counts / row_sums

    col_mean = freqs.mean(axis=0, keepdims=True)
    col_std = freqs.std(axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    Z = (freqs - col_mean) / col_std

    return Z, counts, doc_names, feature_names, rhyme_pairs_by_doc


# --------------------------------------------------------------------------
# 1. Which documents are most similar to the candidate, by rhyme use
# --------------------------------------------------------------------------

def rank_similar_documents(questioned_idx: int, sim_matrix: np.ndarray, doc_names: list[str],
                            author_by_doc: dict[str, str], top_n: int = TOP_N_DOCS) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(doc_names):
        if i == questioned_idx:
            continue
        rows.append({
            "document": name,
            "author": author_by_doc[name],
            "rhyme_similarity": sim_matrix[questioned_idx, i],
        })
    df = pd.DataFrame(rows).sort_values("rhyme_similarity", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# --------------------------------------------------------------------------
# 2. Which specific rhymes are responsible for a given match, with context
# --------------------------------------------------------------------------

def responsible_rhymes(questioned_idx: int, comparison_idx: int, Z: np.ndarray, counts: np.ndarray,
                        feature_names: list[str], top_n: int = TOP_N_RHYMES) -> pd.DataFrame:
    """
    For a pair of documents, ranks rhyme words by how much they pull the two
    documents together: contribution = z_questioned * z_comparison.
    A large positive value means BOTH documents favor that rhyme word more
    than the corpus average, and by a similar amount -- exactly the kind of
    shared quirk that drives up cosine similarity. Only rhymes that BOTH
    documents actually use are kept -- two documents that both happen to
    avoid some rare rhyme word also produce a positive z_q*z_c product, but
    that's mutual absence, not a shared rhyme habit, so it's excluded here.
    """
    z_q, z_c = Z[questioned_idx], Z[comparison_idx]
    counts_q, counts_c = counts[questioned_idx], counts[comparison_idx]
    contribution = z_q * z_c
    eligible = (contribution > 0) & (counts_q > 0) & (counts_c > 0)
    positive_total = contribution[eligible].sum()

    order = np.argsort(-np.where(eligible, contribution, -np.inf))
    rows = []
    for idx in order:
        if len(rows) >= top_n or not eligible[idx]:
            break
        rows.append({
            "rhyme_word": feature_names[idx],
            "count_in_candidate": int(counts[questioned_idx, idx]),
            "count_in_comparison": int(counts[comparison_idx, idx]),
            "contribution": contribution[idx],
            "pct_of_shared_similarity": 100 * contribution[idx] / positive_total if positive_total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def rhyme_context(doc_name: str, rhyme_word: str, rhyme_pairs_by_doc: dict, n: int = CONTEXT_LINES) -> list[str]:
    lines = [line for w, line in rhyme_pairs_by_doc[doc_name] if w == rhyme_word]
    return lines[:n]


# --------------------------------------------------------------------------
# 3. Distribution of rhyme similarity across the corpus (the "null"/background)
# --------------------------------------------------------------------------

def similarity_background(sim_matrix: np.ndarray) -> np.ndarray:
    """All pairwise rhyme similarities in the corpus (excluding self-comparisons),
    used as the background against which any single match can be judged."""
    n = sim_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    return sim_matrix[iu]


def percentile_of(value: float, background: np.ndarray) -> float:
    return float((background < value).mean() * 100)


def plot_distribution(background: np.ndarray, markers: dict[str, float], path: Path) -> None:
    import itertools
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(background, bins=30, color="#888888", alpha=0.65, label="all pairwise rhyme similarities (corpus)")
    colors = itertools.cycle(["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"])
    for (label, val), color in zip(markers.items(), colors):
        ax.axvline(val, linestyle="--", linewidth=2, color=color, label=label)
    ax.set_xlabel("rhyme-word similarity (cosine)")
    ax.set_ylabel("number of document pairs")
    ax.set_title("How regular is rhyme similarity across this corpus?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run() -> dict:
    lines_by_doc, author_by_doc = load_corpus(INPUT_FOLDER, TEXT_COLUMN, AUTHOR_COLUMN)
    if len(lines_by_doc) < 3:
        raise RuntimeError(f"Only found {len(lines_by_doc)} csv files in {INPUT_FOLDER} -- check the path.")

    Z, counts, doc_names, feature_names, rhyme_pairs_by_doc = build_rhyme_matrix(lines_by_doc)
    sim_matrix = cosine_similarity(Z)

    questioned_idx = find_doc(doc_names, QUESTIONED_MATCH)
    known_idxs = []
    for match in KNOWN_MATCHES:
        try:
            known_idxs.append(find_doc(doc_names, match))
        except ValueError as e:
            print(f"(reference '{match}' not found, skipping: {e})")

    print(f"Candidate manuscript: {doc_names[questioned_idx]}  "
          f"(author: {author_by_doc[doc_names[questioned_idx]]})")
    print(f"Rhyme vocabulary used as features: {len(feature_names)} distinct rhyme words "
          f"(appearing in >= {MIN_DOC_FREQ} documents)")

    # 1. Ranked list of most similar documents ---------------------------------
    ranked = rank_similar_documents(questioned_idx, sim_matrix, doc_names, author_by_doc)
    print("\n--- Documents most similar to the candidate by rhyme use ---")
    print(ranked.to_string(index=False))
    ranked.to_csv(OUTPUT_FOLDER / "rhyme_similarity_ranking.csv", index=False)

    for known_idx in known_idxs:
        if known_idx == questioned_idx:
            continue
        known_sim = sim_matrix[questioned_idx, known_idx]
        if doc_names[known_idx] not in ranked["document"].values:
            print(f"\n(Reference text '{doc_names[known_idx]}' is not in the top {TOP_N_DOCS}: "
                  f"its rhyme similarity to the candidate is {known_sim:.3f})")

    # 2. Responsible rhymes + context, for the top matches (+ any reference docs) -
    detail_rows = []
    detail_targets = ranked.head(TOP_N_DETAIL).to_dict("records")
    covered = {r["document"] for r in detail_targets}
    for known_idx in known_idxs:
        name = doc_names[known_idx]
        if known_idx != questioned_idx and name not in covered:
            detail_targets.append({
                "document": name,
                "author": author_by_doc[name],
                "rhyme_similarity": sim_matrix[questioned_idx, known_idx],
            })
            covered.add(name)

    print(f"\n--- Rhymes driving similarity, for the top {TOP_N_DETAIL} matches "
          f"(plus any reference documents) ---")
    for row in detail_targets:
        comp_idx = doc_names.index(row["document"])
        resp = responsible_rhymes(questioned_idx, comp_idx, Z, counts, feature_names)
        print(f"\n  vs. {row['document']}  (rhyme_similarity={row['rhyme_similarity']:.3f})")
        if resp.empty:
            print("    (no shared rhymes pull these two documents together)")
            continue
        for _, r in resp.iterrows():
            ctx_q = rhyme_context(doc_names[questioned_idx], r["rhyme_word"], rhyme_pairs_by_doc)
            ctx_c = rhyme_context(row["document"], r["rhyme_word"], rhyme_pairs_by_doc)
            print(f"    '{r['rhyme_word']}'  "
                  f"(candidate x{r['count_in_candidate']}, comparison x{r['count_in_comparison']}, "
                  f"{r['pct_of_shared_similarity']:.1f}% of shared similarity)")
            for l in ctx_q:
                print(f"        [{doc_names[questioned_idx]}] ...{l}")
            for l in ctx_c:
                print(f"        [{row['document']}] ...{l}")
            detail_rows.append({
                "comparison_document": row["document"],
                "rhyme_similarity": row["rhyme_similarity"],
                "rhyme_word": r["rhyme_word"],
                "count_in_candidate": r["count_in_candidate"],
                "count_in_comparison": r["count_in_comparison"],
                "pct_of_shared_similarity": r["pct_of_shared_similarity"],
                "example_in_candidate": ctx_q[0] if ctx_q else "",
                "example_in_comparison": ctx_c[0] if ctx_c else "",
            })
    pd.DataFrame(detail_rows).to_csv(OUTPUT_FOLDER / "responsible_rhymes.csv", index=False)

    # 3. Distribution of rhyme similarity across the whole corpus ---------------
    background = similarity_background(sim_matrix)
    print("\n--- Distribution of rhyme similarity across the whole corpus ---")
    print(f"  n pairs={len(background)}  mean={background.mean():.3f}  std={background.std():.3f}  "
          f"min={background.min():.3f}  max={background.max():.3f}")
    print("  -> a small std means rhyme similarity is fairly uniform across the corpus (any one match "
          "standing out is meaningful); a large std means similarity varies a lot pair to pair on its own, "
          "so a single high score is less surprising.")

    top_match = ranked.iloc[0]
    pct = percentile_of(top_match["rhyme_similarity"], background)
    print(f"  candidate's best match ({top_match['document']}, sim={top_match['rhyme_similarity']:.3f}) "
          f"sits at the {pct:.1f}th percentile of that background distribution")

    pd.DataFrame({"pairwise_rhyme_similarity": background}).to_csv(
        OUTPUT_FOLDER / "rhyme_similarity_background.csv", index=False)

    if SAVE_PLOT:
        markers = {f"best match: {top_match['document']}": top_match["rhyme_similarity"]}
        for known_idx in known_idxs:
            if known_idx != questioned_idx:
                markers[f"reference: {doc_names[known_idx]}"] = sim_matrix[questioned_idx, known_idx]
        plot_path = OUTPUT_FOLDER / "rhyme_similarity_distribution.png"
        plot_distribution(background, markers, plot_path)
        print(f"  saved distribution plot -> {plot_path}")

    return {
        "ranked": ranked,
        "background": background,
        "top_match": top_match,
    }


if __name__ == "__main__":
    run()
