from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

INPUT_FOLDER = Path("rijm_csv")
TEXT_COLUMN = "text"

# The document you want neighbors for. This can be your known/reference text
# (as below, matching your original snippet) or your questioned/candidate text --
# whichever you want to ask "who talks like this?" about.
ANCHOR_MATCH = "cronycke_van_brabant"

NGRAM_RANGE = (2, 3)      # bigrams AND trigrams together; use (2, 2) for bigrams only
MAX_FEATURES = None       # cap the vocabulary size if the corpus is large; None = no cap
MIN_DOC_FREQ = 2          # an n-gram must appear in >=2 documents to count as "shared" at all

TOP_N_DOCS = 10           # how many nearest-by-phrase documents to list
TOP_N_DETAIL = 3          # how many of those top matches get a full n-gram breakdown
TOP_N_NGRAMS = 20         # how many n-grams to show per breakdown
CONTEXT_SNIPPETS = 2      # example occurrences to show per n-gram per document
CONTEXT_WINDOW = 50       # characters of surrounding text to show on each side

# Specific document pairs to mark on the distribution plot, each as its own
# labeled line -- e.g. to show where two different candidates against the
# same reference text fall relative to the corpus background.
HIGHLIGHT_PAIRS = [
    ("cronycke_van_brabant", "brabantsche_yeesten__6"),
    ("cronycke_van_brabant", "brabantsche_yeesten__7"),
]

SAVE_PLOT = True
OUTPUT_FOLDER = INPUT_FOLDER.parent


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_corpus(folder: Path, text_col: str = "text") -> dict[str, str]:
    """Returns {doc_name: full joined text}. Unlike the rhyme script, here each
    document is one long string, since n-grams can span anywhere in the text,
    not just line-ends."""
    doc_text = {}
    for csv_file in sorted(folder.glob("*.csv")):
        if csv_file.stem == "corpus_summary":
            continue
        df = pd.read_csv(csv_file)
        lines = df[text_col].dropna().astype(str).tolist()
        if not lines:
            continue
        doc_text[csv_file.stem] = " ".join(lines)
    return doc_text


def find_doc(doc_names: list[str], match: str) -> int:
    hits = [i for i, n in enumerate(doc_names) if match.lower() in n.lower()]
    if not hits:
        raise ValueError(f"No document name contains '{match}'. Available: {doc_names}")
    if len(hits) > 1:
        raise ValueError(f"Multiple documents match '{match}': {[doc_names[i] for i in hits]}")
    return hits[0]


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def build_ngram_matrix(doc_text: dict[str, str]):
    doc_names = list(doc_text.keys())
    documents = [doc_text[d] for d in doc_names]

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=NGRAM_RANGE,
                                  token_pattern=r"(?u)\b\w+\b",
                                  max_features=MAX_FEATURES, min_df=MIN_DOC_FREQ)
    X = vectorizer.fit_transform(documents)
    feature_names = list(vectorizer.get_feature_names_out())
    return X, doc_names, feature_names


# --------------------------------------------------------------------------
# 1. Which documents are most similar to the anchor, by n-gram use
# --------------------------------------------------------------------------

def rank_similar_documents(anchor_idx: int, sim_matrix: np.ndarray, doc_names: list[str],
                            top_n: int = TOP_N_DOCS) -> pd.DataFrame:
    rows = [{"document": name, "similarity": sim_matrix[anchor_idx, i]}
            for i, name in enumerate(doc_names) if i != anchor_idx]
    df = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# --------------------------------------------------------------------------
# 2. Which specific n-grams are responsible for a given match, with context
# --------------------------------------------------------------------------

def responsible_ngrams(anchor_idx: int, comparison_idx: int, X, feature_names: list[str],
                        top_n: int = TOP_N_NGRAMS) -> pd.DataFrame:
    """
    Elementwise product of the two TF-IDF rows: an n-gram scores high here only
    if BOTH documents give it a high TF-IDF weight -- i.e. both use that exact
    phrase, and it's distinctive enough (rare enough across the corpus) that
    TF-IDF didn't discount it. This is exactly what cosine similarity sums up,
    so ranking by this product tells you which phrases are actually driving
    the similarity score.
    """
    contrib = X[anchor_idx].multiply(X[comparison_idx]).tocoo()
    if contrib.nnz == 0:
        return pd.DataFrame(columns=["ngram", "tfidf_anchor", "tfidf_comparison",
                                      "contribution", "pct_of_shared_similarity"])

    total = contrib.data.sum()
    order = contrib.data.argsort()[::-1][:top_n]
    anchor_row = np.asarray(X[anchor_idx].todense()).ravel()
    comp_row = np.asarray(X[comparison_idx].todense()).ravel()

    rows = []
    for j in order:
        col = contrib.col[j]
        rows.append({
            "ngram": feature_names[col],
            "tfidf_anchor": anchor_row[col],
            "tfidf_comparison": comp_row[col],
            "contribution": contrib.data[j],
            "pct_of_shared_similarity": 100 * contrib.data[j] / total if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def ngram_context(doc_text: str, phrase: str, n: int = CONTEXT_SNIPPETS,
                   window: int = CONTEXT_WINDOW) -> list[str]:
    """Finds up to n occurrences of `phrase` in doc_text and returns short
    surrounding snippets, so you can see the phrase used in situ."""
    lower_text = doc_text.lower()
    snippets, start = [], 0
    while len(snippets) < n:
        idx = lower_text.find(phrase, start)
        if idx == -1:
            break
        s = max(0, idx - window)
        e = min(len(doc_text), idx + len(phrase) + window)
        snippets.append("..." + doc_text[s:e].strip() + "...")
        start = idx + len(phrase)
    return snippets


# --------------------------------------------------------------------------
# 3. Distribution of similarity across the corpus (the background / "is this random?")
# --------------------------------------------------------------------------

def similarity_background(sim_matrix: np.ndarray) -> np.ndarray:
    """All pairwise similarities in the corpus (excluding self-comparisons) --
    the background against which any single match can be judged."""
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
    ax.hist(background, bins=30, color="#888888", alpha=0.65, label="all pairwise similarities (corpus)")
    colors = itertools.cycle(["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"])
    for (label, val), color in zip(markers.items(), colors):
        ax.axvline(val, linestyle="--", linewidth=2, color=color, label=label)
    ax.set_xlabel("similarity (cosine, TF-IDF n-grams)")
    ax.set_ylabel("number of document pairs")
    ax.set_title(f"How regular is {NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}-gram similarity across this corpus?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run() -> dict:
    doc_text = load_corpus(INPUT_FOLDER, TEXT_COLUMN)
    if len(doc_text) < 3:
        raise RuntimeError(f"Only found {len(doc_text)} csv files in {INPUT_FOLDER} -- check the path.")

    X, doc_names, feature_names = build_ngram_matrix(doc_text)
    sim_matrix = cosine_similarity(X)

    with open(OUTPUT_FOLDER / "tfidf_ngrams.pickle", "wb") as handle:
        pickle.dump({"X": X, "doc_names": doc_names, "feature_names": feature_names,
                     "ngram_range": NGRAM_RANGE}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    anchor_idx = find_doc(doc_names, ANCHOR_MATCH)
    print(f"Anchor document: {doc_names[anchor_idx]}")
    print(f"{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}-gram vocabulary used as features: {len(feature_names)} "
          f"distinct phrases (appearing in >= {MIN_DOC_FREQ} documents)")

    # 1. Ranked list of most similar documents ---------------------------------
    ranked = rank_similar_documents(anchor_idx, sim_matrix, doc_names)
    print("\n--- Documents most similar to the anchor by n-gram use ---")
    print(ranked.to_string(index=False))
    ranked.to_csv(OUTPUT_FOLDER / "ngram_similarity_ranking.csv", index=False)

    # 2. Responsible n-grams + context, for the top matches ---------------------
    detail_rows = []
    print(f"\n--- Phrases driving similarity, for the top {TOP_N_DETAIL} matches ---")
    for _, row in ranked.head(TOP_N_DETAIL).iterrows():
        comp_idx = doc_names.index(row["document"])
        resp = responsible_ngrams(anchor_idx, comp_idx, X, feature_names)
        print(f"\n  vs. {row['document']}  (similarity={row['similarity']:.3f})")
        if resp.empty:
            print("    (no shared n-grams pull these two documents together)")
            continue
        for _, r in resp.iterrows():
            ctx_a = ngram_context(doc_text[doc_names[anchor_idx]], r["ngram"])
            ctx_c = ngram_context(doc_text[row["document"]], r["ngram"])
            print(f"    '{r['ngram']}'  (tfidf {r['tfidf_anchor']:.3f} / {r['tfidf_comparison']:.3f}, "
                  f"{r['pct_of_shared_similarity']:.1f}% of shared similarity)")
            for s in ctx_a:
                print(f"        [{doc_names[anchor_idx]}] {s}")
            for s in ctx_c:
                print(f"        [{row['document']}] {s}")
            detail_rows.append({
                "comparison_document": row["document"],
                "similarity": row["similarity"],
                "ngram": r["ngram"],
                "tfidf_anchor": r["tfidf_anchor"],
                "tfidf_comparison": r["tfidf_comparison"],
                "pct_of_shared_similarity": r["pct_of_shared_similarity"],
                "example_in_anchor": ctx_a[0] if ctx_a else "",
                "example_in_comparison": ctx_c[0] if ctx_c else "",
            })
    pd.DataFrame(detail_rows).to_csv(OUTPUT_FOLDER / "ngram_responsible_phrases.csv", index=False)

    # 3. Distribution of similarity across the whole corpus ---------------------
    background = similarity_background(sim_matrix)
    print("\n--- Distribution of n-gram similarity across the whole corpus ---")
    print(f"  n pairs={len(background)}  mean={background.mean():.3f}  std={background.std():.3f}  "
          f"min={background.min():.3f}  max={background.max():.3f}")
    print("  -> a small std means n-gram similarity is fairly uniform across the corpus (any one match "
          "standing out is meaningful); a large std means similarity naturally varies a lot pair to pair, "
          "so a single high score is less surprising on its own.")

    top_match = ranked.iloc[0]
    pct = percentile_of(top_match["similarity"], background)
    print(f"  anchor's best match ({top_match['document']}, sim={top_match['similarity']:.3f}) "
          f"sits at the {pct:.1f}th percentile of that background distribution")

    pd.DataFrame({"pairwise_ngram_similarity": background}).to_csv(
        OUTPUT_FOLDER / "ngram_similarity_background.csv", index=False)

    if SAVE_PLOT:
        markers = {f"best match: {top_match['document']}": top_match["similarity"]}
        print("\n--- Highlighted document pairs ---")
        for a_match, b_match in HIGHLIGHT_PAIRS:
            try:
                a_idx = find_doc(doc_names, a_match)
                b_idx = find_doc(doc_names, b_match)
            except ValueError as e:
                print(f"  skipping highlight pair {a_match!r} vs {b_match!r}: {e}")
                continue
            sim = sim_matrix[a_idx, b_idx]
            pair_pct = percentile_of(sim, background)
            label = f"{doc_names[a_idx]} vs {doc_names[b_idx]}"
            print(f"  {label}: similarity={sim:.3f}  ({pair_pct:.1f}th percentile of background)")
            markers[f"{label}  (sim={sim:.3f})"] = sim
        plot_path = OUTPUT_FOLDER / "ngram_similarity_distribution.png"
        plot_distribution(background, markers, plot_path)
        print(f"  saved distribution plot -> {plot_path}")

    return {"ranked": ranked, "background": background, "top_match": top_match}


if __name__ == "__main__":
    run()
