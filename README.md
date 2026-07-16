# Hennen van Merchtenen and the voortzetting

A static site comparing Cronycke van Brabant (Hennen van Merchtenen's known
chronicle) against De Voortzetting -- the disputed continuation of the
Brabantsche Yeesten -- across three independent methods: shared rhyme words,
shared bigrams/trigrams, and contrastive style embeddings. All three are now
real, computed results, not placeholder data.

## Files

```
index.html                        the page
css/style.css                     styling
script.js                         renders everything from /data -- no build step
assets/manuscript-bg.png          hero background image

data/
  ngrams.json                       real: TF-IDF bigram/trigram comparison (Cronycke vs. De Voortzetting)
  rhymes.json                       real: rhyme-word comparison (Cronycke vs. De Voortzetting)
  null_calibration.json             real: background similarity stats + percentiles, for both methods
  ngram_similarity_distribution.png real: histogram of n-gram similarity across the corpus
  rhyme_similarity_distribution.png real: histogram of rhyme similarity across the corpus
  embeddings.json                   real: 2D projections of ByT5 contrastive line embeddings

csv_files/                         the corpus -- 263 Middle Dutch rhymed texts, one CSV per document
                                    (author, chapter, line, text columns; author is optional)

rhyme_similarity.py                rhyme-word feature extraction + cosine similarity (standalone/console)
ngram_similarity.py                TF-IDF bigram/trigram feature extraction + cosine similarity (standalone/console)
export_for_website.py              imports the two scripts above, writes data/{ngrams,rhymes,null_calibration}.json
                                    + both *_similarity_distribution.png histograms

contrastive_line_embeddings.py     trains the ByT5 contrastive encoder on the whole corpus -> byt5_stylometry_encoder.pt
create_sembeddings.py              loads that trained encoder, embeds Cronycke + De Voortzetting lines,
                                    writes stylometry_data.json in the data/embeddings.json shape
```

`data.csv` sitting at the repo root is unrelated leftover dummy data (`item,measure,value`)
-- not read by anything here, safe to delete.

## Hosting on GitHub Pages

1. Keep `data/`, `css/`, and `assets/` at the same level as `index.html` --
   `script.js` fetches `data/*.json` with relative paths, and `index.html`
   loads `css/style.css`.
2. Settings -> Pages -> deploy from the branch containing these files.
3. That's it -- no build step, it's a static site.

If you host `data/*.json` in a *different* repo instead, change the four
`fetchJSON("data/...")` calls at the bottom of `script.js` to full
`raw.githubusercontent.com` URLs.

## Regenerating the data

Three separate pipelines feed the three tabs. You only need to re-run a given
one if its inputs changed (corpus edited, new questioned document, retrained
model, etc.) -- they don't depend on each other.

### N-grams + rhymes (the two evidence tabs, and their calibration histograms)

1. Corpus CSVs live in `csv_files/` already. If you add or rename documents,
   update the config block at the top of `export_for_website.py`:
   ```python
   KNOWN_MATCH = "cronycke_van_brabant"
   KNOWN_LABEL = "Cronycke van Brabant"
   QUESTIONED_MATCHES = [
       ("voortzetting", "De Voortzetting"),
   ]
   ```
   `QUESTIONED_MATCHES` is a list, so you can add more than one questioned
   document -- each gets its own stacked comparison block on the site.
2. Run:
   ```
   python3 export_for_website.py
   ```
   This writes `data/ngrams.json`, `data/rhymes.json`,
   `data/null_calibration.json`, and both `data/*_similarity_distribution.png`
   histograms. It leaves `data/embeddings.json` alone.

`rhyme_similarity.py` and `ngram_similarity.py` can also be run directly for
console/CSV output if you just want to explore the corpus without touching
the website -- `export_for_website.py` imports their functions rather than
duplicating the logic, so behavior stays in sync between the two.

**`EXCLUDE_WORDS`** at the top of `ngram_similarity.py` (currently `brabant`,
`hertoghe`, `greve`) drops any n-gram containing one of these words before
similarity is computed -- these are generic/thematic words expected to be
common to any chronicle about Brabant, not a stylistic fingerprint.
`export_for_website.py` passes this same list through, so the website and
the standalone script always agree. If you edit the list, re-run
`export_for_website.py` for it to take effect on the site.

### Contrastive embeddings tab

1. Train the encoder:
   ```
   python3 contrastive_line_embeddings.py
   ```
   This reads the corpus (its `INPUT_FOLDER` to `csv_files` to match this repo before running) and saves
   `byt5_stylometry_encoder.pt`. Needs a GPU for any reasonable training
   time, and network access to download `google/byt5-small` from Hugging
   Face the first time it runs. The trained weights file isn't checked into
   this repo (large binary) -- keep it locally or in your own storage.
2. Generate the embedding points:
   ```
   python3 create_sembeddings.py
   ```
   Loads `byt5_stylometry_encoder.pt`, embeds sampled lines from
   `cronycke_van_brabant` and `voortzetting`, reduces to 2D with PCA, and
   writes `stylometry_data.json`. Note its `note`/`known_label`/
   `questioned_label` are hardcoded near the bottom of the script (currently
   "Cronycke van Brabant" / "Brabantsche Yeesten") rather than read from a
   config block -- edit those lines directly if you want different wording
   (the live `data/embeddings.json` already has these hand-edited to
   "Cornicke" / "De voortzetting").
3. Rename/move `stylometry_data.json` to `data/embeddings.json`.

## Method notes

- **Background/calibration**: for each of the n-gram and rhyme methods,
  "background" is the known text's similarity to every *other single
  document* in the corpus (not all pairwise combinations) -- so the
  histogram shows how similar Cronycke tends to look to an unrelated
  document, as context for how similar it looks to De Voortzetting
  specifically. The percentile in each tab's caption is computed against
  that same distribution.
- **Calibration charts are static images**, not interactive -- `data/
  {ngram,rhyme}_similarity_distribution.png`, generated by the same
  `plot_distribution()` function used in the standalone scripts, so what you
  see on the site is exactly what running those scripts yourself produces.
  Only the actual Cronycke-vs-De-Voortzetting comparison is marked on them;
  there's no automatic "best match against the whole corpus" marker.
- **Context windows** (n-grams/rhymes tabs): each shared term's example
  occurrences show the matched line plus 2 lines of surrounding context on
  each side, up to 3 occurrences per document per term.
- **Known label inconsistency**: `data/ngrams.json`, `data/rhymes.json`, and
  `data/null_calibration.json` currently label the known text "Cronycke van
  Brabant", while `data/embeddings.json` calls it "Cornicke" -- the more
  historically accurate spelling per Reynaert (2019). If you want these to
  match, either update `KNOWN_LABEL` in `export_for_website.py` and re-run
  it, or edit the four JSON files directly (no code changes needed either
  way -- the label is pure data, never hardcoded in `script.js`).

## Not included

- `byt5_stylometry_encoder.pt` -- the trained encoder weights. Large binary,
  regenerate via `contrastive_line_embeddings.py` or keep it in your own
  storage rather than the repo.
