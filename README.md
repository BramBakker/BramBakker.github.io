# Hennen van Merchtenen and the voortzetting

A static site comparing Cronycke van Brabant against two disputed books of the
Brabantsche Yeesten continuation, by shared rhyme words and shared bigrams/trigrams.

## Files

```
index.html                 the page
style.css                  styling
script.js                  renders everything from /data -- no build step
data/
  ngrams.json               real: TF-IDF bigram/trigram comparisons
  rhymes.json               real: rhyme-word comparisons
  null_calibration.json     real: background similarity distributions
  embeddings.json           placeholder -- no embedding model exists yet

rhyme_similarity.py         rhyme-word feature extraction + cosine similarity
ngram_similarity.py         TF-IDF bigram/trigram feature extraction + cosine similarity
export_for_website.py       imports the two scripts above, writes the three real JSON files
```

## Hosting on GitHub Pages

1. Put all of the above in a repo (keep `data/` at the same level as `index.html`
   -- `script.js` fetches `data/*.json` with a relative path).
2. Settings -> Pages -> deploy from the branch containing these files.
3. That's it -- no build step, it's a static site.

If you host `data/*.json` in a *different* repo instead, change the four
`fetchJSON("data/...")` calls at the bottom of `script.js` to full
`raw.githubusercontent.com` URLs.

## Regenerating the data

You only need to do this again if the corpus changes or you add more
questioned documents.

1. Put your corpus CSVs (`author,chapter,line,text` columns; `author` is
   optional) in a `csv_files/` folder next to these scripts.
2. Edit the config block at the top of `export_for_website.py` if your
   filenames or labels differ:
   ```python
   KNOWN_MATCH = "cronycke_van_brabant"
   KNOWN_LABEL = "Cronycke van Brabant"
   QUESTIONED_MATCHES = [
       ("brabantsche_yeesten__6", "Brabantsche Yeesten, boek 6"),
       ("brabantsche_yeesten__7", "Brabantsche Yeesten, boek 7"),
   ]
   ```
3. Run:
   ```
   python3 export_for_website.py
   ```
   This writes `data/ngrams.json`, `data/rhymes.json`, and
   `data/null_calibration.json`. Drop those three into the site's `data/`
   folder (it leaves `embeddings.json` alone).

`rhyme_similarity.py` and `ngram_similarity.py` can also still be run
directly for console/CSV output if you just want to explore the corpus
without touching the website.

## Method notes

- **Background/calibration**: for each method, "background" is the known
  text's similarity to every *other single document* in the corpus (not all
  pairwise combinations) -- so each dot in the calibration chart is one real
  document, hoverable by name. The questioned books' percentile is computed
  against that same distribution.
- **Context windows**: each shared term's example occurrences show the
  matched line plus 2 lines of surrounding context on each side, up to 3
  occurrences per document per term.
- **Contrastive embeddings tab**: intentionally left as placeholder/sample
  data. No model has been trained; the tab exists to preview the intended
  layout. Replace `data/embeddings.json` (shape documented at the top of
  `script.js`) once you have one, and it'll stop being flagged as sample data.

## Not included

- `assets/manuscript-bg.png` (the hero background image) -- add your own; its
  absence doesn't break anything, the hero just shows a plain background.
