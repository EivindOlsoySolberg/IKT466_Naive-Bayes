# Multinomial Naive Bayes - IKT466

Small, from-scratch text classifier for all 20 Newsgroups categories.

## Run

Requires Python 3.10+ and scikit-learn (`python -m pip install scikit-learn`).

```powershell
python main.py
python main.py --data "C:\Users\47984\Downloads\20_newsgroups.tar.gz"
python main.py --alpha 0.5
```

No dataset is downloaded. The default input is the existing `20_newsgroups/`
directory beside the script; `--data` also accepts the local `.tar.gz` archive.
Archive members are read without extracting files. Latin-1 decoding preserves
every body byte. File and category ordering is deterministic. For both input
formats, the loader removes the email header block through the first blank line
(LF or CRLF) before either raw or cleaned tokenization. Body paragraphs are
preserved; a document without a header/body separator is left unchanged.

## Project structure

```text
main.py             Data loading, tokenization, splits and evaluation
naive_bayes.py      Standard-library MultinomialNaiveBayes classifier
README.md           Usage and short experiment report
results.txt         Output from the complete six-experiment run
20_newsgroups/      Existing local dataset, 20 category directories
```

## Method

The classifier accepts lists of tokens in `fit(documents, labels)`,
`predict(document)` and `predict_many(documents)`. It counts documents per class
and word occurrences per class with `Counter`. Training alone determines the
vocabulary. For class c, the prior is its document count divided by the total
training document count. Word likelihoods use
`(count(word, c) + alpha) / (total_words(c) + alpha * vocabulary_size)`.
Alpha defaults to 1.0 and must be positive and finite.

Prediction maximizes `log(prior) + sum(count(word) * log(likelihood))`.
Unknown words are ignored. Empty or unknown-only documents use the priors;
ties are resolved by alphabetical class order. No existing classifier,
vectorizer, TF-IDF, stemming or lemmatization is used.

Raw tokenization splits on whitespace, retaining case, punctuation and numbers.
Cleaned tokenization lowercases, replaces Unicode punctuation and numbers with
spaces, splits on whitespace and removes scikit-learn's fixed English stopword
list. This import provides only a static list; it does not implement word counts
or learn from the corpus. Replacing removed characters with spaces avoids
accidentally concatenating words.

Stratified splitting uses random_state=42. A single 20% test set is held fixed.
The remaining 80% is split into 50/30, 60/20 and 70/10 percentage points of the
complete corpus. Fractional document counts are rounded by the splitting utility.
Each model is fit only on its training partition. Applying fixed tokenization
rules to validation/test text does not learn any statistics from those sets.
Precision, recall and F1 are unweighted macro averages across all 20 test classes.
Test scores compare the prescribed configurations; they are not used to tune alpha
or select preprocessing. No final retraining on validation/test documents occurs.

## Interpretation and limitations

Complete header-free run on the supplied archive: 19,997 documents, 20 categories, alpha=1.
All values below are percentages. Precision/recall/F1 are test macro averages.

| Mode | Train/validation/test | Validation accuracy | Test accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 50/30/20 | 64.99 | 67.17 | 75.11 | 67.17 | 67.51 |
| Raw | 60/20/20 | 66.62 | 69.25 | 76.78 | 69.25 | 69.94 |
| Raw | 70/10/20 | 68.50 | 71.20 | 77.40 | 71.20 | 71.72 |
| Cleaned | 50/30/20 | 78.58 | 79.35 | 80.18 | 79.35 | 78.39 |
| Cleaned | 60/20/20 | 78.77 | 79.88 | 80.92 | 79.88 | 78.99 |
| Cleaned | 70/10/20 | 79.20 | 79.72 | 80.97 | 79.72 | 78.72 |

Cleaning improved test accuracy by approximately 12.18, 10.63 and 8.52 percentage points,
respectively (using the displayed rounded scores). Raw test accuracy increased by 4.03 points between the smallest
and largest training sets. Cleaned accuracy varied by only 0.53 points and did
not improve monotonically: 60/20/20 had the highest test accuracy, while 70/10/20
had the highest validation accuracy. The 60/20/20 macro F1 was 78.99%. More training data therefore helped the raw
representation here but did not guarantee better cleaned-text results.

At 50/30/20 the vocabulary shrank from 244,144 raw tokens to 102,632 cleaned
tokens, consistent with less fragmentation into case/punctuation variants.
This observation supports a sparsity explanation, but these experiments change
several preprocessing operations together and cannot identify each one's effect.

Compared with the previous header-retaining run, test accuracy fell by approximately
14.48/13.40/12.53 points for raw text and 8.57/8.50/8.20 points for cleaned text
(50/30/20, 60/20/20, 70/10/20; differences of displayed rounded scores).
The test indices are unchanged because document ordering, labels and split rules
are unchanged. This decrease is consistent with the earlier scores benefiting from
header information, including fields that can expose the target category.

Original email headers, including `Newsgroups:`, are removed consistently before
both experiments. Signatures and quoted replies inside the body remain, and can
still provide shortcuts. Index partitions are disjoint, but duplicate or related
posts across partitions are another possible source of optimistic scores.
No deduplication or additional body cleaning was added.

Cleaning can merge case/punctuation variants, reduce sparsity and remove common
words that carry little topic information. It can also discard informative numbers,
capitalization or symbols. More training documents generally improve count estimates,
but the realized change is dataset dependent. Validation sizes and contents differ
across configurations, so their accuracies are less directly comparable than scores
on the fixed test set. A single random seed cannot establish statistical significance.

## Verification

Hand-calculated checks covered document priors, word totals, the smoothing
denominator, likelihood normalization, repeated-word scoring, unknown words,
empty documents, refitting and both tokenizers. Archive and directory loading
produced exactly identical document contents, labels and ordering. Runtime assertions
check that train, validation and test indices are disjoint and cover the corpus.
The assignment was reviewed again after implementation.

After the header-removal update, checks covered LF/CRLF separators, folded headers,
preserved body paragraphs and headerless input. All 19,997 supplied documents had
a header/body separator. Body-only archive and directory loads matched exactly.
For each of the six configurations, a separate check verified that the learned
vocabulary equaled the union of training tokens, excluded all held-out-only tokens,
and that the three index partitions were disjoint and covered the entire corpus.
SHA256 comparisons confirmed that the classifier and tokenization function were
unchanged.
