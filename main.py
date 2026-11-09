"""Run six reproducible experiments on a local 20 Newsgroups dataset."""

import argparse
from pathlib import Path
import re
import tarfile
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from naive_bayes import MultinomialNaiveBayes


def load_dataset(path):
    """Read local files or archive members, removing headers before tokenization."""
    documents, labels = [], []
    if path.is_dir():
        for category in sorted(path.iterdir()):
            if category.is_dir():
                for file in sorted(category.iterdir()):
                    if file.is_file():
                        documents.append(file.read_bytes().decode("latin-1"))
                        labels.append(category.name)
    else:
        records = []
        # Stream compressed members in archive order to avoid repeated gzip seeks.
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                parts = Path(member.name).parts
                if member.isfile() and len(parts) >= 2:
                    with archive.extractfile(member) as stream:
                        records.append((member.name, stream.read().decode("latin-1"), parts[-2]))
        for _, text, label in sorted(records):
            documents.append(text)
            labels.append(label)
    if not documents:
        raise ValueError(f"No documents found in {path}")
    # The first blank line separates the complete header block from the body.
    # Apply once for both input formats and both tokenization modes.
    documents = [re.split(r"\r?\n\r?\n", text, maxsplit=1)[-1] for text in documents]
    return documents, labels


def tokenize(text, mode):
    if mode == "raw":
        return text.split()
    # Replace punctuation/numbers with spaces to avoid joining adjacent words.
    text = "".join(
        " " if unicodedata.category(char)[0] in "PN" else char
        for char in text.lower()
    )
    return [word for word in text.split() if word not in ENGLISH_STOP_WORDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "20_newsgroups")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    documents, labels = load_dataset(args.data)
    categories = sorted(set(labels))
    print(f"Loaded {len(documents)} documents in {len(categories)} categories.", flush=True)
    print("Email headers removed before raw and cleaned tokenization.", flush=True)

    # Choose the test indices once, independently of preprocessing and training size.
    pool, test = train_test_split(
        list(range(len(documents))), test_size=0.2, stratify=labels, random_state=42
    )
    rows = []
    for mode in ("raw", "cleaned"):
        # Fixed, data-independent rules and a static stopword list need no fitting.
        tokens = [tokenize(document, mode) for document in documents]
        for train_percent, validation_percent in ((50, 30), (60, 20), (70, 10)):
            train, validation = train_test_split(
                pool, train_size=train_percent / 80,
                stratify=[labels[i] for i in pool], random_state=42
            )
            assert not (set(train) & set(validation) or set(train) & set(test)
                        or set(validation) & set(test))
            assert len(train) + len(validation) + len(test) == len(documents)
            model = MultinomialNaiveBayes(args.alpha)
            # Only training documents contribute vocabulary, counts and priors.
            model.fit([tokens[i] for i in train], [labels[i] for i in train])
            validation_predictions = model.predict_many(tokens[i] for i in validation)
            test_predictions = model.predict_many(tokens[i] for i in test)
            validation_accuracy = accuracy_score(
                [labels[i] for i in validation], validation_predictions
            )
            truth = [labels[i] for i in test]
            test_accuracy = accuracy_score(truth, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, test_predictions, labels=categories, average="macro", zero_division=0
            )
            split = f"{train_percent}/{validation_percent}/20"
            rows.append((mode, split, validation_accuracy, test_accuracy, precision, recall, f1))
            print(f"Finished {mode:7} {split}: "
                  f"n={len(train)}/{len(validation)}/{len(test)}, "
                  f"vocabulary={len(model.vocabulary)}", flush=True)

    print("\nAll metrics in percent; precision/recall/F1 are test-set macro averages.")
    print(f"{'Mode':8} {'Split':10} {'Val acc':>8} {'Test acc':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    for mode, split, *metrics in rows:
        print(f"{mode:8} {split:10}" + "".join(f"{100 * metric:10.2f}" for metric in metrics))


if __name__ == "__main__":
    main()
