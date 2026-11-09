"""Multinomial Naive Bayes using only Python's standard library."""

from collections import Counter
from math import isfinite, log


class MultinomialNaiveBayes:
    """Fit and predict tokenized documents (lists of word strings)."""

    def __init__(self, alpha=1.0):
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and greater than zero")
        self.alpha = alpha
        self.classes = []

    def fit(self, documents, labels):
        if len(documents) != len(labels) or not labels:
            raise ValueError("Provide equally sized, nonempty documents and labels")
        self.class_document_counts = Counter(labels)
        self.classes = sorted(self.class_document_counts)
        self.word_counts = {label: Counter() for label in self.classes}
        self.vocabulary = set()

        # Count every word occurrence, grouped by its document's class.
        for document, label in zip(documents, labels):
            self.word_counts[label].update(document)
            self.vocabulary.update(document)
        self.total_word_counts = {
            label: sum(counts.values()) for label, counts in self.word_counts.items()
        }
        # Priors use document counts, not word counts: P(c) = N_c / N.
        self.class_priors = {
            label: count / len(labels)
            for label, count in self.class_document_counts.items()
        }
        self.log_priors = {c: log(p) for c, p in self.class_priors.items()}
        # Smoothing adds alpha once per training-vocabulary word.
        self.denominators = {
            c: self.total_word_counts[c] + self.alpha * len(self.vocabulary)
            for c in self.classes
        }
        return self

    def predict(self, document):
        if not self.classes:
            raise ValueError("Call fit before predict")
        counts = Counter(word for word in document if word in self.vocabulary)
        scores = {}
        for label in self.classes:
            score = self.log_priors[label]
            for word, count in counts.items():
                likelihood = (
                    self.word_counts[label][word] + self.alpha
                ) / self.denominators[label]
                # Repeated words contribute count * log P(word | class).
                score += count * log(likelihood)
            scores[label] = score
        # Unknown-only or empty documents use priors; ties use sorted class order.
        return max(self.classes, key=scores.get)

    def predict_many(self, documents):
        return [self.predict(document) for document in documents]
