"""
antipluto.classifier.stylometric
=============================
Stylometric feature extractor for email text.
Extracts word-level and character-level TF-IDF n-grams with casing and
punctuation preserved to capture human stylistic variance vs. LLM regularity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

logger = logging.getLogger(__name__)


class StylometricExtractor:
    """Extracts combined word and character n-gram TF-IDF representations."""

    def __init__(
        self,
        word_ngram_range: tuple[int, int] = (1, 2),
        word_max_features: int = 50000,
        char_ngram_range: tuple[int, int] = (3, 5),
        char_max_features: int = 50000,
        sublinear_tf: bool = True,
    ):
        self.word_ngram_range = word_ngram_range
        self.word_max_features = word_max_features
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.sublinear_tf = sublinear_tf

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram_range,
            max_features=self.word_max_features,
            lowercase=False,  # Casing is an important stylometric marker
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.char_ngram_range,
            max_features=self.char_max_features,
            lowercase=False,
            sublinear_tf=self.sublinear_tf,
        )

        self._is_fitted = False

    @property
    def total_features(self) -> int:
        if not self._is_fitted:
            return 0
        n_w = len(self.word_vectorizer.vocabulary_) if hasattr(self.word_vectorizer, "vocabulary_") else 0
        n_c = len(self.char_vectorizer.vocabulary_) if hasattr(self.char_vectorizer, "vocabulary_") else 0
        return n_w + n_c

    def fit(self, texts: List[str]) -> StylometricExtractor:
        """Fit word and character TF-IDF vectorizers on training email text."""
        logger.info("Fitting word TF-IDF vectorizer (max_features=%d)...", self.word_max_features)
        self.word_vectorizer.fit(texts)

        logger.info("Fitting character TF-IDF vectorizer (max_features=%d)...", self.char_max_features)
        self.char_vectorizer.fit(texts)

        self._is_fitted = True
        logger.info(
            "StylometricExtractor fitted: %d word features + %d char features = %d total features",
            len(self.word_vectorizer.vocabulary_),
            len(self.char_vectorizer.vocabulary_),
            self.total_features,
        )
        return self

    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        """Transform texts into a combined sparse TF-IDF matrix."""
        if not self._is_fitted:
            raise RuntimeError("StylometricExtractor must be fitted before calling transform().")

        w_feats = self.word_vectorizer.transform(texts)
        c_feats = self.char_vectorizer.transform(texts)
        combined = sparse.hstack([w_feats, c_feats], format="csr")
        return combined

    def fit_transform(self, texts: List[str]) -> sparse.csr_matrix:
        """Fit on texts and transform them into a combined sparse matrix."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: Union[str, Path]) -> None:
        """Serialize fitted extractor to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word_vectorizer": self.word_vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "is_fitted": self._is_fitted,
            "word_ngram_range": self.word_ngram_range,
            "word_max_features": self.word_max_features,
            "char_ngram_range": self.char_ngram_range,
            "char_max_features": self.char_max_features,
            "sublinear_tf": self.sublinear_tf,
        }
        joblib.dump(payload, path)
        logger.info("Saved StylometricExtractor to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> StylometricExtractor:
        """Load fitted extractor from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"StylometricExtractor checkpoint not found: {path}")

        payload = joblib.load(path)
        extractor = cls(
            word_ngram_range=payload["word_ngram_range"],
            word_max_features=payload["word_max_features"],
            char_ngram_range=payload["char_ngram_range"],
            char_max_features=payload["char_max_features"],
            sublinear_tf=payload["sublinear_tf"],
        )
        extractor.word_vectorizer = payload["word_vectorizer"]
        extractor.char_vectorizer = payload["char_vectorizer"]
        extractor._is_fitted = payload.get("is_fitted", True)
        logger.info("Loaded StylometricExtractor from %s (%d features)", path, extractor.total_features)
        return extractor
