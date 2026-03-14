"""BM25 keyword search implementation."""

import logging
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 keyword-based retriever.

    This class implements a keyword-based search using the BM25 algorithm,
    which is effective for finding documents containing specific terms.
    """

    def __init__(self, documents: list[Document]) -> None:
        """Initialize the BM25 retriever with documents.

        Args:
            documents: List of documents to index.
        """
        self.documents = documents
        self._initialize_index()

    def _preprocess_text(self, text: str) -> list[str]:
        """Preprocess text for BM25 indexing.

        Args:
            text: Text to preprocess.

        Returns:
            List[str]: List of preprocessed tokens.
        """
        text = text.lower()

        tokens = re.findall(r"\w+", text)

        return [token for token in tokens if len(token) > 1]

    def _initialize_index(self) -> None:
        """Initialize the BM25 index."""
        if not self.documents:
            logger.warning("No documents provided for BM25 indexing. Creating empty index.")
            self.corpus = []
            self.bm25 = None
            return

        self.corpus = [self._preprocess_text(doc.page_content) for doc in self.documents]

        self.bm25 = BM25Okapi(self.corpus)
        logger.info("Initialized BM25 index with %d documents", len(self.documents))

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Search for documents matching the query.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples.
        """
        if self.bm25 is None or not self.documents:
            logger.warning("No documents in index, returning empty result")
            return []

        query_tokens = self._preprocess_text(query)

        if not query_tokens:
            logger.warning("No valid tokens in query, returning empty result")
            return []

        scores = self.bm25.get_scores(query_tokens)

        results = sorted(zip(self.documents, scores, strict=False), key=lambda x: x[1], reverse=True)[:k]

        logger.info("BM25 search for '%s' returned %d results", query, len(results))
        return results
