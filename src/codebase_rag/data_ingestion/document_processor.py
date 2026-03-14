"""Module for processing documents from repositories."""

import logging

from langchain_core.documents import Document

from .chunking import DocumentChunker
from .git_loader import GitLoader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents from repositories.

    This class orchestrates the document processing pipeline, including
    loading files from the Git repository, chunking them appropriately,
    and preparing them for indexing.
    """

    def __init__(
        self,
        git_loader: GitLoader | None = None,
        document_chunker: DocumentChunker | None = None,
    ) -> None:
        """Initialize the DocumentProcessor.

        Args:
            git_loader: Optional GitLoader instance.
            document_chunker: Optional DocumentChunker instance.
        """
        self.git_loader = git_loader or GitLoader()
        self.document_chunker = document_chunker or DocumentChunker()

    def process(
        self,
        included_dirs: list[str] | None = None,
        included_files: list[str] | None = None,
    ) -> list[Document]:
        """Process all relevant files from the repository.

        Args:
            included_dirs: List of directory paths to include.
            included_files: List of specific files to include.

        Returns:
            List[Document]: Processed and chunked documents ready for indexing.
        """
        self.git_loader.clone_or_pull()

        file_paths = self.git_loader.get_file_paths(included_dirs, included_files)

        all_documents = []
        for file_path in file_paths:
            logger.info("Processing %s", file_path)
            documents = self.document_chunker.process_file(file_path)
            all_documents.extend(documents)
            logger.debug("Added %d chunks from %s", len(documents), file_path)

        logger.info("Processed %d files into %d chunks", len(file_paths), len(all_documents))
        return all_documents
