import semchunk
from transformers import AutoTokenizer
from typing import List, Callable
from src.preprocess.preprocessor import TextPreprocessor

class SemanticChunker(TextPreprocessor):
    """
    A preprocessor that chunks text semantically using the semchunk library.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", chunk_size: int = 128):
        """
        Initializes the SemanticChunker.

        Args:
            model_name: The Hugging Face model name or path for tokenization.
            chunk_size: The target chunk size in tokens.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self._chunker: Callable[[str], List[str]] = self._create_chunker()

    def _create_chunker(self) -> Callable[[str], List[str]]:
        """Creates the internal semchunk chunker function."""
        try:
            chunker = semchunk.chunkerify(self.model_name, self.chunk_size)
            if chunker:
                return chunker

            # Fallback if direct name resolution fails
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            chunker = semchunk.chunkerify(tokenizer, self.chunk_size)
            if chunker:
                return chunker

            raise ValueError("Failed to create chunker with both model name and tokenizer.")

        except Exception as e:
            print(f"Error creating semchunk chunker for model '{self.model_name}': {e}")
            # Re-raise as a RuntimeError to indicate initialization failure
            raise RuntimeError(f"Could not initialize SemanticChunker for model: {self.model_name}") from e

    def pre_process(self, text: str) -> List[str]:
        """
        Chunks the input text into semantic segments.

        Args:
            text: The input string to chunk.

        Returns:
            A list of string chunks. Returns an empty list if input is empty
            or if chunking produces no results.
        """
        if not text or not text.strip():
            return []
        try:
            # The chunker function handles the actual chunking logic
            return self._chunker(text)
        except Exception as e:
            # Log unexpected errors during processing, but don't crash
            print(f"Chunking failed for text starting with '{text[:50]}...': {e}")
            return [] # Return empty list on processing failure