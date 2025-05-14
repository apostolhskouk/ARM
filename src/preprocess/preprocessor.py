from abc import ABC, abstractmethod
from typing import List

class TextPreprocessor(ABC):
    """Abstract base class for text preprocessing."""

    @abstractmethod
    def pre_process(self, text: str) -> List[str]:
        """Processes raw text into a list of strings (e.g., chunks,tokens)."""
        pass