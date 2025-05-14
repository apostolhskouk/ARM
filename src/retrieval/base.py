import abc
import json
from dataclasses import dataclass, field 
from typing import List, Dict, Any


@dataclass
class RetrievalResult:
    """Stores a single retrieval result."""
    score: float
    object: str # The retrieved text from the indexed field
    metadata: Dict[str, Any] = field(default_factory=dict) # Dictionary for metadata

    def __repr__(self) -> str:
        # Provide a cleaner representation for printing lists of results
        metadata_str = ", ".join(f"{k}: {v}" for k, v in self.metadata.items())
        return f"RetrievalResult(score={self.score:.4f}, object='{self.object}...', metadata={{{metadata_str}}})"

    def __hash__(self):
        """Define hash based on the 'object' field for uniqueness in sets."""
        return hash(self.object)

    def __eq__(self, other):
        """Define equality based on the 'object' field."""
        if not isinstance(other, RetrievalResult):
            return NotImplemented
        return self.object == other.object

# --- Abstract Base Class for Retrievers ---

class BaseRetriever(abc.ABC):
    """
    Abstract Base Class for different retrieval indexing and querying implementations.
    """

    @abc.abstractmethod
    def index(self,
            input_jsonl_path: str,
            output_folder: str,
            field_to_index: str,
            metadata_fields: List[str]) -> None:
        """
        Indexes documents from a JSON Lines file.

        Args:
            input_jsonl_path: Path to the JSON Lines file containing documents.
                            Each line is expected to be a JSON object.
            output_folder: Path to the directory where index files will be stored.
                        This directory will be created if it doesn't exist.
            field_to_index: The name of the field within each JSON object
                            whose content should be indexed for searching.
            metadata_fields: A list of field names within each JSON object
                            to store as metadata alongside the indexed field.
                            These fields can be retrieved later.
        """
        pass

    @abc.abstractmethod
    def retrieve(self,
                nlqs: List[str],
                output_folder: str, # Changed from index_folder to match index method arg name
                k: int) -> List[List[RetrievalResult]]:
        """
        Retrieves the top-k relevant documents for a given natural language query.

        Args:
            nlq: The natural language query string.
            output_folder: Path to the directory containing the previously built index files.
            k: The maximum number of results to return.

        Returns:
            A list of RetrievalResult objects, sorted by relevance score (descending).
            Each object contains the score, the retrieved object string (from the
            indexed field), and a dictionary of requested metadata.
        """
        pass