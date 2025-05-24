from typing import List, Optional, Dict, Tuple
from collections import Counter
from src.retrieval.base import RetrievalResult
from src.retrieval.dense import FaissDenseRetriever
from src.utils.query_decompostion import QueryDecomposer
from src.utils.query_decomposition_vllm import QueryDecomposer as VLLMQueryDecomposer
from tqdm import tqdm
class DenseRetrieverWithDecomposition(FaissDenseRetriever):
    """
    Extends FaissDenseRetriever to perform query decomposition before retrieval.
    Retrieves candidates for each sub-query and combines/ranks them based on voting.
    """
    def __init__(
        self,
        embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
        model_name: str = "llama3.1:8b",
        decomposition_cache_folder: Optional[str] = None,
        use_vllm: bool = False
    ):
        super().__init__(model_name_or_path=embedding_model_name)
        self.use_vllm = use_vllm
        if not self.use_vllm:
            self.decomposer = QueryDecomposer(
                model_name,
                output_folder=decomposition_cache_folder
            )
        else:
            self.decomposer = VLLMQueryDecomposer(
                output_folder=decomposition_cache_folder
            )

    def retrieve(
        self,
        nlqs: List[str],
        output_folder: str,
        k: int
    ) -> List[List[RetrievalResult]]:
        """
        Decomposes all queries, retrieves results for all sub-queries,
        then combines and returns top-k per original query based on voting.
        """
        if not nlqs:
            return []

        if self.use_vllm:
            decomposed_nlqs_batch: List[List[str]] = self.decomposer.decompose_batch(nlqs)
    
            all_sub_queries: List[str] = []
            original_query_indices: List[int] = []
            sub_query_groups: List[List[str]] = [] 

            for i, single_nlq_decompositions in enumerate(decomposed_nlqs_batch):
                current_s_queries = single_nlq_decompositions if single_nlq_decompositions else [nlqs[i]]
                sub_query_groups.append(current_s_queries)

                for sub_q in current_s_queries:
                    all_sub_queries.append(sub_q)
                    original_query_indices.append(i)

        else:
    
            all_sub_queries: List[str] = []
            original_query_indices: List[int] = []
            sub_query_groups: List[List[str]] = []

            for i, nlq in enumerate(tqdm(nlqs, desc=f"Decomposing with {self.decomposer.ollama_model}")):
                sub_queries = self.decomposer.decompose(nlq) or [nlq]
                sub_query_groups.append(sub_queries)
                for sub_q in sub_queries:
                    all_sub_queries.append(sub_q)
                    original_query_indices.append(i)

        if not all_sub_queries:
            return [[] for _ in nlqs]

        # 2. Retrieve for all sub-queries in one batch call (Keep this part the same)
        all_sub_results_nested = super().retrieve(all_sub_queries, output_folder, k)

        # 3. Re-group results and count occurrences (votes) by original query index
        # Store counts and one representative RetrievalResult per unique object
        grouped_votes: Dict[int, Dict[str, Tuple[int, RetrievalResult]]] = {i: {} for i in range(len(nlqs))}

        for sub_query_idx, sub_results in enumerate(all_sub_results_nested):
            original_idx = original_query_indices[sub_query_idx]
            current_group_votes = grouped_votes[original_idx]
            for r in sub_results:
                object_key = r.object # Assume r.object is the unique identifier (e.g., text or ID)
                if object_key not in current_group_votes:
                    # First time seeing this object for this original query
                    current_group_votes[object_key] = (1, r) # Store count = 1 and the result object
                else:
                    # Increment vote count, keep the existing representative result object
                    count, existing_result = current_group_votes[object_key]
                    current_group_votes[object_key] = (count + 1, existing_result)

        # 4. Select top-k based on votes for each original query
        final_batches: List[List[RetrievalResult]] = []
        for i in range(len(nlqs)):
            # Get the dictionary of {object_key: (vote_count, result_obj)} for the current query
            results_with_votes = grouped_votes[i]

            # Sort items by vote count (descending)
            # items() gives [(key, (count, result)), ...]
            # We sort by item[1][0] which is the count
            sorted_by_votes = sorted(
                results_with_votes.items(),
                key=lambda item: item[1][0],
                reverse=True
            )

            # Extract the RetrievalResult objects from the top-k voted items
            top_k_results = [result_obj for key, (count, result_obj) in sorted_by_votes[:k]]
            final_batches.append(top_k_results)

        return final_batches