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
        model_name: str = "gaunernst/gemma-3-27b-it-int4-awq",
        decomposition_cache_folder: Optional[str] = None,
        use_vllm: bool = True
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
                model_name_or_path=model_name,
                output_folder=decomposition_cache_folder,
                gpu_memory_utilization = 0.65
            )

    def retrieve(
        self,
        nlqs: List[str],
        output_folder: str,
        k: int
    ) -> List[List[RetrievalResult]]:
        """
        Decomposes all queries, retrieves for original + sub-queries,
        then combines and returns top-k per original query using Reciprocal Rank Fusion (RRF).
        """
        if not nlqs:
            return []

        # 1. Decompose queries and include the original query in the retrieval set
        all_queries_to_retrieve: List[str] = []
        original_query_indices: List[int] = []

        if self.use_vllm:
            decomposed_nlqs_batch: List[List[str]] = self.decomposer.decompose_batch(nlqs)
            for i, sub_queries in enumerate(decomposed_nlqs_batch):
                # Key Change: Add the original query to the list of queries to run
                queries_for_this_nlq = [nlqs[i]] + sub_queries
                for q in queries_for_this_nlq:
                    all_queries_to_retrieve.append(q)
                    original_query_indices.append(i)
        else:
            for i, nlq in enumerate(tqdm(nlqs, desc=f"Decomposing with {self.decomposer.ollama_model}")):
                sub_queries = self.decomposer.decompose(nlq) or []
                queries_for_this_nlq = [nlq] + sub_queries
                for q in queries_for_this_nlq:
                    all_queries_to_retrieve.append(q)
                    original_query_indices.append(i)

        if not all_queries_to_retrieve:
            return [[] for _ in nlqs]

        # 2. Retrieve for all queries (original + sub-queries) in one batch call
        all_retrieved_results_nested = super().retrieve(all_queries_to_retrieve, output_folder, k)

        # 3. Fuse results using Reciprocal Rank Fusion (RRF)
        rrf_k = 60  # RRF constant
        grouped_results: Dict[int, Dict[str, Dict]] = {i: {} for i in range(len(nlqs))}

        for query_idx, single_query_results in enumerate(all_retrieved_results_nested):
            original_idx = original_query_indices[query_idx]
            
            for rank, r in enumerate(single_query_results, 1):
                object_key = r.object
                rrf_score = 1 / (rrf_k + rank)

                if object_key not in grouped_results[original_idx]:
                    grouped_results[original_idx][object_key] = {
                        'score': rrf_score,
                        'result_obj': r
                    }
                else:
                    grouped_results[original_idx][object_key]['score'] += rrf_score

        # 4. Select top-k based on final RRF scores
        final_batches: List[List[RetrievalResult]] = []
        for i in range(len(nlqs)):
            results_with_scores = grouped_results[i].values()

            sorted_by_score = sorted(
                results_with_scores,
                key=lambda item: item['score'],
                reverse=True
            )

            top_k_results = [item['result_obj'] for item in sorted_by_score[:k]]
            final_batches.append(top_k_results)

        return final_batches