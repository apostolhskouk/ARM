from typing import List, Optional,Dict
from src.retrieval.base import RetrievalResult
from src.retrieval.dense_rerank import DenseRetrieverWithReranker
from src.utils.query_decompostion import QueryDecomposer
from src.utils.query_decomposition_vllm import QueryDecomposer as VLLMQueryDecomposer

from tqdm import tqdm


class DenseRetrieverWithDecompositionAndReranker(DenseRetrieverWithReranker):
    """
    Combines query decomposition with dense retrieval + reranking.
    For each original query:
      1. Decompose into sub-queries.
      2. Batch dense-retrieve + rerank all sub-queries in one call.
      3. Flatten, sort, and return top-k.
    """
    def __init__(
        self,
        embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
        reranker_model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        model_name: str = "llama3.1:8b",
        decomposition_cache_folder: Optional[str] = None,
        use_vllm: bool = True
    ):
        super().__init__(
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name
        )
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
        Decomposes all queries, retrieves+reranks results for all sub-queries
        in one batch call, then combines and returns top-k per original query.
        """
        if not nlqs:
            return []

        # 1. Decompose all queries first
        if self.use_vllm:
            decomposed_nlqs_batch: List[List[str]] = self.decomposer.decompose_batch(nlqs)
    
            all_sub_queries: List[str] = []
            original_query_indices: List[int] = []

            for i, single_nlq_decompositions in enumerate(decomposed_nlqs_batch):
                current_s_queries = single_nlq_decompositions if single_nlq_decompositions else [nlqs[i]]

                for sub_q in current_s_queries:
                    all_sub_queries.append(sub_q)
                    original_query_indices.append(i)

        else:
    
            all_sub_queries: List[str] = []
            original_query_indices: List[int] = []

            for i, nlq in enumerate(tqdm(nlqs, desc=f"Decomposing with {self.decomposer.ollama_model}")):
                sub_queries = self.decomposer.decompose(nlq) or [nlq]
                for sub_q in sub_queries:
                    all_sub_queries.append(sub_q)
                    original_query_indices.append(i)


        if not all_sub_queries:
            return [[] for _ in nlqs]

        # 2. Retrieve + Rerank for all sub-queries in one batch call
        # It returns List[List[RetrievalResult]], one list per sub_query
        all_sub_results_nested = super().retrieve(all_sub_queries, output_folder, k)

        # 3. Re-group results by original query index
        grouped_results: Dict[int, List[RetrievalResult]] = {i: [] for i in range(len(nlqs))}
        seen_texts_per_query: Dict[int, set] = {i: set() for i in range(len(nlqs))}

        for sub_query_idx, sub_results in enumerate(all_sub_results_nested):
            original_idx = original_query_indices[sub_query_idx]
            current_seen = seen_texts_per_query[original_idx]
            for r in sub_results:
                # 4. Deduplicate within each original query's results
                if r.object not in current_seen:
                    current_seen.add(r.object)
                    grouped_results[original_idx].append(r)

        # 5. Sort and take top-k for each original query
        final_batches: List[List[RetrievalResult]] = []
        for i in range(len(nlqs)):
            results_for_query = grouped_results[i]
            # Results from super().retrieve should already be reranked and sorted
            # But we aggregate from multiple sub-queries, so we must resort
            results_for_query.sort(key=lambda x: x.score, reverse=True)
            final_batches.append(results_for_query[:k])

        return final_batches
