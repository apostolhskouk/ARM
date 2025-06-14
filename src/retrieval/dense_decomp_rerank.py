from typing import List, Optional,Dict
from src.retrieval.base import RetrievalResult
from src.retrieval.dense import FaissDenseRetriever
from src.utils.query_decompostion import QueryDecomposer
from src.utils.query_decomposition_vllm import QueryDecomposer as VLLMQueryDecomposer
from mxbai_rerank import MxbaiRerankV2
import torch
from tqdm import tqdm


class DenseRetrieverWithDecompositionAndReranker(FaissDenseRetriever):
    """
    Implements an advanced retrieval pipeline with full reranking.
    This strategy prioritizes maximum quality over performance.
    1. Decomposes a query into sub-queries.
    2. Gathers a large set of candidates using fast dense retrieval for the original
       query and all sub-queries.
    3. Aggregates and deduplicates all candidates.
    4. Reranks the ENTIRE set of unique candidates using a powerful cross-encoder
       against the original query to get the final results.
    """
    def __init__(
        self,
        embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
        reranker_model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        model_name: str = "gaunernst/gemma-3-27b-it-int4-awq",
        decomposition_cache_folder: Optional[str] = None,
        use_vllm: bool = True
    ):
        super().__init__(model_name_or_path=embedding_model_name)
        self.reranker = MxbaiRerankV2(reranker_model_name, device_map="cuda:1")
        self._reranker_model_name = reranker_model_name

        self.use_vllm = use_vllm
        if not self.use_vllm:
            self.decomposer = QueryDecomposer(
                model_name,
                output_folder=decomposition_cache_folder
            )
        else:
            self.decomposer = VLLMQueryDecomposer(
                model_name_or_path=model_name,
                output_folder=decomposition_cache_folder
            )

    def retrieve(
        self,
        nlqs: List[str],
        output_folder: str,
        k: int,
    ) -> List[List[RetrievalResult]]:
        """
        Executes the full retrieval and reranking pipeline without RRF.
        """
        if not nlqs:
            return []

        # === STAGE 1: DECOMPOSITION & CANDIDATE GATHERING ===
        all_queries_to_retrieve: List[str] = []
        original_query_indices: List[int] = []

        if self.use_vllm:
            decomposed_nlqs_batch: List[List[str]] = self.decomposer.decompose_batch(nlqs)
            for i, sub_queries in enumerate(decomposed_nlqs_batch):
                queries_for_this_nlq = [nlqs[i]] + sub_queries
                for q in queries_for_this_nlq:
                    all_queries_to_retrieve.append(q)
                    original_query_indices.append(i)
        else:
            for i, nlq in enumerate(tqdm(nlqs, desc="Decomposing queries")):
                sub_queries = self.decomposer.decompose(nlq) or []
                queries_for_this_nlq = [nlq] + sub_queries
                for q in queries_for_this_nlq:
                    all_queries_to_retrieve.append(q)
                    original_query_indices.append(i)

        if not all_queries_to_retrieve:
            return [[] for _ in nlqs]

        candidate_results_nested = super().retrieve(
            all_queries_to_retrieve, output_folder, k
        )

        # === STAGE 2: AGGREGATE AND DEDUPLICATE CANDIDATES ===
        # We collect all unique documents in a dictionary to deduplicate.
        deduplicated_candidates: Dict[int, Dict[str, RetrievalResult]] = {i: {} for i in range(len(nlqs))}

        for query_idx, single_query_results in enumerate(candidate_results_nested):
            original_nlq_idx = original_query_indices[query_idx]
            for r in single_query_results:
                # The dictionary key handles deduplication automatically.
                deduplicated_candidates[original_nlq_idx][r.object] = r

        # === STAGE 3: FULL RERANKING OF ALL UNIQUE CANDIDATES ===
        final_batches: List[List[RetrievalResult]] = []
        torch.cuda.empty_cache()

        for i, nlq in enumerate(tqdm(nlqs, desc=f"Full reranking with {self._reranker_model_name}")):
            unique_candidates_map = deduplicated_candidates[i]
            if not unique_candidates_map:
                final_batches.append([])
                continue
            
            # Prepare the full list of unique document texts for the reranker
            docs_to_rerank = list(unique_candidates_map.keys())

            try:
                # Rerank the ENTIRE deduplicated set against the ORIGINAL query
                reranked_output = self.reranker.rank(
                    query=nlq,
                    documents=docs_to_rerank,
                    top_k=k,
                    return_documents=True,
                    batch_size=32 
                )

                reranked_results: List[RetrievalResult] = []
                for item in reranked_output:
                    # Retrieve the original result object to preserve metadata
                    original_result = unique_candidates_map.get(item.document)
                    if original_result:
                        reranked_results.append(RetrievalResult(
                            object=item.document,
                            score=float(item.score),
                            metadata=original_result.metadata
                        ))
                
                final_batches.append(reranked_results)

            except Exception as e:
                print(f"ERROR during full reranking for query '{nlq[:50]}...': {e}.")
                final_batches.append([])

        return final_batches