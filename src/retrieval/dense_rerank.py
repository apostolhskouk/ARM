from typing import List
from mxbai_rerank import MxbaiRerankV2
from src.retrieval.base import RetrievalResult
from src.retrieval.dense import FaissDenseRetriever
import numpy as np
from tqdm import tqdm
import torch
class DenseRetrieverWithReranker(FaissDenseRetriever):
    """
    Extends FaissDenseRetriever by adding a reranking step using mixedbread-ai's reranker.
    Performs a single batch retrieval then per-query reranking.
    """
    def __init__(
        self,
        embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
        reranker_model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        k_multiplier: int = 3
    ):
        super().__init__(model_name_or_path=embedding_model_name,)
        self.reranker = MxbaiRerankV2(reranker_model_name,device_map="cuda:1" )
        self._reranker_model_name = reranker_model_name
        self._k_multiplier = k_multiplier

    def retrieve(
        self,
        nlqs: List[str],
        output_folder: str,
        k: int
    ) -> List[List[RetrievalResult]]:

        initial_batches = super().retrieve(nlqs, output_folder, k*self._k_multiplier)
        final_batches: List[List[RetrievalResult]] = []
        torch.cuda.empty_cache()
        for i, (nlq, initial) in enumerate(tqdm(zip(nlqs, initial_batches),desc=f"Reranking with {self._reranker_model_name}")):
            if not initial:
                final_batches.append([])
                continue

            valid_initial_results = [r for r in initial if isinstance(r.object, str) and r.object]
            if not valid_initial_results:
                final_batches.append([])
                continue

            docs_to_rerank = [r.object for r in valid_initial_results]
            original_results_map = {r.object: r for r in valid_initial_results}

            try:
                reranked = self.reranker.rank(
                    query=nlq,
                    documents=docs_to_rerank,
                    top_k=k,
                    return_documents=True,
                    batch_size=8
                )

                processed_results: List[RetrievalResult] = []
                if reranked:
                    for item in reranked:
                        if not all(hasattr(item, attr) for attr in ['document', 'score']):
                             print(f"Warning: Skipping unexpected item format from reranker: {item}")
                             continue

                        doc_text = item.document
                        score = item.score       

                        original_result = original_results_map.get(doc_text)

                        if original_result:
                            # Still validate the score type/value
                            final_score = float(score) if isinstance(score, (int, float)) and np.isfinite(score) else original_result.score
                            processed_results.append(RetrievalResult(
                                score=final_score,
                                object=original_result.object, # Already validated as string
                                metadata=original_result.metadata
                            ))
                        # else: Should not happen

                else:
                    print(f"Warning: Reranker returned no results for query {i} with top_k={k}. Falling back.")
                    processed_results = valid_initial_results

                processed_results.sort(key=lambda r: r.score, reverse=True)
                final_batches.append(processed_results) # No need to slice [:k]

            except Exception as e:
                 # This except block should now handle potential errors correctly
                 print(f"ERROR during reranking step for query {i} ('{nlq[:50]}...'): {e}. Falling back.")
                 valid_initial_results.sort(key=lambda r: r.score, reverse=True)
                 final_batches.append(valid_initial_results[:k])

        return final_batches