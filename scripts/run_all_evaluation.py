import json
import time
import gc
from pathlib import Path
import pandas as pd
import wandb
from typing import List, Dict, Tuple, Type
from src.utils.evaluator import EvaluationMetrics
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from src.retrieval.dense_rerank import DenseRetrieverWithReranker
from src.retrieval.dense_decomp import DenseRetrieverWithDecomposition
from src.retrieval.dense_decomp_rerank import DenseRetrieverWithDecompositionAndReranker
from src.retrieval.arm_beam import ARMRetriever
from src.retrieval.react import ReActRetriever

BENCHMARK_DIR = Path("assets/all_data/benchmarks_subsampled")
BENCHMARK_FILENAMES = [
    "bird.json", "fetaqa.json", "feverous.json", "mt_raig.json",
    "multi_hop_rag.json", "ottqa.json", "spider.json", "tabfact.json",
    "table_bench.json"
]

RETRIEVERS_CONFIG: Dict[str, Type[BaseRetriever]] = {
    "BM25": PyseriniBM25Retriever,
    "Dense": FaissDenseRetriever,
    "DenseRerank": DenseRetrieverWithReranker,
    "DenseDecomp": DenseRetrieverWithDecomposition,
    "DenseDecompRerank": DenseRetrieverWithDecompositionAndReranker,
    "ARM": ARMRetriever,
    "ReAct": ReActRetriever,
}

BM25_INDEX_BASE_DIR = Path("assets/all_data/indexes/bm25/")
DENSE_INDEX_BASE_DIR = Path("assets/all_data/indexes/dense_infly_v1_1.5b/")
EVALUATION_N_VALUES = [1, 3, 5, 10]
RETRIEVAL_K = 50
WANDB_PROJECT_NAME = "all_benchamrks"
WANDB_ENTITY_NAME = "lakhs"

DECOMPOSITION_MODEL_NAME = "gaunernst/gemma-3-27b-it-int4-awq"
DECOMPOSITION_CACHE_FOLDER = "assets/all_decompositions/"
EMBEDDING_MODEL_INF_RETRIEVER = "infly/inf-retriever-v1-1.5b"
REACT_LLM_MODEL_PATH = "assets/cache/gemma-3-27b-it.Q4_K_M.gguf"
ARM_VLLM_MODEL_PATH = "gaunernst/gemma-3-27b-it-int4-awq"
ARM_NGRAM_LLM_MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_benchmark_file_stem(benchmark_filename: str) -> str:
    return Path(benchmark_filename).stem

def get_index_folder_for_retrieval(retriever_name: str, benchmark_stem: str) -> Path:
    is_bm25 = retriever_name == "BM25"
    base_path = BM25_INDEX_BASE_DIR if is_bm25 else DENSE_INDEX_BASE_DIR
    return base_path / benchmark_stem

def load_benchmark_data(benchmark_filepath: Path) -> Tuple[List[str], List[List[str]]]:
    with open(benchmark_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    queries = [item["query"] for item in data]
    ground_truth_ids_list = [item["document_ids"] for item in data]
    return queries, ground_truth_ids_list

def main():
    evaluator = EvaluationMetrics(n_values=EVALUATION_N_VALUES)

    for retriever_name, RetrieverClass in RETRIEVERS_CONFIG.items():
        retriever_instance = RetrieverClass()
        is_agentic_retriever = isinstance(retriever_instance, (ARMRetriever, ReActRetriever))

        for benchmark_filename in BENCHMARK_FILENAMES:
            benchmark_stem = get_benchmark_file_stem(benchmark_filename)
            benchmark_filepath = BENCHMARK_DIR / benchmark_filename

            run_name = f"{retriever_name}-{benchmark_stem}"
            group_name = f"retriever-{retriever_name}"
            
            config_dict = {
                "retriever_name": retriever_name,
                "benchmark_file": benchmark_filename,
                "benchmark_stem": benchmark_stem,
                "evaluation_n_values": EVALUATION_N_VALUES,
                "retrieval_k_passed_to_method": RETRIEVAL_K,
                "wandb_project": WANDB_PROJECT_NAME,
                "wandb_entity": WANDB_ENTITY_NAME,
            }
            
            run = wandb.init(
                project=WANDB_PROJECT_NAME,
                entity=WANDB_ENTITY_NAME,
                name=run_name,
                group=group_name,
                config=config_dict,
                reinit=True,
                job_type="evaluation"
            )

            index_folder_for_retrieval = get_index_folder_for_retrieval(retriever_name, benchmark_stem)
            queries, ground_truth_ids_list = load_benchmark_data(benchmark_filepath)
            num_queries = len(queries)

            start_time = time.time()
            retrieved_results_nested_list: List[List[RetrievalResult]] = retriever_instance.retrieve(
                nlqs=queries,
                output_folder=str(index_folder_for_retrieval),
                k=RETRIEVAL_K
            )
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            wandb.log({
                "retrieval_time_seconds": retrieval_time,
                "queries_per_second": num_queries / retrieval_time if retrieval_time > 0 else 0
            })

            predicted_doc_ids_list = [[res.object for res in inner_list] for inner_list in retrieved_results_nested_list]

            if is_agentic_retriever:
                avg_distinct, avg_calls = retriever_instance.display_metrics(verbose=False)
                avg_distinct = avg_distinct if avg_distinct is not None else 0
                avg_calls = avg_calls if avg_calls is not None else 0

                wandb.log({
                    f"agentic_avg_distinct_retrieved_objects": avg_distinct,
                    f"agentic_avg_llm_search_calls": avg_calls
                })

                k_for_eval = max(1, int(round(avg_distinct)))
                
                metrics_at_k_for_eval = evaluator.calculate_metrics_for_single_n(
                    ground_truth_ids=ground_truth_ids_list,
                    predicted_ids=predicted_doc_ids_list,
                    n_value=k_for_eval
                )

                agentic_summary_df_data = {
                    "#calls ↓": [avg_calls],
                    "Avg #obj. ↓": [avg_distinct],
                    "P @avg_obj": [metrics_at_k_for_eval.get("Precision", 0.0) * 100.0],
                    "R @avg_obj": [metrics_at_k_for_eval.get("Recall", 0.0) * 100.0],
                    "F1 @avg_obj": [metrics_at_k_for_eval.get("F1", 0.0) * 100.0],
                    "PR @avg_obj": [metrics_at_k_for_eval.get("Perfect Recall", 0.0) * 100.0],
                    "k_for_eval": [k_for_eval]
                }
                agentic_summary_df = pd.DataFrame(agentic_summary_df_data)
                
                wandb.log({f"agentic_summary_table/{benchmark_stem}": wandb.Table(dataframe=agentic_summary_df)})
                
                wandb.log({
                    f"agentic_{benchmark_stem}_P_at_avg_obj": metrics_at_k_for_eval.get("Precision", 0.0) * 100.0,
                    f"agentic_{benchmark_stem}_R_at_avg_obj": metrics_at_k_for_eval.get("Recall", 0.0) * 100.0,
                    f"agentic_{benchmark_stem}_F1_at_avg_obj": metrics_at_k_for_eval.get("F1", 0.0) * 100.0,
                    f"agentic_{benchmark_stem}_PR_at_avg_obj": metrics_at_k_for_eval.get("Perfect Recall", 0.0) * 100.0,
                    f"agentic_{benchmark_stem}_k_for_eval": k_for_eval,
                })

            else:
                results_df = evaluator.calculate_metrics(
                    ground_truth_ids=ground_truth_ids_list,
                    predicted_ids=predicted_doc_ids_list
                )
                wandb.log({f"evaluation_table/{benchmark_stem}": wandb.Table(dataframe=results_df.reset_index())})

                for n_val in results_df.index:
                    wandb.log({
                        f"eval_{benchmark_stem}_Precision@{n_val}": results_df.loc[n_val, 'Precision'],
                        f"eval_{benchmark_stem}_Recall@{n_val}": results_df.loc[n_val, 'Recall'],
                        f"eval_{benchmark_stem}_F1@{n_val}": results_df.loc[n_val, 'F1'],
                        f"eval_{benchmark_stem}_PerfectRecall@{n_val}": results_df.loc[n_val, 'Perfect Recall'],
                        f"eval_{benchmark_stem}_Perc_Preds_Less_Than_n@{n_val}": results_df.loc[n_val, 'Perc_Preds_Less_Than_n']
                    })
            
            wandb.log({"status": "completed"})
            wandb.finish()
            run = None 
        
        del retriever_instance
        gc.collect()

if __name__ == "__main__":
    main()