import json
import time
from typing import List, Dict
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from src.retrieval.dense_rerank import DenseRetrieverWithReranker
from src.retrieval.dense_decomp import DenseRetrieverWithDecomposition
from src.retrieval.dense_decomp_rerank import DenseRetrieverWithDecompositionAndReranker
from src.feverous.feverous_evaluator import FeverousEvaluation
import argparse
from pathlib import Path
import pandas as pd
import wandb

BENCHMARK_FILE_PATH = Path("assets/feverous/benchmark.json")
INDEX_BASE_DIR = Path("assets/feverous/")
DATA_DIR = Path("assets/feverous/serialized_output")
DECOMP_CACHE_DIR = INDEX_BASE_DIR / "decompositions_cache"
INDEX_LEVELS_TO_RUN = ["table", "row", "cell"]
K_RESULTS = 30
EVALUATION_N_VALUES = [1, 3, 5, 10] 
SERIALIZATION_FILENAMES = {
    "table": "serialized_table_level.jsonl",
    "row": "serialized_row_level.jsonl",
    "cell": "serialized_cell_level.jsonl",
}
METADATA_FIELDS_TO_INDEX = ["page_title", "source"]
FIELD_TO_INDEX = "object"
REACT_LLM_MODEL_PATH = "assets/cache/Qwen2.5-32B-Instruct-Q4_K_M.gguf" 
EMBEDDING_MODEL_NAME = "WhereIsAI/UAE-Large-V1"
RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"
OLLAMA_MODEL = "llama3.1:8b"

def get_output_folder(base_dir: Path, level: str, retriever_instance: BaseRetriever, model_name: str) -> Path:
    """Determines the index output folder based on retriever type and level."""
    retriever_name = type(retriever_instance).__name__
    if isinstance(retriever_instance, (FaissDenseRetriever, DenseRetrieverWithReranker, DenseRetrieverWithDecomposition, DenseRetrieverWithDecompositionAndReranker)):
        safe_model_name = model_name.split('/')[-1].replace('_', '-')
        return base_dir / "faiss_indexes" / f"dense_{level}_{safe_model_name}"
    elif isinstance(retriever_instance, PyseriniBM25Retriever):
        return base_dir / "pyserini_indexes" / f"bm25_{level}_index"
    else:
        raise ValueError(f"Unknown retriever type {retriever_name}, cannot determine index path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="(Optional) WandB entity name; defaults to your authenticated entity"
    )
    args = parser.parse_args()
    with open(BENCHMARK_FILE_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f) 
    all_nlqs = [record.get('claim') for record in benchmark_data if record.get('claim')]
    num_queries = len(all_nlqs)
    evaluator = FeverousEvaluation(n_values=EVALUATION_N_VALUES)
    all_evaluation_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {} 
    # --- Loop through Index Levels ---
    for index_level in INDEX_LEVELS_TO_RUN:
        all_evaluation_results[index_level] = {}
        # --- Determine Input Data for this Level ---
        input_jsonl_filename = SERIALIZATION_FILENAMES.get(index_level)
        input_jsonl_path = DATA_DIR / input_jsonl_filename
        retrievers_for_level: List[BaseRetriever] = []
        retriever_instances = {
            "BM25": lambda: PyseriniBM25Retriever(),
            "Dense": lambda: FaissDenseRetriever(model_name_or_path=EMBEDDING_MODEL_NAME),
            #"Dense+Rerank": lambda: DenseRetrieverWithReranker(embedding_model_name=EMBEDDING_MODEL_NAME, reranker_model_name=RERANKER_MODEL_NAME),
            #"Dense+Decomp": lambda: DenseRetrieverWithDecomposition(embedding_model_name=EMBEDDING_MODEL_NAME, model_name=OLLAMA_MODEL, decomposition_cache_folder=str(DECOMP_CACHE_DIR)),
            #"Dense+Decomp+Rerank": lambda: DenseRetrieverWithDecompositionAndReranker(embedding_model_name=EMBEDDING_MODEL_NAME, reranker_model_name=RERANKER_MODEL_NAME, ollama_model=OLLAMA_MODEL, decomposition_cache_folder=str(DECOMP_CACHE_DIR)),
        }
        for name, init_func in retriever_instances.items():
            retriever_instance = init_func()
            retrievers_for_level.append(retriever_instance)
        for retriever in retrievers_for_level:
            retriever_name = type(retriever).__name__
            run_config = {
                "retriever": retriever_name,
                "index_level": index_level,
                "k_results": K_RESULTS,
                "evaluation_n_values": EVALUATION_N_VALUES,
                "embedding_model": EMBEDDING_MODEL_NAME if isinstance(retriever, (FaissDenseRetriever, DenseRetrieverWithReranker, DenseRetrieverWithDecomposition, DenseRetrieverWithDecompositionAndReranker,)) else None, 
                "reranker_model": RERANKER_MODEL_NAME if isinstance(retriever, (DenseRetrieverWithReranker, DenseRetrieverWithDecompositionAndReranker)) else None,
                "ollama_model": OLLAMA_MODEL if isinstance(retriever, (DenseRetrieverWithDecomposition, DenseRetrieverWithDecompositionAndReranker)) else None,
                "input_data": str(input_jsonl_path.name),
                "benchmark_file": str(BENCHMARK_FILE_PATH.name),
            }
            run = wandb.init(
                project="feverous-retrieval-benchmark-sample",
                entity=args.wandb_entity,
                group=f"level-{index_level}",
                name=f"run-{index_level}-{retriever_name}",
                config=run_config,
                reinit=True,
                job_type="evaluation"
            )
            output_folder = get_output_folder(INDEX_BASE_DIR, index_level, retriever, EMBEDDING_MODEL_NAME)
            retriever.index(
                input_jsonl_path=str(input_jsonl_path),
                output_folder=str(output_folder),
                field_to_index=FIELD_TO_INDEX,
                metadata_fields=METADATA_FIELDS_TO_INDEX
            )
            start_time = time.time()
            retrieved_results: List[List[RetrievalResult]] = retriever.retrieve(
                nlqs=all_nlqs,
                output_folder=str(output_folder),
                k=K_RESULTS # K_RESULTS is ignored by ReAct's retrieve logic, but passed for consistency
            )
            end_time = time.time()
            retrieval_time = end_time - start_time
            # --- Log Retrieval Time ---
            wandb.log({"retrieval_time_seconds": retrieval_time})
            wandb.log({"queries_per_second": num_queries / retrieval_time})

            evaluation_output = evaluator.evaluate(
                json_path=str(BENCHMARK_FILE_PATH),
                predictions=retrieved_results
            )
            # Store raw output for final reporting if needed outside WandB
            all_evaluation_results[index_level][retriever_name] = evaluation_output
            # Add react specific metrics if they exist, for final reporting
            # --- Standard Logging: Multiple Tables ---
            metrics_to_log = {}
            summary_metrics = {}
            for eval_level_name, df in evaluation_output.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Log DataFrame as wandb Table per eval level
                    metrics_to_log[f"evaluation_table/{eval_level_name}"] = wandb.Table(dataframe=df.reset_index()) # Include @n column

                    # Log key metrics (e.g., @K_RESULTS) as scalars
                    if K_RESULTS in df.index:
                        summary_metrics[f"eval_{eval_level_name}_Precision@{K_RESULTS}"] = df.loc[K_RESULTS, 'Precision']
                        summary_metrics[f"eval_{eval_level_name}_Recall@{K_RESULTS}"] = df.loc[K_RESULTS, 'Recall']
                        summary_metrics[f"eval_{eval_level_name}_F1@{K_RESULTS}"] = df.loc[K_RESULTS, 'F1']
                        summary_metrics[f"eval_{eval_level_name}_PerfectRecall@{K_RESULTS}"] = df.loc[K_RESULTS, 'Perfect Recall']
                    # Log metrics for other N values as well
                    for n_val in EVALUATION_N_VALUES:
                        if n_val != K_RESULTS and n_val in df.index: 
                            summary_metrics[f"eval_{eval_level_name}_Precision@{n_val}"] = df.loc[n_val, 'Precision']
                            summary_metrics[f"eval_{eval_level_name}_Recall@{n_val}"] = df.loc[n_val, 'Recall']
                            summary_metrics[f"eval_{eval_level_name}_F1@{n_val}"] = df.loc[n_val, 'F1']
                            summary_metrics[f"eval_{eval_level_name}_PerfectRecall@{n_val}"] = df.loc[n_val, 'Perfect Recall']

                wandb.log(metrics_to_log)
                if summary_metrics:
                    wandb.log(summary_metrics) # Log scalar summary metrics

            # --- Common Logging for all Retrievers ---
            wandb.log({"status": "completed"})
            wandb.finish()
            run = None # Reset run variable

    # --- Final Combined Reporting ---
    print("\n\n" + "="*30 + " FINAL EVALUATION SUMMARY " + "="*30)

    for level, level_results in all_evaluation_results.items():
        print(f"\n--- INDEX LEVEL: {level.upper()} ---")
        if not level_results:
            print("No results for this level.")
            continue

        for retriever_name, eval_data in level_results.items():
            print(f"\n  --- Retriever: {retriever_name} ---")
            if isinstance(eval_data.get("error"), str):
                print(f"    ERROR during processing: {eval_data['error']}")
                 # Print react metrics even if there was an error later
                if "react_specific_metrics" in eval_data:
                    metrics = eval_data["react_specific_metrics"]
                    avg_distinct = metrics.get('avg_distinct_retrieved_objects', 'N/A')
                    avg_calls = metrics.get('avg_llm_search_calls', 'N/A')
                    print(f"    ReAct Specific Metrics (collected before error):")
                    print(f"      Average Distinct Retrieved Objects: {avg_distinct:.2f}" if isinstance(avg_distinct, float) else f"      Average Distinct Retrieved Objects: {avg_distinct}")
                    print(f"      Average LLM Search Calls: {avg_calls:.2f}" if isinstance(avg_calls, float) else f"      Average LLM Search Calls: {avg_calls}")
                continue # Skip detailed evaluation metric printing on error

            if not eval_data:
                 print("    No evaluation data recorded.")
                 continue

            # Print ReAct specific metrics if they exist
            if "react_specific_metrics" in eval_data:
                metrics = eval_data["react_specific_metrics"]
                avg_distinct = metrics.get('avg_distinct_retrieved_objects', 'N/A')
                avg_calls = metrics.get('avg_llm_search_calls', 'N/A')
                print(f"    ReAct Specific Metrics:")
                # Format numbers nicely if they are floats
                print(f"      Average Distinct Retrieved Objects: {avg_distinct:.2f}" if isinstance(avg_distinct, float) else f"      Average Distinct Retrieved Objects: {avg_distinct}")
                print(f"      Average LLM Search Calls: {avg_calls:.2f}" if isinstance(avg_calls, float) else f"      Average LLM Search Calls: {avg_calls}")

            # Print Standard Evaluation Metrics
            print("    Evaluation Metrics (@N):")
            standard_eval_metrics_found = False
            for eval_level_name, df in eval_data.items():
                # Skip special keys used for storage
                if eval_level_name in ["react_specific_metrics", "error"]:
                    continue

                standard_eval_metrics_found = True
                print(f"\n      Evaluation Level: {eval_level_name}")
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Format numbers for better readability in print
                    df_display = df.copy()
                    for col in ['Precision', 'Recall', 'F1', 'Perfect Recall']:
                        if col in df_display.columns:
                            df_display[col] = df_display[col].map('{:.4f}'.format)
                    if 'Perc_Preds_Less_Than_n' in df_display.columns:
                         # Check if column contains numeric data before formatting
                        if pd.api.types.is_numeric_dtype(df_display['Perc_Preds_Less_Than_n']):
                            df_display['Perc_Preds_Less_Than_n'] = df_display['Perc_Preds_Less_Than_n'].map('{:.1f}%'.format)
                        else:
                             df_display['Perc_Preds_Less_Than_n'] = df_display['Perc_Preds_Less_Than_n'].astype(str) # Convert to string if not numeric

                    print(df_display.to_string())
                else:
                    print("      No results DataFrame found or DataFrame is empty.")

            # Handle cases where only react metrics might exist but no standard eval DFs
            if not standard_eval_metrics_found and "react_specific_metrics" not in eval_data: # Check if NO metrics were printed at all
                 print("    No evaluation data recorded.")
            elif not standard_eval_metrics_found and "react_specific_metrics" in eval_data:
                 # This case means ReAct metrics were printed, but evaluation DFs were empty/missing
                 print("    Standard evaluation metric DataFrames (P/R/F1) are missing or empty.")


    print("\n" + "="*80)
    print("Combined benchmark finished.")