import json
import time
import os
from typing import List, Dict
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from src.retrieval.dense_rerank import DenseRetrieverWithReranker
from src.retrieval.dense_decomp import DenseRetrieverWithDecomposition
from src.retrieval.dense_decomp_rerank import DenseRetrieverWithDecompositionAndReranker
from src.feverous.feverous_evaluator import FeverousEvaluation
from pathlib import Path
import pandas as pd
import wandb
import traceback
from src.retrieval.react import ReActRetriever # Assuming ReActRetriever is in this location

BENCHMARK_FILE_PATH = Path("assets/feverous/benchmark.json") # Use Path object
INDEX_BASE_DIR = Path("assets/feverous/")
DATA_DIR = Path("assets/feverous/serialized_output")
DECOMP_CACHE_DIR = INDEX_BASE_DIR / "decompositions_cache"

# Levels to iterate through
INDEX_LEVELS_TO_RUN = ["table", "row", "cell"]
K_RESULTS = 30
EVALUATION_N_VALUES = [1, 3, 5] # N values for P/R/F1@N

SERIALIZATION_FILENAMES = {
    "table": "serialized_table_level.jsonl",
    "row": "serialized_row_level.jsonl",
    "cell": "serialized_cell_level.jsonl",
}
METADATA_FIELDS_TO_INDEX = ["page_title", "source"]
FIELD_TO_INDEX = "object"
# Define REACT_LLM_MODEL_NAME (ensure this path is correct)
REACT_LLM_MODEL_PATH = "assets/cache/Qwen2.5-32B-Instruct-Q4_K_M.gguf" 
# Models (Use environment variables or a config file ideally)
EMBEDDING_MODEL_NAME = "WhereIsAI/UAE-Large-V1"
RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"
OLLAMA_MODEL = "llama3.1:8b"
VLLM_MODEL = "Qwen/Qwen3-32B" 
# Helper to get index path
def get_output_folder(base_dir: Path, level: str, retriever_instance: BaseRetriever, model_name: str) -> Path:
    """Determines the index output folder based on retriever type and level."""
    retriever_name = type(retriever_instance).__name__
    if isinstance(retriever_instance, (FaissDenseRetriever, DenseRetrieverWithReranker, DenseRetrieverWithDecomposition, DenseRetrieverWithDecompositionAndReranker,ReActRetriever)):
        # Use the embedding model name for dense retrievers including ReAct
        safe_model_name = model_name.split('/')[-1].replace('_', '-')
        return base_dir / "faiss_indexes" / f"dense_{level}_{safe_model_name}"
    elif isinstance(retriever_instance, PyseriniBM25Retriever):
        return base_dir / "pyserini_indexes" / f"bm25_{level}_index"
    else:
        raise ValueError(f"Unknown retriever type {retriever_name}, cannot determine index path.")

if __name__ == "__main__":

    # --- Prepare Environment ---
    print("--- Preparing Environment ---")
    INDEX_BASE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DECOMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Benchmark Data ---
    print(f"\n--- Loading Benchmark Data from: {BENCHMARK_FILE_PATH} ---")

    with open(BENCHMARK_FILE_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f) # Assuming benchmark is a JSON list of objects

    all_nlqs = [record.get('claim') for record in benchmark_data if record.get('claim')]
    if not all_nlqs:
        print("Error: No 'claim' field found in benchmark data or data is empty.")
        exit(1)
    num_queries = len(all_nlqs)
    print(f"Loaded {num_queries} queries for benchmark.")


    evaluator = FeverousEvaluation(n_values=EVALUATION_N_VALUES)
    all_evaluation_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {} # {level: {retriever: {eval_level: df}}}


    # --- Loop through Index Levels ---
    for index_level in INDEX_LEVELS_TO_RUN:
        print(f"\n{'='*20} PROCESSING INDEX LEVEL: {index_level.upper()} {'='*20}")
        all_evaluation_results[index_level] = {}

        # --- Determine Input Data for this Level ---
        input_jsonl_filename = SERIALIZATION_FILENAMES.get(index_level)
        if not input_jsonl_filename:
            print(f"Error: Invalid index level '{index_level}'. Skipping.")
            continue
        input_jsonl_path = DATA_DIR / input_jsonl_filename

        print(f"Using input data: {input_jsonl_path}")

        # --- Initialize Retrievers for this level ---
        print("\n--- Initializing Retrievers ---")
        retrievers_for_level: List[BaseRetriever] = []
        retriever_instances = {
            #"BM25": lambda: PyseriniBM25Retriever(),
            #"Dense": lambda: FaissDenseRetriever(model_name_or_path=EMBEDDING_MODEL_NAME),
            #"Dense+Rerank": lambda: DenseRetrieverWithReranker(embedding_model_name=EMBEDDING_MODEL_NAME, reranker_model_name=RERANKER_MODEL_NAME),
            "Dense+Decomp": lambda: DenseRetrieverWithDecomposition(embedding_model_name=EMBEDDING_MODEL_NAME, model_name=OLLAMA_MODEL, decomposition_cache_folder=str(DECOMP_CACHE_DIR)),
            #"Dense+Decomp+Rerank": lambda: DenseRetrieverWithDecompositionAndReranker(embedding_model_name=EMBEDDING_MODEL_NAME, reranker_model_name=RERANKER_MODEL_NAME, ollama_model=OLLAMA_MODEL, decomposition_cache_folder=str(DECOMP_CACHE_DIR)),
            #"ReAct": lambda: ReActRetriever(dense_model_name_or_path=EMBEDDING_MODEL_NAME, model_path=REACT_LLM_MODEL_PATH)
        }

        for name, init_func in retriever_instances.items():
            try:
                retriever_instance = init_func()
                retrievers_for_level.append(retriever_instance)
                print(f"Successfully initialized {name}")
            except Exception as e:
                print(f"Could not initialize {name}: {e}")

        if not retrievers_for_level:
            print(f"No retrievers initialized for level {index_level}. Skipping level.")
            continue

        # --- Loop through Retrievers for Indexing, Retrieval, and Evaluation ---
        for retriever in retrievers_for_level:
            retriever_name = type(retriever).__name__
            print(f"\n--- Processing: {retriever_name} (Level: {index_level}) ---")
            run_config = {
                "retriever": retriever_name,
                "index_level": index_level,
                "k_results": K_RESULTS,
                "evaluation_n_values": EVALUATION_N_VALUES,
                "embedding_model": EMBEDDING_MODEL_NAME if isinstance(retriever, (FaissDenseRetriever, DenseRetrieverWithReranker, DenseRetrieverWithDecomposition, DenseRetrieverWithDecompositionAndReranker, ReActRetriever)) else None, # ReAct uses dense model
                "reranker_model": RERANKER_MODEL_NAME if isinstance(retriever, (DenseRetrieverWithReranker, DenseRetrieverWithDecompositionAndReranker)) else None,
                "ollama_model": OLLAMA_MODEL if isinstance(retriever, (DenseRetrieverWithDecomposition, DenseRetrieverWithDecompositionAndReranker)) else None,
                "react_llm_model": REACT_LLM_MODEL_PATH if isinstance(retriever, ReActRetriever) else None, # Use the correct variable
                "input_data": str(input_jsonl_path.name),
                "benchmark_file": str(BENCHMARK_FILE_PATH.name),
            }
            run = wandb.init(
                project="feverous-retrieval-benchmark",
                entity="lakhs",
                group=f"level-{index_level}",
                name=f"run-{index_level}-{retriever_name}",
                config=run_config,
                reinit=True,
                job_type="evaluation"
            )
            try:
                # --- Determine Index Path ---
                # Pass EMBEDDING_MODEL_NAME for dense-based retrievers including ReAct
                output_folder = get_output_folder(INDEX_BASE_DIR, index_level, retriever, EMBEDDING_MODEL_NAME)

                # --- Indexing ---
                print(f"Indexing to: {output_folder}")
                retriever.index(
                    input_jsonl_path=str(input_jsonl_path),
                    output_folder=str(output_folder),
                    field_to_index=FIELD_TO_INDEX,
                    metadata_fields=METADATA_FIELDS_TO_INDEX
                )
                print("Indexing complete.")

                # --- Retrieval ---
                print(f"Retrieving top-{K_RESULTS} for {num_queries} queries...")
                start_time = time.time()
                retrieved_results: List[List[RetrievalResult]] = retriever.retrieve(
                    nlqs=all_nlqs,
                    output_folder=str(output_folder),
                    k=K_RESULTS # K_RESULTS is ignored by ReAct's retrieve logic, but passed for consistency
                )
                end_time = time.time()
                retrieval_time = end_time - start_time
                print(f"Retrieval completed in {retrieval_time:.4f} seconds.")

                # --- Log Retrieval Time ---
                wandb.log({"retrieval_time_seconds": retrieval_time})
                if num_queries > 0 and retrieval_time > 0:
                     wandb.log({"queries_per_second": num_queries / retrieval_time})

                # Calculate and log ReAct specific metrics BEFORE evaluation
                react_specific_metrics = {}
                if isinstance(retriever, ReActRetriever):
                    avg_distinct, avg_calls = retriever.display_metrics(verbose=False) # Get metrics without printing here
                    react_specific_metrics = {
                        "avg_distinct_retrieved_objects": avg_distinct if avg_distinct is not None else 0,
                        "avg_llm_search_calls": avg_calls if avg_calls is not None else 0
                    }
                    wandb.log(react_specific_metrics) # Log the scalar ReAct metrics


                # --- Evaluation ---
                print("Running FEVEROUS Evaluation...")
                if len(retrieved_results) != num_queries:
                    print(f"ERROR: Prediction length mismatch for {retriever_name} (Level: {index_level}). Skipping evaluation.")
                    wandb.log({"status": "error", "error_message": "Prediction length mismatch"})
                    all_evaluation_results[index_level][retriever_name] = {"error": "Prediction length mismatch"}
                    # Add react specific metrics if they exist, for final reporting
                    if react_specific_metrics:
                         all_evaluation_results[index_level][retriever_name]["react_specific_metrics"] = react_specific_metrics
                    # Go to finally block to finish wandb run
                    # No need for continue here, the try block will end
                else:
                    evaluation_output = evaluator.evaluate(
                        json_path=str(BENCHMARK_FILE_PATH),
                        predictions=retrieved_results
                    )
                    # Store raw output for final reporting if needed outside WandB
                    all_evaluation_results[index_level][retriever_name] = evaluation_output
                    # Add react specific metrics if they exist, for final reporting
                    if react_specific_metrics:
                         all_evaluation_results[index_level][retriever_name]["react_specific_metrics"] = react_specific_metrics

                    print(f"Evaluation complete for {retriever_name} (Level: {index_level}).")

                    # --- Log Evaluation Metrics (Conditional Logging) ---
                    if isinstance(retriever, ReActRetriever):
                        avg_distinct_retrieved = react_specific_metrics.get("avg_distinct_retrieved_objects")
                        avg_llm_calls = react_specific_metrics.get("avg_llm_search_calls")

                        if avg_distinct_retrieved is not None and avg_llm_calls is not None:
                            k_for_eval = max(1, int(round(avg_distinct_retrieved)))

                            gt_lists_map = {
                                "exact_sentence_table": evaluator.last_gt_exact,
                                "table_page": evaluator.last_gt_table_page,
                                "page": evaluator.last_gt_page,
                            }
                            pred_lists_map = {
                                "exact_sentence_table": evaluator.last_pred_exact,
                                "table_page": evaluator.last_pred_table_page,
                                "page": evaluator.last_pred_page,
                            }

                            for eval_level_name in ["exact_sentence_table", "table_page", "page"]:
                                current_gt_list = gt_lists_map.get(eval_level_name)
                                current_pred_list = pred_lists_map.get(eval_level_name)

                                if current_gt_list is not None and current_pred_list is not None:
                                    metrics_at_k_for_eval = evaluator.calculate_metrics_for_single_n(
                                        ground_truth_ids=current_gt_list,
                                        predicted_ids=current_pred_list,
                                        n_value=k_for_eval
                                    )

                                    react_custom_df_data = {
                                        "#calls ↓": [avg_llm_calls],
                                        "Avg #obj. ↓": [avg_distinct_retrieved],
                                        "P": [metrics_at_k_for_eval.get("Precision", 0.0) * 100.0],
                                        "R": [metrics_at_k_for_eval.get("Recall", 0.0) * 100.0],
                                        "F1": [metrics_at_k_for_eval.get("F1", 0.0) * 100.0],
                                        "PR": [metrics_at_k_for_eval.get("Perfect Recall", 0.0) * 100.0],
                                    }
                                    react_custom_df = pd.DataFrame(react_custom_df_data)
                                    column_order = ["#calls ↓", "Avg #obj. ↓", "P", "R", "F1", "PR"]
                                    react_custom_df = react_custom_df[column_order]

                                    wandb.log({
                                        f"react_summary_table/{eval_level_name}": wandb.Table(dataframe=react_custom_df)
                                    })

                                    wandb.log({
                                        f"react_{eval_level_name}_avg_calls": avg_llm_calls,
                                        f"react_{eval_level_name}_avg_obj": avg_distinct_retrieved,
                                        f"react_{eval_level_name}_P_at_avg_obj": metrics_at_k_for_eval.get("Precision", 0.0) * 100.0,
                                        f"react_{eval_level_name}_R_at_avg_obj": metrics_at_k_for_eval.get("Recall", 0.0) * 100.0,
                                        f"react_{eval_level_name}_F1_at_avg_obj": metrics_at_k_for_eval.get("F1", 0.0) * 100.0,
                                        f"react_{eval_level_name}_PR_at_avg_obj": metrics_at_k_for_eval.get("Perfect Recall", 0.0) * 100.0,
                                    })
                                    
                    else:
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
                                     if n_val != K_RESULTS and n_val in df.index: # Avoid double logging if K_RESULTS is in EVALUATION_N_VALUES
                                        summary_metrics[f"eval_{eval_level_name}_Precision@{n_val}"] = df.loc[n_val, 'Precision']
                                        summary_metrics[f"eval_{eval_level_name}_Recall@{n_val}"] = df.loc[n_val, 'Recall']
                                        summary_metrics[f"eval_{eval_level_name}_F1@{n_val}"] = df.loc[n_val, 'F1']
                                        summary_metrics[f"eval_{eval_level_name}_PerfectRecall@{n_val}"] = df.loc[n_val, 'Perfect Recall']


                        wandb.log(metrics_to_log)
                        if summary_metrics:
                            wandb.log(summary_metrics) # Log scalar summary metrics

                    # --- Common Logging for all Retrievers ---
                    wandb.log({"status": "completed"})

            except Exception as e:
                print(f"\n--- ERROR processing {retriever_name} at level {index_level} ---")
                error_msg = f"{type(e).__name__}: {e}"
                print(error_msg)
                print("Traceback:")
                tb_str = traceback.format_exc()
                print(tb_str)
                print("----------------------------------------------------------")
                all_evaluation_results[index_level][retriever_name] = {"error": error_msg}
                 # Log React specific metrics even on error if available and calculated
                if react_specific_metrics:
                     all_evaluation_results[index_level][retriever_name]["react_specific_metrics"] = react_specific_metrics

                if run: # Check if wandb.init was successful
                    wandb.log({"status": "error", "error_message": error_msg, "traceback": tb_str[:2000]}) # Log truncated traceback
            finally:
                # --- Finish WANDB Run ---
                if run: # Check if wandb.init was successful
                    wandb.finish()
                    run = None # Reset run variable
                    print(f"Wandb run finished for {retriever_name} (Level: {index_level}).")

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