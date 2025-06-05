import abc
import json
import time
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import wandb
from src.utils.evaluator import EvaluationMetrics
from src.retrieval.dense import FaissDenseRetriever
from src.retrieval.base import BaseRetriever, RetrievalResult

# --- Constants ---
BENCHMARK_DIR = Path("assets/all_data/benchmarks_subsampled")
BENCHMARK_FILENAMES = [
    "bird.json", "fetaqa.json", "feverous.json", "mt_raig.json",
    "multi_hop_rag.json", "ottqa.json", "spider.json", "tabfact.json",
    "table_bench.json"
]

EMBEDDING_MODELS_CONFIG = {
  "Alibaba-NLP/gte-modernbert-base": "dense_alibaba_gte_modernbert",
  "Alibaba-NLP/gte-multilingual-base": "dense_alibaba_gte_multilingual",
  "Snowflake/snowflake-arctic-embed-l-v2.0": "dense_arctic_embed_l_v2",
  "Snowflake/snowflake-arctic-embed-s": "dense_artic_embed_s",
  "BAAI/bge-m3": "dense_bge_m3",
  "intfloat/multilingual-e5-large-instruct": "dense_e5_large_instruct",
  "infly/inf-retriever-v1-1.5b": "dense_infly_v1_1.5b",
  "jinaai/jina-embeddings-v3": "dense_jina_v3",
  "nomic-ai/nomic-embed-text-v2-moe": "dense_nomic_embed_text_v2_moe",
  "nomic-ai/modernbert-embed-base": "dense_nomic_modernbert",
  "WhereIsAI/UAE-Large-V1": "dense_uae_large_v1"
}
INDEXES_BASE_DIR_TEMPLATE = "assets/all_data/indexes/{embedding_folder_key}/"

EVALUATION_N_VALUES = [1, 3, 5, 10]
RETRIEVAL_K = 9000
WANDB_PROJECT_NAME = "all_benchmarks_multi_embedding_dense"
WANDB_ENTITY_NAME = "lakhs" 
ACCURACY_METRIC_KEYS = ['Precision', 'Recall', 'F1', 'Perfect Recall']

# --- Helper Functions ---
def get_benchmark_file_stem(benchmark_filename: str) -> str:
    """Extracts the stem (filename without extension) from a benchmark filename."""
    return Path(benchmark_filename).stem

def get_embedding_model_short_name(model_name_or_path: str) -> str:
    """Generates a short name for an embedding model from its full name or path."""
    return model_name_or_path.split('/')[-1]

def get_dense_index_folder(embedding_folder_key: str, benchmark_stem: str) -> Path:
    """Constructs the path to the pre-built dense index for a given model and benchmark."""
    return Path(INDEXES_BASE_DIR_TEMPLATE.format(embedding_folder_key=embedding_folder_key)) / benchmark_stem

def load_benchmark_data(benchmark_filepath: Path) -> Tuple[List[str], List[List[str]]]:
    """Loads queries and ground truth document IDs from a benchmark JSON file."""
    with open(benchmark_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    queries = [item["query"] for item in data]
    ground_truth_ids_list = [item["document_ids"] for item in data]
    return queries, ground_truth_ids_list

# --- Main Evaluation Script ---
def main():
    """Main function to run evaluations for all embedding models and datasets."""
    evaluator = EvaluationMetrics(n_values=EVALUATION_N_VALUES)
    all_runs_data_for_summary = []

    for model_full_name, embedding_folder_key in EMBEDDING_MODELS_CONFIG.items():
        model_short_name = get_embedding_model_short_name(model_full_name)
        print(f"Processing Embedding Model: {model_full_name}")

        retriever_instance = FaissDenseRetriever(
            model_name_or_path=model_full_name,
            )

        for benchmark_filename in BENCHMARK_FILENAMES:
            benchmark_stem = get_benchmark_file_stem(benchmark_filename)
            benchmark_filepath = BENCHMARK_DIR / benchmark_filename
            print(f"  Evaluating on Benchmark: {benchmark_stem}")

            run_name = f"Dense-{model_short_name}-{benchmark_stem}"
            group_name = f"Dense-{model_short_name}"
            
            config_dict = {
                "retriever_type": "FaissDenseRetriever",
                "embedding_model_full_name": model_full_name,
                "embedding_model_short_name": model_short_name,
                "embedding_folder_key": embedding_folder_key,
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

            index_folder = get_dense_index_folder(embedding_folder_key, benchmark_stem)
            queries, ground_truth_ids_list = load_benchmark_data(benchmark_filepath)
            num_queries = len(queries)

            start_time = time.time()
            retrieval_k = RETRIEVAL_K
            if benchmark_filename == "bird.json":
                retrieval_k = retrieval_k * 10
            retrieved_results_nested_list: List[List[RetrievalResult]] = retriever_instance.retrieve(
                nlqs=queries,
                output_folder=str(index_folder),
                k=retrieval_k
            )
            end_time = time.time()
            retrieval_time = end_time - start_time
            qps = num_queries / retrieval_time if retrieval_time > 0 else 0
            
            wandb.log({
                "retrieval_time_seconds": retrieval_time,
                "queries_per_second": qps
            })

            predicted_doc_ids_list = [
                [f"{res.metadata['page_title']}_{res.metadata['source']}" for res in inner_list]
                for inner_list in retrieved_results_nested_list
            ]

            results_df = evaluator.calculate_metrics(
                ground_truth_ids=ground_truth_ids_list,
                predicted_ids=predicted_doc_ids_list
            )
            wandb.log({f"evaluation_table/{benchmark_stem}": wandb.Table(dataframe=results_df.reset_index())})

            current_run_summary_metrics = {
                "embedding_model_full_name": model_full_name,
                "embedding_model_short_name": model_short_name,
                "dataset": benchmark_stem,
                "qps": qps
            }

            for n_val in results_df.index:
                for metric_key_from_df_column in results_df.columns:
                    if metric_key_from_df_column in ACCURACY_METRIC_KEYS:
                        metric_value = results_df.loc[n_val, metric_key_from_df_column]
                        wandb_metric_name = f"eval_{benchmark_stem}_{metric_key_from_df_column.replace(' ', '')}@{n_val}"
                        wandb.log({wandb_metric_name: metric_value})
                        
                        summary_metric_col_name = f"{metric_key_from_df_column.replace(' ', '')}@{n_val}"
                        current_run_summary_metrics[summary_metric_col_name] = metric_value
            
            all_runs_data_for_summary.append(current_run_summary_metrics)
            
            wandb.log({"status": "completed"})
            wandb.finish()
            run = None 
        
        del retriever_instance
        gc.collect()
        print(f"Finished all benchmarks for model: {model_full_name}\n")

    print("All individual evaluations complete. Generating summary WandB run...")
    
    summary_run = wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        name="All_Embeddings_Dense_Overall_Summary",
        job_type="summary_analysis",
        reinit=True
    )

    summary_df = pd.DataFrame(all_runs_data_for_summary)

    qps_pivot_table = summary_df.pivot_table(
        index="dataset", columns="embedding_model_short_name", values="qps"
    )
    summary_run.log({"QPS_by_dataset_and_model": wandb.Table(dataframe=qps_pivot_table.reset_index())})

    mean_qps_per_model = summary_df.groupby("embedding_model_short_name")["qps"].mean().reset_index()
    mean_qps_per_model = mean_qps_per_model.sort_values(by="qps", ascending=False)
    summary_run.log({"Mean_QPS_per_model": wandb.Table(dataframe=mean_qps_per_model)})

    for n_val_for_summary in EVALUATION_N_VALUES:
        for acc_metric_key_raw in ACCURACY_METRIC_KEYS:
            acc_metric_key_clean = acc_metric_key_raw.replace(' ', '')
            summary_df_metric_column = f"{acc_metric_key_clean}@{n_val_for_summary}"

            metric_pivot_table = summary_df.pivot_table(
                index="dataset", columns="embedding_model_short_name", values=summary_df_metric_column
            )
            summary_run.log({f"{summary_df_metric_column}_by_dataset_and_model": wandb.Table(dataframe=metric_pivot_table.reset_index())})

            mean_metric_per_model = summary_df.groupby("embedding_model_short_name")[summary_df_metric_column].mean().reset_index()
            mean_metric_per_model = mean_metric_per_model.sort_values(by=summary_df_metric_column, ascending=False)
            summary_run.log({f"Mean_{summary_df_metric_column}_per_model": wandb.Table(dataframe=mean_metric_per_model)})

    summary_run.log({"status": "completed"})
    summary_run.finish()
    print("WandB summary run complete.")

if __name__ == "__main__":
    main()