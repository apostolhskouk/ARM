import json
import os
from pathlib import Path
benchmark_output_dir = Path("assets/target/benchmarks")
benchmark_output_dir.mkdir(parents=True, exist_ok=True)

raw_data_dir = Path("assets/target_raw") 
raw_data_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(raw_data_dir.resolve())
os.environ['HF_DATASETS_CACHE'] = str((raw_data_dir.resolve() / "datasets"))
os.environ['HF_HUB_CACHE'] = str((raw_data_dir.resolve() / "hub"))
from target_benchmark.dataset_loaders import (
    HFDatasetLoader,
    Text2SQLDatasetLoader,
)
from target_benchmark.dataset_loaders.LoadersDataModels import (
    HFDatasetConfigDataModel,
    Text2SQLDatasetConfigDataModel,
    NeedleInHaystackDatasetConfigDataModel,
)
from target_benchmark.dataset_loaders.TargetDatasetConfig import (
    QUESTION_ANSWERING_DATASETS,
    FACT_VER_DATASETS,
    TEXT_2_SQL_DATASETS,
    NEEDLE_IN_HAYSTACK_DATASETS,
    DEFAULT_INFAGENTDA_DATASET_CONFIG,
)
from target_benchmark.dictionary_keys import (
    QUERY_ID_COL_NAME,
    QUERY_COL_NAME,
    DATABASE_ID_COL_NAME,
    TABLE_ID_COL_NAME,
)

def main():


    ALL_DATASET_CONFIGS = {}
    ALL_DATASET_CONFIGS.update(QUESTION_ANSWERING_DATASETS)
    ALL_DATASET_CONFIGS.update(FACT_VER_DATASETS)
    ALL_DATASET_CONFIGS.update(TEXT_2_SQL_DATASETS)
    ALL_DATASET_CONFIGS.update(NEEDLE_IN_HAYSTACK_DATASETS)
    ALL_DATASET_CONFIGS[DEFAULT_INFAGENTDA_DATASET_CONFIG.dataset_name] = DEFAULT_INFAGENTDA_DATASET_CONFIG

    for dataset_name_key, config_model in ALL_DATASET_CONFIGS.items():
        print(f"Processing queries for dataset: {dataset_name_key}...")

        if isinstance(config_model, NeedleInHaystackDatasetConfigDataModel) or \
           not hasattr(config_model, 'hf_queries_dataset_path') or \
           not config_model.hf_queries_dataset_path:
            print(f"  Dataset {dataset_name_key} has no query path or is NeedleInHaystack. Skipping benchmark creation.")
            print("-" * 30)
            continue

        loader = None
        config_data_dict = config_model.model_dump()

        if isinstance(config_model, Text2SQLDatasetConfigDataModel):
            loader = Text2SQLDatasetLoader(**config_data_dict)
        elif isinstance(config_model, HFDatasetConfigDataModel): # Catches HFDatasetConfigDataModel and its children not already caught
            loader = HFDatasetLoader(**config_data_dict)
        else:
            print(f"  Skipping {dataset_name_key}: Loader type not determined for benchmark creation.")
            print("-" * 30)
            continue
        
        loader.load() # This loads both corpus and queries if paths are set

        if loader.queries is None:
            print(f"  Queries for {dataset_name_key} could not be loaded. Skipping benchmark creation.")
            print("-" * 30)
            continue

        benchmark_entries = []
        
        # Iterate through each query in the loaded queries dataset
        # loader.get_queries() returns the Hugging Face Dataset object
        for query_data in loader.get_queries():
            query_id = query_data.get(QUERY_ID_COL_NAME)
            query_text = query_data.get(QUERY_COL_NAME)
            db_id_from_query = query_data.get(DATABASE_ID_COL_NAME)
            table_ids_from_query = query_data.get(TABLE_ID_COL_NAME)

            if query_id is None or query_text is None or db_id_from_query is None or table_ids_from_query is None:
                print(f"  Skipping a query in {dataset_name_key} due to missing essential fields. Query data: {query_data}")
                continue

            relevant_docs_list = []
            if isinstance(table_ids_from_query, str):
                relevant_docs_list.append({
                    "database_id": str(db_id_from_query),
                    "table_id": str(table_ids_from_query)
                })
            elif isinstance(table_ids_from_query, list):
                for tid in table_ids_from_query:
                    relevant_docs_list.append({
                        "database_id": str(db_id_from_query),
                        "table_id": str(tid)
                    })
            else:

                print(f"  Unexpected type for table_ids in {dataset_name_key}, query_id {query_id}: {type(table_ids_from_query)}. Setting empty relevant_docs.")


            benchmark_entry = {
                "query_id": str(query_id), # Ensure query_id is string
                "query_text": str(query_text),
                "relevant_docs": relevant_docs_list
            }
            benchmark_entries.append(benchmark_entry)
        
        benchmark_file_path = benchmark_output_dir / f"{dataset_name_key}_benchmark.json"
        with open(benchmark_file_path, "w") as f_out:
            json.dump(benchmark_entries, f_out, indent=2)
        
        print(f"Finished creating benchmark file for {dataset_name_key}. Output saved to {benchmark_file_path}")
        print("-" * 30)

if __name__ == "__main__":
    main()