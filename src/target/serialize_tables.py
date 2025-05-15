import json
import os
from pathlib import Path
output_dir = Path("assets/target")
output_dir.mkdir(parents=True, exist_ok=True)

raw_data_dir = Path("assets/target_raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(raw_data_dir.resolve())
os.environ['HF_DATASETS_CACHE'] = str((raw_data_dir.resolve() / "datasets"))
os.environ['HF_HUB_CACHE'] = str((raw_data_dir.resolve() / "hub"))

from target_benchmark.dataset_loaders import (
    HFDatasetLoader,
    Text2SQLDatasetLoader,
    NeedleInHaystackDataLoader,
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
    DATABASE_ID_COL_NAME,
    TABLE_COL_NAME,
    TABLE_ID_COL_NAME,
    CONTEXT_COL_NAME,
)

def main():



    ALL_DATASET_CONFIGS = {}
    ALL_DATASET_CONFIGS.update(QUESTION_ANSWERING_DATASETS)
    ALL_DATASET_CONFIGS.update(FACT_VER_DATASETS)
    ALL_DATASET_CONFIGS.update(TEXT_2_SQL_DATASETS)
    ALL_DATASET_CONFIGS.update(NEEDLE_IN_HAYSTACK_DATASETS)
    ALL_DATASET_CONFIGS[DEFAULT_INFAGENTDA_DATASET_CONFIG.dataset_name] = DEFAULT_INFAGENTDA_DATASET_CONFIG

    for dataset_name_key, config_model in ALL_DATASET_CONFIGS.items():
        print(f"Processing dataset: {dataset_name_key}...")
        
        loader = None
        config_data_dict = config_model.model_dump()

        # Pass cache_dir to loaders if they support it, otherwise rely on HF_HOME
        # For HFDatasetLoader and its children, load_dataset respects HF_HOME/HF_DATASETS_CACHE
        # For Text2SQLDatasetLoader's snapshot_download, it respects HF_HOME/HF_HUB_CACHE
        
        if isinstance(config_model, Text2SQLDatasetConfigDataModel):
            # Text2SQLDatasetLoader's snapshot_download will use HF_HOME
            loader = Text2SQLDatasetLoader(**config_data_dict)
        elif isinstance(config_model, NeedleInHaystackDatasetConfigDataModel):
            loader = NeedleInHaystackDataLoader(**config_data_dict)
        elif isinstance(config_model, HFDatasetConfigDataModel):
            loader = HFDatasetLoader(**config_data_dict)
        else:
            print(f"Skipping {dataset_name_key}: Unknown or unsupported config model type {type(config_model)}")
            continue
        
        try:
            loader.load()
        except Exception as e:
            print(f"Error loading dataset {dataset_name_key}: {e}")
            print(f"Skipping {dataset_name_key}.")
            print("-" * 30)
            continue


        all_serialized_records_for_dataset = []
        
        # NeedleInHaystackDataLoader might not have queries, but we are processing corpus here.
        # It should still allow corpus iteration.
        if loader.corpus is None and not isinstance(loader, NeedleInHaystackDataLoader): # NIH might have no queries but should have corpus
             print(f"  Corpus for {dataset_name_key} is None. Skipping corpus processing.")
        else:
            for batch in loader.convert_corpus_table_to(output_format="nested array", batch_size=1):
                db_id = batch[DATABASE_ID_COL_NAME][0]
                original_table_id = batch[TABLE_ID_COL_NAME][0]
                table_content = batch[TABLE_COL_NAME][0]
                context = batch[CONTEXT_COL_NAME][0]

                if not table_content or not table_content[0]:
                    print(f"  Skipping table (empty or no headers): db_id='{db_id}', table_id='{original_table_id}'")
                    continue

                headers = table_content[0]
                data_rows = table_content[1:]

                table_display_name = None
                if context and isinstance(context, dict):
                    table_display_name = context.get('table_page_title') or \
                                         context.get('section_title') or \
                                         context.get('title') # Adding generic title from context
                
                # For Text2SQL, original_table_id is often the table name
                if not table_display_name and isinstance(config_model, Text2SQLDatasetConfigDataModel):
                    table_display_name = str(original_table_id)


                for row_values in data_rows:
                    if len(row_values) != len(headers):
                        print(f"  Warning: Mismatch in columns for table_id='{original_table_id}', db_id='{db_id}'. Headers: {len(headers)}, Row: {len(row_values)}. Skipping row.")
                        continue

                    serialized_parts = []
                    for j in range(len(headers)):
                        serialized_parts.append(f"{str(headers[j])}: {str(row_values[j])}")
                    
                    current_row_col_value_pairs = ", ".join(serialized_parts)

                    if table_display_name:
                        final_serialized_row_string = f"Table: {table_display_name}, {current_row_col_value_pairs}"
                    else:
                        final_serialized_row_string = current_row_col_value_pairs
                    
                    record = {
                        "database_id": str(db_id),
                        "table_id": str(original_table_id), 
                        "serialized_row": final_serialized_row_string,
                    }
                    
                    # Optionally, if you want to keep a separate metadata field for the derived table name
                    if table_display_name:
                        record["table_name_from_context"] = table_display_name

                    all_serialized_records_for_dataset.append(record)
        
        output_file_path = output_dir / f"{dataset_name_key}_serialized.json"
        with open(output_file_path, "w") as f_out:
            json.dump(all_serialized_records_for_dataset, f_out, indent=2)
        
        print(f"Finished processing {dataset_name_key}. Output saved to {output_file_path}")
        print("-" * 30)

if __name__ == "__main__":
    main()