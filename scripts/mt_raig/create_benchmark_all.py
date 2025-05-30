import json
import os

INPUT_BENCHMARK_PATH = "assets/mt-raig/benchmark.json"
OUTPUT_BENCHMARK_PATH = "assets/all_data/benchmarks/mt_raig.json"
SERIALIZED_DATA_LOOKUP_PATH = "assets/all_data/serialized_data/mt_raig.jsonl" 

def build_source_to_page_title_map(lookup_file_path):
    source_map = {}
    if not os.path.exists(lookup_file_path):
        print(f"Warning: Lookup file not found at {lookup_file_path}. Document IDs might be incomplete.")
        return source_map
        
    with open(lookup_file_path, 'r', encoding='utf-8') as f_lookup:
        for line in f_lookup:
            try:
                record = json.loads(line)
                source = record.get("source")
                page_title = record.get("page_title")
                if source and page_title:
                    source_map[source] = page_title
            except json.JSONDecodeError:
                pass 
    return source_map

def main():
    source_to_page_title = build_source_to_page_title_map(SERIALIZED_DATA_LOOKUP_PATH)
    
    with open(INPUT_BENCHMARK_PATH, 'r', encoding='utf-8') as f_in:
        input_data = json.load(f_in)
        
    output_data_list = []
    
    for record in input_data:
        query_text = record.get("question", "")
        gold_table_ids = record.get("gold_table_id_set", [])
        
        document_ids_for_record = []
        for table_id in gold_table_ids:
            lookup_key = f"table_{table_id}"
            page_title = source_to_page_title.get(lookup_key)
            
            if page_title is not None:
                document_id = f"{page_title}_{lookup_key}" # lookup_key is already "table_{table_id}"
                document_ids_for_record.append(document_id)
            else:
                # Optionally handle cases where page_title is not found for a source
                # For now, we just skip adding it
                pass

        output_record = {
            "query": query_text,
            "document_ids": sorted(list(set(document_ids_for_record))) # Ensure uniqueness and sort
        }
        output_data_list.append(output_record)
        
    output_dir = os.path.dirname(OUTPUT_BENCHMARK_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_BENCHMARK_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(output_data_list, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()