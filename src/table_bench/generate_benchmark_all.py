import json
import os

INPUT_JSONL_PATH = "assets/table_bench/TableBench_with_id.jsonl"
OUTPUT_JSON_PATH = "assets/all_data/benchmarks/table_bench.json"

def main():
    output_data_list = []
    
    with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            record = json.loads(line)
            
            query_text = record.get("question", "")
            qtype_val = record.get("qtype", "")
            qsubtype_val = record.get("qsubtype", "")
            table_id_val = record.get("table_id", "")
            
            document_id_string = f"{qtype_val}{qsubtype_val}_{table_id_val}"
            
            output_record = {
                "query": query_text,
                "document_ids": [document_id_string]
            }
            output_data_list.append(output_record)
            
    output_dir = os.path.dirname(OUTPUT_JSON_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(output_data_list, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()