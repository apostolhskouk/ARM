import json
import os

#INPUT_JSON_PATH = "assets/target/benchmarks/bird-validation_benchmark.json"
#OUTPUT_JSON_PATH = "assets/all_data/benchmarks/bird.json"


#INPUT_JSON_PATH = "assets/target/benchmarks/fetaqa_benchmark.json"
#OUTPUT_JSON_PATH = "assets/all_data/benchmarks/fetaqa.json"

#INPUT_JSON_PATH = "assets/target/benchmarks/ottqa_benchmark.json"
#OUTPUT_JSON_PATH = "assets/all_data/benchmarks/ottqa.json"


#INPUT_JSON_PATH = "assets/target/benchmarks/spider-test_benchmark.json"
#OUTPUT_JSON_PATH = "assets/all_data/benchmarks/spider.json"

#INPUT_JSON_PATH = "assets/target/benchmarks/tabfact_benchmark.json"
#OUTPUT_JSON_PATH = "assets/all_data/benchmarks/tabfact.json"

ALL_INPUT_JSON_PATHS = [
    "assets/target/benchmarks/bird-validation_benchmark.json",
    "assets/target/benchmarks/fetaqa_benchmark.json",
    "assets/target/benchmarks/ottqa_benchmark.json",
    "assets/target/benchmarks/spider-test_benchmark.json",
    "assets/target/benchmarks/tabfact_benchmark.json"
]

ALL_OUTPUT_JSON_PATHS = [
    "assets/all_data/benchmarks/bird.json",
    "assets/all_data/benchmarks/fetaqa.json",
    "assets/all_data/benchmarks/ottqa.json",
    "assets/all_data/benchmarks/spider.json",
    "assets/all_data/benchmarks/tabfact.json"
]

def main():
    for INPUT_JSON_PATH, OUTPUT_JSON_PATH in zip(ALL_INPUT_JSON_PATHS, ALL_OUTPUT_JSON_PATHS):
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f_in:
            input_data = json.load(f_in)
            
        output_data_list = []
        
        for record in input_data:
            query_text = record.get("query_text", "")
            relevant_docs_list = record.get("relevant_docs", [])
            
            document_ids_for_record = []
            for doc_info in relevant_docs_list:
                database_id = doc_info.get("database_id", "")
                table_id = doc_info.get("table_id", "")
                
                if database_id and table_id: # Ensure both parts are present
                    document_id_string = f"{database_id}_table_{table_id}"
                    document_ids_for_record.append(document_id_string)
            
            output_record = {
                "query": query_text,
                "document_ids": document_ids_for_record
            }
            output_data_list.append(output_record)
            
        output_dir = os.path.dirname(OUTPUT_JSON_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f_out:
            json.dump(output_data_list, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()