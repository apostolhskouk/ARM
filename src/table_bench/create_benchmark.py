import json

def create_benchmark_json(input_augmented_jsonl_path, output_benchmark_json_path):
    with open(input_augmented_jsonl_path, 'r') as infile, open(output_benchmark_json_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            
            benchmark_record = {
                "qtype": record.get("qtype"),
                "qsubtype": record.get("qsubtype"),
                "question": record.get("question"),
                "table_id": record.get("table_id")
            }
            
            outfile.write(json.dumps(benchmark_record) + '\n')

if __name__ == '__main__':
    input_file_path = 'assets/table_bench/TableBench_with_id.jsonl'
    output_file_path = 'assets/table_bench/benchmark.json'
    create_benchmark_json(input_file_path, output_file_path)