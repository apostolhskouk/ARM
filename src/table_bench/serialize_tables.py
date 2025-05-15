import json

def serialize_tables_to_rows(input_augmented_jsonl_path, output_serialized_jsonl_path):
    processed_table_ids = set()
    
    with open(input_augmented_jsonl_path, 'r') as infile, open(output_serialized_jsonl_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            
            table_id = record.get("table_id")
            table_info = record.get("table")

            if table_id and table_info and table_id not in processed_table_ids:
                processed_table_ids.add(table_id)
                
                columns = table_info.get("columns", [])
                data_rows = table_info.get("data", [])
                
                if not columns:
                    continue

                for data_row_values in data_rows:
                    if len(data_row_values) == len(columns):
                        row_kv_pairs = []
                        for i in range(len(columns)):
                            column_name = columns[i]
                            cell_value = data_row_values[i]
                            row_kv_pairs.append(f"{column_name}: {cell_value}")
                        
                        serialized_row_string = ", ".join(row_kv_pairs)
                        
                        output_json_record = {
                            "table_id": table_id,
                            "row_serialization": serialized_row_string
                        }
                        outfile.write(json.dumps(output_json_record) + '\n')

if __name__ == '__main__':
    input_file_path = 'assets/table_bench/TableBench_with_id.jsonl'
    output_file_path = 'assets/table_bench/serialized_tables.jsonl'
    serialize_tables_to_rows(input_file_path, output_file_path)