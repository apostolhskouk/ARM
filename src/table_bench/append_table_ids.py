import json

def augment_json_with_table_ids_to_file(input_jsonl_file_path, output_jsonl_file_path):
    table_signature_to_id_map = {}
    current_table_id_counter = 1
    
    with open(input_jsonl_file_path, 'r') as infile, open(output_jsonl_file_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            
            table_info = record.get("table")
            
            if table_info:
                columns = tuple(table_info.get("columns", []))
                data_content = table_info.get("data", [])
                
                immutable_data_rows = tuple(tuple(str(cell) for cell in row) for row in data_content)
                
                table_signature = (columns, immutable_data_rows)

                if table_signature not in table_signature_to_id_map:
                    assigned_id = f"table_{current_table_id_counter}"
                    table_signature_to_id_map[table_signature] = assigned_id
                    current_table_id_counter += 1
                
                record["table_id"] = table_signature_to_id_map[table_signature]
            else:
                record["table_id"] = None 
            
            outfile.write(json.dumps(record) + '\n')
    
    print(f"Augmented JSONL file saved to: {output_jsonl_file_path}")
    print(f"Total unique tables found: {current_table_id_counter - 1}")

if __name__ == '__main__':
    input_file_path = 'assets/table_bench/TableBench.jsonl'
    output_file_path = 'assets/table_bench/TableBench_new.jsonl'
    augment_json_with_table_ids_to_file(input_file_path, output_file_path)