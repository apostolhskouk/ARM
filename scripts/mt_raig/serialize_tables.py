import json
import os
def serialize_tables_by_row(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        for table in tables_data:
            original_table_id = table.get("id", "")
            page_title_val = table.get("title", "") # This is the 'title' for page_title
            header_list = table.get("header", [])
            cells = table.get("cell", [])

            if not header_list: # Skip if no headers
                continue

            source_val = f"table_{original_table_id}"

            for row_cell_values in cells:
                if len(row_cell_values) != len(header_list):
                    continue 

                object_parts = []
                for h, v in zip(header_list, row_cell_values):
                    object_parts.append(f"[H] {str(h)}: {str(v)}")
                
                col_val_string = " , ".join(object_parts)
                
                object_field_value = f"{str(page_title_val)} [SEP] {col_val_string}"
                
                output_record = {
                    "page_title": page_title_val,
                    "source": source_val,
                    "object": object_field_value
                }
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')

input_file_path = "assets/mt-raig/table_corpus.json" 
output_file_path = "assets/mt-raig/serialized_output/serialized_tables_by_row.jsonl"

serialize_tables_by_row(input_file_path, output_file_path)
print(f"Serialized tables from {input_file_path} to {output_file_path}")