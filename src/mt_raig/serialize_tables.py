import json
import os
def serialize_tables_by_row(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    all_rows_serialized = []

    for table in tables_data:
        table_id = table.get("id", "")
        title = table.get("title", "")
        header_list = table.get("header", [])
        cells = table.get("cell", [])

        header_str_for_representation = ", ".join(map(str, header_list))

        for row_cell_values in cells:
            
            row_key_value_pairs = []
            for h, v in zip(header_list, row_cell_values):
                row_key_value_pairs.append(f"{str(h)} is {str(v)}")
            
            row_content_str = ", ".join(row_key_value_pairs)
            
            text_representation = f"Title: {str(title)}. Headers: {header_str_for_representation}. Row: {row_content_str}"
            
            serialized_row_object = {
                "table_id": table_id,
                "title": title,
                "header": header_list, 
                "row_values": row_cell_values,
                "row_representation": text_representation
            }
            all_rows_serialized.append(serialized_row_object)
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_rows_serialized, f, indent=2, ensure_ascii=False)

input_file_path = "assets/mt-raig/table_corpus.json" 
output_file_path = "assets/mt-raig/serialized_output/serialized_tables_by_row.json"

serialize_tables_by_row(input_file_path, output_file_path)