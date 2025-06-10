import json
import os

def create_schema_file(input_path):
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_schema{ext}"

    processed_tables = set()

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            
            table_key = (data['page_title'], data['source'])

            if table_key not in processed_tables:
                page_title = data['page_title']
                source = data['source']
                table_name = source.replace('table_', '')
                
                object_parts = data['object'].split('[H]')
                columns = [part.split(':')[0].strip() for part in object_parts[1:]]
                
                column_string = " [H] ".join(columns)
                new_object_string = f"{page_title} [SEP] {table_name} [SEP] [H] {column_string}"

                output_record = {
                    "page_title": page_title,
                    "source": source,
                    "object": new_object_string
                }

                outfile.write(json.dumps(output_record) + '\n')
                processed_tables.add(table_key)

file_paths = [
    'assets/all_data/serialized_data/spider.jsonl',
    'assets/all_data/serialized_data/bird.jsonl'
]

for path in file_paths:
    if os.path.exists(path):
        create_schema_file(path)
        print(f"Processed {path} and created its schema file.")
    else:
        print(f"File not found: {path}")