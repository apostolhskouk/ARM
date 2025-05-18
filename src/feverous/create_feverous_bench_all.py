import json
import os


INPUT_JSON_PATH = "assets/feverous/benchmark.json"
OUTPUT_JSON_PATH = "assets/all_data/benchmarks/feverous.json"


def extract_table_index(id_part_str, prefix_str):
    if id_part_str.startswith(prefix_str):
        stripped_part = id_part_str[len(prefix_str):]
        index_parts = stripped_part.split('_')
        if index_parts and len(index_parts[0]) > 0 and index_parts[0].isdigit():
            return index_parts[0]
    return None

def process_single_evidence_id(evidence_id_str):
    parts = evidence_id_str.split('_', 1)
    if len(parts) < 2:
        return None
    
    page_title = parts[0]
    id_suffix = parts[1]
    
    source_identifier = None
    
    if id_suffix.startswith("sentence_"):
        source_identifier = id_suffix
    elif id_suffix.startswith("cell_"):
        table_idx = extract_table_index(id_suffix, "cell_")
        if table_idx is not None:
            source_identifier = f"table_{table_idx}"
    elif id_suffix.startswith("header_cell_"):
        table_idx = extract_table_index(id_suffix, "header_cell_")
        if table_idx is not None:
            source_identifier = f"table_{table_idx}"
    elif id_suffix.startswith("table_caption_"):
        table_idx = extract_table_index(id_suffix, "table_caption_")
        if table_idx is not None:
            source_identifier = f"table_{table_idx}"
    elif id_suffix.startswith("table_") : 
        source_identifier = id_suffix
        
    if source_identifier:
        return f"{page_title}_{source_identifier}"
    return None

def main():
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f_in:
        input_data = json.load(f_in)
    
    output_data_list = []
    
    for record in input_data:
        query_text = record.get("claim", "")
        evidence_ids_list = record.get("evidence_ids", [])
        
        processed_document_ids = set()
        for ev_id in evidence_ids_list:
            doc_id = process_single_evidence_id(ev_id)
            if doc_id:
                processed_document_ids.add(doc_id)
        if not processed_document_ids:
            continue
        output_record = {
            "query": query_text,
            "document_ids": sorted(list(processed_document_ids))
        }
        output_data_list.append(output_record)
        
    output_dir = os.path.dirname(OUTPUT_JSON_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(output_data_list, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()