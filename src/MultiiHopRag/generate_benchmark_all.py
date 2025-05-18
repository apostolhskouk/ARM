import json
import os

INPUT_BENCHMARK_PATH = "assets/MultiHopRag/MultiHopRAG.json"
OUTPUT_BENCHMARK_PATH = "assets/all_data/benchmarks/multi_hop_rag.json"

def main():
    with open(INPUT_BENCHMARK_PATH, 'r', encoding='utf-8') as f_in:
        input_data = json.load(f_in)
        
    output_data_list = []
    
    for record in input_data:
        query_text = record.get("query", "")
        evidence_list = record.get("evidence_list", [])
        
        document_ids_for_record = []
        for evidence_item in evidence_list:
            source = evidence_item.get("source", "")
            title = evidence_item.get("title", "")
            
            # Replace spaces and other problematic characters in title if needed for a cleaner ID
            # For now, direct concatenation as per "{source}_sentence_{title}"
            # If title can be very long or contain many special chars, a hashing or slugify approach might be better
            # but sticking to the literal interpretation of the format string.
            document_id = f"{source}_sentence_{title}"
            document_ids_for_record.append(document_id)

        output_record = {
            "query": query_text,
            "document_ids": document_ids_for_record # Duplicates are preserved if present in source data construction
        }
        output_data_list.append(output_record)
        
    output_dir = os.path.dirname(OUTPUT_BENCHMARK_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_BENCHMARK_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(output_data_list, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()