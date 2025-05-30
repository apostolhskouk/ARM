import json
from chonky import ParagraphSplitter
from chonky.markup_remover import MarkupRemover
import os
from tqdm import tqdm
input_file_path = "assets/MultiHopRag/corpus.json"
output_file_path = "assets/all_data/serialized_data/multi_hop_rag.jsonl"

original_text_field_name = "body"
chunk_output_field_name = "chunked_text"

remover = MarkupRemover()
splitter = ParagraphSplitter(
  model_id="mirth/chonky_modernbert_base_1",
  device="cuda:1"
)

output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir): # Ensure output_dir is not empty
    os.makedirs(output_dir)

with open(input_file_path, 'r', encoding='utf-8') as f_in, \
     open(output_file_path, 'w', encoding='utf-8') as f_out:
    
    data = json.load(f_in)

    for record in tqdm(data, desc="Processing records", unit="record"):
        input_record_title = record.get("title", "")
        # category = record.get("category", "") # Not used in the new output format
        input_record_source = record.get("source", "")
        
        text_to_chunk = record.get(original_text_field_name, "")
        
        if not text_to_chunk:
            continue
            
        plain_text = remover(text_to_chunk)
        
        chunks = splitter(plain_text)
        
        for chunk_text in chunks:
            # Construct the 'object' field
            object_field_value = f"{input_record_title} [SEP] {chunk_text}"
            
            # Assign values for 'page_title' and 'source' based on requirements
            page_title_val = input_record_source # page_title uses the original "source"
            source_val = input_record_title      # source uses the original "title"
            
            new_output_record = {
                "page_title": page_title_val,
                "source": f"sentence_{source_val}",
                "object": object_field_value
            }
            f_out.write(json.dumps(new_output_record, ensure_ascii=False) + '\n')