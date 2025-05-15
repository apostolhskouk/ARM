import json
from chonky import ParagraphSplitter
from chonky.markup_remover import MarkupRemover
import os
from tqdm import tqdm
input_file_path = "assets/MultiHopRag/corpus.json"
output_file_path = "assets/MultiHopRag/corpus_chunked.json"

original_text_field_name = "body"
chunk_output_field_name = "chunked_text"

remover = MarkupRemover()
splitter = ParagraphSplitter(
  model_id="mirth/chonky_modernbert_base_1",
  device="cuda:1"
)

output_records = []

with open(input_file_path, 'r', encoding='utf-8') as f_in:
    data = json.load(f_in)

for record in tqdm(data, desc="Processing records", unit="record"):
    title = record.get("title", "")
    category = record.get("category", "")
    source = record.get("source", "")
    
    text_to_chunk = record.get(original_text_field_name, "")
    
    if not text_to_chunk:
        continue
        
    plain_text = remover(text_to_chunk)
    
    chunks = splitter(plain_text)
    
    for chunk_text in chunks:
        formatted_chunk_text = f"Title: {title}\n\n{chunk_text}"
        
        new_chunk_record = {
            "title": title,
            "category": category,
            "source": source,
            chunk_output_field_name: formatted_chunk_text
        }
        output_records.append(new_chunk_record)

output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(output_records, f_out, indent=4, ensure_ascii=False)