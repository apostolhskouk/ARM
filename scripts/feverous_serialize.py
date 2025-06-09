import json
import os
from tqdm import tqdm  # For progress bar

from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
from feverous.utils.wiki_sentence import WikiSentence
from feverous.utils.wiki_table import WikiTable

BENCHMARK_FILE_PATH = "assets/feverous/benchmark.json"
DATABASE_PATH = "assets/feverous/feverous_wikiv1.db"  
OUTPUT_DIR = "assets/feverous/serialized_output" 

OUTPUT_TABLE_LEVEL_FILE = os.path.join(OUTPUT_DIR, "serialized_table_level.jsonl")
OUTPUT_ROW_LEVEL_FILE = os.path.join(OUTPUT_DIR, "serialized_row_level.jsonl")
OUTPUT_CELL_LEVEL_FILE = os.path.join(OUTPUT_DIR, "serialized_cell_level.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP = " [SEP] " 
ROW_SEP = " , " 
TABLE_ROW_CONCAT_SEP = " ; " 

def safe_get_cell_text(cells, index, default=""):
    """Safely get cell text by index, handling out-of-bounds."""
    try:
        return str(cells[index])
    except IndexError:
        return default

def safe_get_header_text(headers, index, default=""):
    """Safely get header text by index."""
    try:
        return headers[index]
    except IndexError:
        return default
  
# 1. Read target page titles
print(f"Reading target page titles from: {BENCHMARK_FILE_PATH}")
with open(BENCHMARK_FILE_PATH, 'r', encoding='utf-8') as f:
    benchmark_data = json.load(f)

all_page_titles = set()
for record in benchmark_data:
    if "page_titles" in record and isinstance(record["page_titles"], list):
        for title in record["page_titles"]:
            if isinstance(title, str):
                all_page_titles.add(title)

unique_page_titles = sorted(list(all_page_titles))
print(f"Found {len(unique_page_titles)} unique page titles to process.")

# 2. Initialize Database Connection
print(f"Connecting to database: {DATABASE_PATH}")
db = FeverousDB(DATABASE_PATH)


# 3. Process Pages and Serialize
# Using file handles to write line by line (memory efficient for large outputs)
with open(OUTPUT_TABLE_LEVEL_FILE, 'w', encoding='utf-8') as f_table, \
     open(OUTPUT_ROW_LEVEL_FILE, 'w', encoding='utf-8') as f_row, \
     open(OUTPUT_CELL_LEVEL_FILE, 'w', encoding='utf-8') as f_cell:

    for page_title in tqdm(unique_page_titles, desc="Processing Pages"):
        page_json = db.get_doc_json(page_title)
        if not page_json:
            # print(f"Warning: Page '{page_title}' not found in database. Skipping.")
            continue

        wiki_page = WikiPage(page_title, page_json)

        for element_id in wiki_page.page_order:
            element = wiki_page.get_element_by_id(element_id)
            source_id = element.get_id() # e.g., "sentence_5", "table_1"

            # --- Process Sentences ---
            if isinstance(element, WikiSentence):
                sentence_text = str(element)
                serialized_object = f"{page_title}{SEP}{sentence_text}"
                output_record = {
                    "object": serialized_object,
                    "page_title": page_title,
                    "source": source_id
                }
                record_json_str = json.dumps(output_record, ensure_ascii=False)
                # Sentences are the same for all representations
                f_table.write(record_json_str + "\n")
                f_row.write(record_json_str + "\n")
                f_cell.write(record_json_str + "\n")

            # --- Process Tables ---
            elif isinstance(element, WikiTable):
                table_id = source_id
                rows = element.get_rows()
                if not rows: continue # Skip empty tables

                # Try to get caption (heuristic: preceding sentence)
                caption = None
                try:
                    prev_elem = wiki_page.get_previous_k_elements(table_id, k=1)
                    if prev_elem and isinstance(prev_elem[0], WikiSentence):
                        caption = str(prev_elem[0])
                except (IndexError, KeyError):
                    pass # No previous element or error getting it

                caption_prefix = f"{caption}{SEP}" if caption else ""
                base_prefix = f"{page_title}{SEP}{caption_prefix}" # Base prefix for all table serializations

                has_header = rows[0].is_header_row()
                headers = []
                if has_header:
                    headers = [str(cell) for cell in rows[0].get_row_cells()]

                table_level_row_strings = []
                start_row_index = 1 if has_header else 0

                for i, row_object in enumerate(rows):
                    cells = row_object.get_row_cells()
                    cell_texts = [str(cell) for cell in cells]

                    # --- Cell Level Serialization ---
                    for j, cell_object in enumerate(cells):
                        cell_text = str(cell_object)
                        header_text = ""
                        # Include header text only if header exists AND it's not the header row itself
                        if has_header and i > 0:
                            header_text = safe_get_header_text(headers, j) + SEP

                        cell_level_object = f"{base_prefix}{header_text}{cell_text}"
                        cell_record = {
                            "object": cell_level_object.strip(),
                            "page_title": page_title,
                            "source": table_id
                        }
                        f_cell.write(json.dumps(cell_record, ensure_ascii=False) + "\n")

                    # --- Row Level Serialization (Skip header row content itself) ---
                    if has_header and i == 0:
                        continue # Skip header row for row/table level output

                    row_level_object_parts = []
                    if has_header: # Format as "Header: Value"
                        for j, cell_text in enumerate(cell_texts):
                            header_val = safe_get_header_text(headers, j)
                            row_level_object_parts.append(f"{header_val}: {cell_text}")
                        row_content_str = ROW_SEP.join(row_level_object_parts)
                    else: # Just concatenate cell values
                        row_content_str = ROW_SEP.join(cell_texts)

                    row_level_object = f"{base_prefix}{row_content_str}"
                    row_record = {
                        "object": row_level_object.strip(),
                        "page_title": page_title,
                        "source": table_id
                    }
                    f_row.write(json.dumps(row_record, ensure_ascii=False) + "\n")

                    # Collect for table-level serialization
                    table_level_row_strings.append(row_content_str)


                # --- Table Level Serialization ---
                # (Only if there were non-header rows or if there was no header)
                if table_level_row_strings:
                    table_content_str = TABLE_ROW_CONCAT_SEP.join(table_level_row_strings)
                    table_level_object = f"{base_prefix}{table_content_str}"
                    table_record = {
                        "object": table_level_object.strip(),
                        "page_title": page_title,
                        "source": table_id
                    }
                    f_table.write(json.dumps(table_record, ensure_ascii=False) + "\n")

print("\nSerialization Complete.")
print(f"Table-level JSONL saved to: {OUTPUT_TABLE_LEVEL_FILE}")
print(f"Row-level JSONL saved to: {OUTPUT_ROW_LEVEL_FILE}")
print(f"Cell-level JSONL saved to: {OUTPUT_CELL_LEVEL_FILE}")