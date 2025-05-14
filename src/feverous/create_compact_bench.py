import json
import os
import sys
from collections import defaultdict # Might not be needed now, but harmless

print(f"Python version: {sys.version}")

# --- Configuration ---
benchmark_input_path = 'assets/feverous/benchmark.json' # Input file
compact_output_path = 'assets/feverous/benchmark_compact.json' # Output file
error_log_path = 'assets/feverous/benchmark_compact_warnings.log' # Log for issues

# --- Warning Log ---
warnings_log = []
def log_warning(message):
    print(f"Warning: {message}") 
    warnings_log.append(message)

# --- Load Benchmark Data ---
print(f"Loading benchmark data from: {benchmark_input_path}...")
try:
    with open(benchmark_input_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    print(f"Loaded {len(benchmark_data)} benchmark records.")
except FileNotFoundError:
    log_warning(f"Input file not found at {benchmark_input_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    log_warning(f"Error decoding JSON from {benchmark_input_path}: {e}")
    sys.exit(1)
except Exception as e:
    log_warning(f"An unexpected error occurred loading {benchmark_input_path}: {e}")
    sys.exit(1)

# --- Process Data and Create Compact Format ---
benchmark_compact_data = []
processed_count = 0

print("Processing records to create compact format with aggregated ground truth...")
for record in benchmark_data:
    processed_count += 1
    try:
        claim_text = record.get("claim")
        claim_id = record.get("id", "N/A") # Get ID for logging
        evidence_ids = record.get("evidence_ids", []) # Default to empty list

        if claim_text is None:
            log_warning(f"Record missing 'claim' field (ID: {claim_id}). Skipping.")
            continue
        if not isinstance(evidence_ids, list):
             log_warning(f"'evidence_ids' is not a list for claim ID {claim_id}. Treating as empty.")
             evidence_ids = []

        page_titles_set = set()
        ground_truth_tables_set = set() # Store unique table IDs like "table_0"
        has_sentence_evidence = False   # Flag if any sentence is needed

        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str) or '_' not in evidence_id:
                log_warning(f"Invalid evidence ID format: '{evidence_id}' for claim ID {claim_id}. Skipping this ID.")
                continue

            # Split only on the first underscore
            parts = evidence_id.split('_', 1)
            if len(parts) != 2:
                 log_warning(f"Could not split evidence ID '{evidence_id}' correctly for claim ID {claim_id}. Skipping this ID.")
                 continue

            page_title = parts[0]
            element_part = parts[1] # e.g., "sentence_0", "cell_0_4_1", "table_caption_1"

            page_titles_set.add(page_title)

            # --- Aggregate Ground Truth ---
            if element_part.startswith("sentence_"):
                has_sentence_evidence = True
            elif element_part.startswith("cell_") or element_part.startswith("header_cell_"):
                # Format: (header_)?cell_[TABLE_INDEX]_[ROW]_[COL]
                element_parts = element_part.split('_')
                if len(element_parts) >= 3:
                    table_index = element_parts[-3] # Table index is third from the end
                    if table_index.isdigit():
                         table_id = f"table_{table_index}"
                         ground_truth_tables_set.add(table_id)
                    else:
                        log_warning(f"Could not extract numeric table index from cell ID '{element_part}' in '{evidence_id}' for claim ID {claim_id}.")
                else:
                     log_warning(f"Unexpected cell ID format '{element_part}' in '{evidence_id}' for claim ID {claim_id}.")
            elif element_part.startswith("table_caption_"):
                # Format: table_caption_[TABLE_INDEX]
                element_parts = element_part.split('_')
                if len(element_parts) == 3:
                    table_index = element_parts[2]
                    if table_index.isdigit():
                        table_id = f"table_{table_index}"
                        ground_truth_tables_set.add(table_id)
                    else:
                        log_warning(f"Could not extract numeric table index from caption ID '{element_part}' in '{evidence_id}' for claim ID {claim_id}.")
                else:
                     log_warning(f"Unexpected table caption ID format '{element_part}' in '{evidence_id}' for claim ID {claim_id}.")
            elif element_part.startswith("table_"):
                 # Could be table_X directly (less common but possible)
                 # Ensure it's not something else like table_caption
                 if "_caption_" not in element_part:
                      element_parts = element_part.split('_')
                      if len(element_parts) == 2 and element_parts[0] == "table" and element_parts[1].isdigit():
                          ground_truth_tables_set.add(element_part) # Add "table_X" directly
                      # else: # Ignore if format is unexpected, already handled by cell/caption checks
                      #    log_warning(f"Potentially ambiguous table ID format '{element_part}' in '{evidence_id}' for claim ID {claim_id}.")


            # Ignore 'item_' prefixes (list items) as per requirement

        # --- Consolidate final ground truth list ---
        final_ground_truth_elements = sorted(list(ground_truth_tables_set)) # Start with sorted table IDs
        if has_sentence_evidence:
            final_ground_truth_elements.append("sentences") # Add generic sentence marker if needed

        # Convert set of titles to a sorted list
        page_titles_list = sorted(list(page_titles_set))

        # Only create record if there's some ground truth identified
        if final_ground_truth_elements:
            compact_record = {
                "question": claim_text,
                "page_titles": page_titles_list,
                "ground_truth_elements": final_ground_truth_elements
            }
            benchmark_compact_data.append(compact_record)
        # else: # Optionally log claims with no identifiable ground truth after filtering
        #     log_warning(f"Claim ID {claim_id} resulted in empty ground truth after filtering evidence: {evidence_ids}")


    except Exception as e:
        log_warning(f"Unexpected error processing record (ID: {claim_id}): {e}")

print(f"\nProcessed {processed_count} records.")

# --- Write Compact Benchmark File ---
print(f"Writing compact benchmark data to: {compact_output_path}...")
try:
    with open(compact_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(benchmark_compact_data, outfile, indent=2, ensure_ascii=False)
    print(f"Successfully wrote {len(benchmark_compact_data)} records to {compact_output_path}")
except IOError as e:
    log_warning(f"Error writing compact benchmark file: {e}")
    sys.exit(1)
except Exception as e:
    log_warning(f"An unexpected error occurred while writing compact benchmark file: {e}")
    sys.exit(1)

# --- Write Warnings Log ---
if warnings_log:
    print(f"\n{len(warnings_log)} warnings were generated during compacting. Writing log to {error_log_path}")
    # Ensure directory exists for log file if putting it in the same place as output
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    try:
        with open(error_log_path, 'w', encoding='utf-8') as f:
            for warning in warnings_log:
                f.write(warning + '\n')
    except IOError as e:
        print(f"Error writing warnings log: {e}")
else:
    print("\nNo warnings generated during compacting.")

print("\nScript finished successfully!")