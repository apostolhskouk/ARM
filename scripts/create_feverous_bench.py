import json
import os
import glob
from tqdm import tqdm # For progress bars (optional, install with 'pip install tqdm')
import sys # To check Python version for f-string compatibility if needed

print(f"Python version: {sys.version}")


challenges_file_path = 'assets/feverous/feverous_dev_challenges.jsonl'
wiki_files_directory = 'assets/feverous/FeverousWikiv1'
benchmark_output_path = 'assets/feverous/benchmark.json'
dev_data_output_path = 'assets/feverous/dev_data.json'

benchmark_data = []
required_page_titles = set()

print(f"Processing challenges file: {challenges_file_path}...")

if not os.path.exists(challenges_file_path):
    print(f"Error: Challenges file not found at {challenges_file_path}")
    sys.exit(1)

line_count = 0
processed_claims = 0
skipped_lines = 0

with open(challenges_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="Reading challenges"):
        line_count += 1
        try:
            data = json.loads(line.strip())

            if not data.get('id') or 'claim' not in data or 'evidence' not in data:
                skipped_lines += 1
                continue

            claim_id = data['id']
            claim_text = data['claim']
            evidence_list = data.get('evidence', []) # Default to empty list

            all_evidence_ids_for_claim = []

            if isinstance(evidence_list, list):
                for evidence_set in evidence_list:
                    # Ensure evidence_set is a dictionary and has 'content'
                    if isinstance(evidence_set, dict) and 'content' in evidence_set:
                        content_ids = evidence_set.get('content', [])
                        if isinstance(content_ids, list):
                            all_evidence_ids_for_claim.extend(content_ids)
                            # Extract page titles from content IDs
                            for element_id in content_ids:
                                if isinstance(element_id, str) and '_' in element_id:
                                    # Split only on the first underscore to handle titles with underscores
                                    page_title = element_id.split('_', 1)[0]
                                    required_page_titles.add(page_title)
                                else:
                                     print(f"Warning: Invalid element ID format '{element_id}' in claim {claim_id} on line {line_count}")
                        else:
                            print(f"Warning: 'content' is not a list in evidence set for claim {claim_id} on line {line_count}")
                    else:
                         print(f"Warning: Invalid evidence set format for claim {claim_id} on line {line_count}: {evidence_set}")
            # Handle the case seen in the SUPPORTS example where 'evidence' might be a single dict (less likely based on raw data)
            elif isinstance(evidence_list, dict) and 'content' in evidence_list:
                 print(f"Note: Evidence for claim {claim_id} on line {line_count} is a single dictionary, not a list.")
                 content_ids = evidence_list.get('content', [])
                 if isinstance(content_ids, list):
                     all_evidence_ids_for_claim.extend(content_ids)
                     for element_id in content_ids:
                         if isinstance(element_id, str) and '_' in element_id:
                             page_title = element_id.split('_', 1)[0]
                             required_page_titles.add(page_title)
                         else:
                             print(f"Warning: Invalid element ID format '{element_id}' in claim {claim_id} on line {line_count}")
                 else:
                     print(f"Warning: 'content' is not a list in evidence dict for claim {claim_id} on line {line_count}")
            else:
                 print(f"Warning: Unexpected format for 'evidence' field for claim {claim_id} on line {line_count}: {type(evidence_list)}")


            # Add to benchmark data
            benchmark_data.append({
                "id": claim_id,
                "claim": claim_text,
                "evidence_ids": all_evidence_ids_for_claim
            })
            processed_claims += 1

        except json.JSONDecodeError:
            print(f"Skipping line {line_count} due to JSON decode error: {line.strip()[:100]}...")
            skipped_lines += 1
        except Exception as e:
            print(f"An unexpected error occurred processing line {line_count}: {e}")
            print(f"Line content: {line.strip()[:100]}...")
            skipped_lines += 1


print(f"\nFinished processing challenges file.")
print(f"Total lines read: {line_count}")
print(f"Processed claims: {processed_claims}")
print(f"Skipped lines: {skipped_lines}")
print(f"Identified {len(required_page_titles)} unique Wikipedia page titles required for evidence.")

# --- Step 2: Write benchmark.json ---
print(f"\nWriting benchmark data to: {benchmark_output_path}...")
try:
    with open(benchmark_output_path, 'w', encoding='utf-8') as outfile:
        # Use ensure_ascii=False to prevent escaping non-ASCII characters
        json.dump(benchmark_data, outfile, indent=2, ensure_ascii=False)
    print("Successfully wrote benchmark.json")
except IOError as e:
    print(f"Error writing benchmark file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while writing benchmark file: {e}")
    sys.exit(1)


# --- Step 3: Process Wikipedia Files and Extract Required Content ---

print(f"\nProcessing Wikipedia files from directory: {wiki_files_directory}...")

# Check if the wiki directory exists
if not os.path.isdir(wiki_files_directory):
    print(f"Error: Wikipedia directory not found at {wiki_files_directory}")
    sys.exit(1)

# Find all wiki JSONL files
wiki_files = glob.glob(os.path.join(wiki_files_directory, 'wiki_*.jsonl'))

if not wiki_files:
    print(f"Error: No 'wiki_*.jsonl' files found in {wiki_files_directory}")
    sys.exit(1)

print(f"Found {len(wiki_files)} wiki files to process.")

dev_wiki_data = {}
processed_pages_count = 0
matched_pages_count = 0
wiki_file_line_errors = 0

for wiki_file_path in tqdm(wiki_files, desc="Processing Wiki Files"):
    try:
        with open(wiki_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    wiki_page = json.loads(line.strip())
                    processed_pages_count += 1

                    page_title = wiki_page.get('title')

                    # Check if this page title is one we need
                    if page_title and page_title in required_page_titles:
                        dev_wiki_data[page_title] = wiki_page
                        matched_pages_count += 1

                except json.JSONDecodeError:
                    # print(f"Skipping line in {os.path.basename(wiki_file_path)} due to JSON decode error: {line.strip()[:100]}...")
                    wiki_file_line_errors += 1
                except Exception as e:
                     print(f"An unexpected error occurred processing line in {os.path.basename(wiki_file_path)}: {e}")
                     print(f"Line content: {line.strip()[:100]}...")
                     wiki_file_line_errors += 1

    except IOError as e:
        print(f"Error reading wiki file {wiki_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred processing wiki file {wiki_file_path}: {e}")


print(f"\nFinished processing Wikipedia files.")
print(f"Total Wikipedia pages processed: {processed_pages_count}")
print(f"Kept content for {len(dev_wiki_data)} required pages.")
if wiki_file_line_errors > 0:
    print(f"Encountered {wiki_file_line_errors} errors reading lines in wiki files.")

# --- Check for missing pages ---
found_pages = set(dev_wiki_data.keys())
missing_pages = required_page_titles - found_pages
if missing_pages:
    print(f"\nWarning: Could not find content for {len(missing_pages)} required page titles.")
    # You might want to log these missing titles to a file for investigation
    # print("Missing titles:", list(missing_pages)[:20]) # Print first few missing
    try:
        with open("missing_wiki_pages.log", "w", encoding='utf-8') as log_file:
             for title in sorted(list(missing_pages)):
                 log_file.write(title + '\n')
        print("List of missing titles saved to missing_wiki_pages.log")
    except IOError as e:
        print(f"Could not write missing pages log file: {e}")

# --- Step 4: Write dev_data.json ---
print(f"\nWriting development Wikipedia data to: {dev_data_output_path}...")
try:
    with open(dev_data_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(dev_wiki_data, outfile, indent=2, ensure_ascii=False)
    print("Successfully wrote dev_data.json")
except IOError as e:
    print(f"Error writing dev data file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while writing dev data file: {e}")
    sys.exit(1)

print("\nScript finished successfully!")