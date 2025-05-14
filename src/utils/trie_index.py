import json
import argparse 
import marisa_trie
import os
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simple_tokenize(text):
    """Simple tokenizer that splits on words/punctuation, removes [SEP], and lowercases."""
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    # Explicitly remove the '[SEP]' token
    tokens = [token for token in tokens if token != '[SEP]']
    # Lowercase and remove empty/whitespace-only tokens
    return [token.lower() for token in tokens if token.strip()]

def generate_ngrams(tokens, min_n, max_n):
    """Generates n-grams from a list of tokens."""
    num_tokens = len(tokens)
    for n in range(min_n, max_n + 1):
        for i in range(num_tokens - n + 1):
            yield tuple(tokens[i:i+n]) # Yield tuple for hashability

def build_marisa_trie_index(input_jsonl_path, output_trie_path, min_n=1, max_n=3):
    """
    Builds a MARISA-Trie index from n-grams extracted from a JSON Lines file.

    Args:
        input_jsonl_path (str): Path to the input JSON Lines file (.jsonl). Will be based on the 'object' field.
        output_trie_path (str): Path where the MARISA-Trie file will be saved.
        min_n (int): Minimum n-gram size.
        max_n (int): Maximum n-gram size.
    """
    logging.info(f"Starting Trie index build from '{input_jsonl_path}'")
    logging.info(f"N-gram range: {min_n} to {max_n}")

    unique_ngram_strings = set()
    processed_lines = 0
    errors = 0
    total_lines = 0 # For tqdm progress bar

    # --- Phase 1: Collect unique n-gram strings (Reading JSONL) ---
    logging.info("Phase 1: Reading JSONL and collecting unique n-grams...")
    try:
        # Optional: Count lines first for a better progress bar estimate
        logging.info("Counting lines in input file...")
        try:
            with open(input_jsonl_path, 'r', encoding='utf-8') as f_count:
                total_lines = sum(1 for _ in f_count)
            logging.info(f"Found {total_lines} lines.")
        except FileNotFoundError:
             logging.error(f"Input file not found during line count: '{input_jsonl_path}'")
             return # Exit if file not found even for counting


        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            # Iterate over each line in the JSONL file
            for line in tqdm(f, total=total_lines if total_lines > 0 else None, desc="Processing JSONL lines"):
                processed_lines += 1
                line = line.strip() # Remove leading/trailing whitespace
                if not line: # Skip empty lines
                    continue

                try:
                    # Parse the single line as a JSON object
                    data = json.loads(line)

                    # Extract the 'object' field
                    serialization_text = data.get("object")

                    if serialization_text and isinstance(serialization_text, str):
                        tokens = simple_tokenize(serialization_text)
                        if tokens:
                            for ngram_tuple in generate_ngrams(tokens, min_n, max_n):
                                ngram_string = " ".join(ngram_tuple)
                                unique_ngram_strings.add(ngram_string)
                    # else: # Optionally log records without valid 'object' text
                    #     logging.debug(f"Skipping line {processed_lines}: No valid 'object' string found or not a string.")

                except json.JSONDecodeError:
                    errors += 1
                    logging.warning(f"Skipping line {processed_lines}: Invalid JSON format.")
                # Keep catching other errors during the processing of a single line
                except Exception as e:
                    errors += 1
                    logging.error(f"Error processing line {processed_lines}: {e}")

    except FileNotFoundError:
        logging.error(f"Input file not found: '{input_jsonl_path}'")
        return
    except Exception as e:
        # Catch potential errors during file opening itself (e.g., permissions)
        logging.error(f"An unexpected error occurred during file reading: {e}")
        return

    logging.info(f"Finished reading file. Processed {processed_lines} lines.")
    if errors > 0:
        logging.warning(f"Encountered {errors} errors during JSON parsing or processing.")

    num_unique_ngrams = len(unique_ngram_strings)
    if num_unique_ngrams == 0:
        logging.warning("No n-grams were extracted. Trie will be empty.")
        # Depending on requirements, you might want to return here or still save an empty Trie
        # return

    logging.info(f"Found {num_unique_ngrams} unique n-grams.")

    # --- Phase 2: Build and Save MARISA-Trie ---
    logging.info("Phase 2: Building MARISA-Trie...")
    try:
        # Convert set to list for Trie construction (order doesn't matter for MARISA-Trie)
        ngram_list = list(unique_ngram_strings)

        # Build the Trie (uses unicode strings directly)
        trie = marisa_trie.Trie(ngram_list)
        logging.info("MARISA-Trie built successfully.")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_trie_path)
        if output_dir and not os.path.exists(output_dir):
            logging.info(f"Creating output directory: '{output_dir}'")
            os.makedirs(output_dir)

        # Save the Trie
        trie.save(output_trie_path)
        logging.info(f"Trie saved successfully to '{output_trie_path}'")

    except Exception as e:
        logging.error(f"An error occurred during Trie building or saving: {e}")

def find_continuations(prefix, trie_path):
    """
    Loads a MARISA-Trie and finds all keys that start with the given prefix.

    Args:
        prefix (str): The prefix string to search for.
        trie_path (str): Path to the saved MARISA-Trie file.

    Returns:
        list: A list of strings from the Trie that start with the prefix.
            Returns an empty list if the trie cannot be loaded or no matches are found.
    """
    try:
        trie = marisa_trie.Trie()
        trie.load(trie_path)
    except FileNotFoundError:
        logging.error(f"Trie file not found: '{trie_path}'")
        return []
    except Exception as e:
        logging.error(f"Error loading Trie index from '{trie_path}': {e}")
        return []

    search_prefix = prefix.lower()

    try:
        continuations = trie.keys(prefix=search_prefix)
        return continuations
    except Exception as e:
        logging.error(f"An error occurred during prefix search for '{search_prefix}': {e}")
        return []
    
def find_next_word_continuations(original_prefix, trie_path):
    """
    Loads a MARISA-Trie and finds both sub-token completions and
    the next distinct single tokens following the given prefix.

    - Sub-token completions are returned without a leading space.
    - Next complete tokens are returned *with* a leading space.

    Args:
        original_prefix (str): The prefix string. Can be empty.
        trie_path (str): Path to the saved MARISA-Trie file.

    Returns:
        list: A sorted list of unique completion strings.
              If the original_prefix is empty, returns all 1-grams (no spaces).
    """
    # --- 1. Load the Trie ---
    try:
        trie = marisa_trie.Trie()
        trie.load(trie_path)
    except FileNotFoundError:
        logging.error(f"Trie file not found: '{trie_path}'")
        return []
    except Exception as e:
        logging.error(f"Error loading Trie index from '{trie_path}': {e}")
        return []

    # --- 2. Prepare Search Prefix ---
    prefix_tokens = simple_tokenize(original_prefix)
    search_prefix = " ".join(prefix_tokens)

    results_set = set()

    # --- 3. Handle Empty Prefix Case (Find all 1-grams) ---
    if not search_prefix:
        try:
            all_keys = trie.keys()
            for key in all_keys:
                if ' ' not in key and key: # Ensure it's a single token and not empty
                    results_set.add(key) # Add 1-grams without leading space
        except Exception as e:
            logging.error(f"An error occurred while retrieving keys for 1-gram search: {e}")
            return []
        return sorted(list(results_set))

    # --- 4. Perform Main Prefix Search ---
    try:
        continuations = trie.keys(prefix=search_prefix)
    except Exception as e:
        logging.error(f"An error occurred during prefix search for '{search_prefix}': {e}")
        return []

    # --- 5. Process Continuations ---
    len_search_prefix = len(search_prefix)
    for cont in continuations:
        if len(cont) <= len_search_prefix: # Skip if continuation is same or shorter (shouldn't happen with prefix search)
             continue

        # Get the part *after* the search prefix
        remainder = cont[len_search_prefix:]

        if not remainder: # Skip if remainder is empty
            continue

        # Case 1: Remainder starts with a space -> It's a "next token"
        if remainder.startswith(' '):
            # Extract the *first* token after the space
            next_token = remainder[1:].split(' ', 1)[0]
            if next_token: # Ensure the extracted token is not empty
                results_set.add(" " + next_token) # Add with leading space

        # Case 2: Remainder does NOT start with space -> It's a sub-token completion
        else:
            # Ensure the remainder is just the completion, containing no spaces itself
            if ' ' not in remainder:
                results_set.add(remainder) # Add without leading space

    return sorted(list(results_set))

def get_distinct_tokens_from_trie(trie_path):
    """
    Loads a MARISA-Trie and extracts all unique individual tokens
    (words/punctuation) that constitute the stored n-grams.

    Args:
        trie_path (str): Path to the saved MARISA-Trie file.

    Returns:
        list: A sorted list of unique tokens found across all n-grams in the Trie.
              Returns an empty list if the trie cannot be loaded or is empty.
    """
    # --- 1. Load the Trie ---
    try:
        trie = marisa_trie.Trie()
        trie.load(trie_path)
    except FileNotFoundError:
        logging.error(f"Trie file not found: '{trie_path}'")
        return []
    except Exception as e:
        logging.error(f"Error loading Trie index from '{trie_path}': {e}")
        return []

    distinct_tokens = set()
    num_keys_processed = 0

    # --- 2. Iterate Through All Keys (N-grams) ---
    try:
        # Using items() might be slightly more memory efficient if values were stored,
        # but keys() is direct if only keys are needed.
        all_ngrams = trie.keys()
        total_keys = len(all_ngrams) # Get total count for tqdm if possible
    except Exception as e:
        # Catch potential errors during the key retrieval process itself
        logging.error(f"An error occurred while retrieving keys from the trie: {e}")
        return []


    # --- 3. Split N-grams and Collect Tokens ---
    # Use tqdm for progress visualization, especially for large tries
    for ngram_string in tqdm(all_ngrams, desc="Extracting Tokens", total=total_keys):
        # Split the space-separated n-gram string back into tokens
        # Note: ' '.split(' ') results in ['',''], we rely on simple_tokenize
        # having removed empty strings during build, so tokens should be valid.
        tokens_in_ngram = ngram_string.split(' ')
        # Add all tokens from this n-gram to the set
        distinct_tokens.update(tokens_in_ngram)
        num_keys_processed += 1

    # Final check log

    # --- 4. Return Sorted List ---
    return sorted(list(distinct_tokens))



if __name__ == "__main__":
    input_jsonl_path = "assets/feverous/serialized_output/serialized_table_level.jsonl"
    output_trie_path = "assets/feverous/trie_indexes/ngrams_table_level_1_3.marisa"
    # Define n-gram range
    min_ngram = 1
    max_ngram = 3

    # Call the function to build the index
    build_marisa_trie_index(input_jsonl_path, output_trie_path, min_ngram, max_ngram)
