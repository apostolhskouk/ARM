import guidance
import re
from collections import defaultdict

# --- 1. Set up the Environment (Using a Mock LLM for demonstration) ---
# In a real scenario, replace with:
# guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo") # Or your preferred LLM
# Or guidance.llm = guidance.llms.Transformers("mistralai/Mistral-7B-Instruct-v0.1")
guidance.llm = guidance.llms.Mock() # Use Mock for predictable testing initially

# --- 2. Define Dummy Data ---
# List of valid n-grams (strings with 1-3 space-separated words/tokens)
# We add some plausible options related to the example.
dummy_ngrams = [
    "name", "full name",
    "jesus college", "college",
    "alumni", "former", "university", "student",
    "graduated", "degree", "educated", "postgraduate", "study",
    "1960", "year 1960"
]

# --- 3. Preprocess N-grams for Prefix Lookup ---
# We'll create a dictionary mapping a prefix tuple to a list of valid next words.
# Using tuples as keys because lists aren't hashable.
# NOTE: This uses simple space splitting. Real tokenization is more complex
#       and depends on the specific LLM's tokenizer. Guidance typically
#       handles tokenization internally, but generating constraints based on
#       strings requires careful handling of spaces/word boundaries.
prefix_lookup = defaultdict(list)
tokenized_ngrams = []

print("Tokenizing N-grams (using space splitting):")
for ngram in dummy_ngrams:
    # Simple space split - might need adjustment for real tokenizers
    tokens = ngram.split(' ')
    # Add leading space to non-first tokens for guidance matching (often needed)
    processed_tokens = [tokens[0]] + [" " + t for t in tokens[1:]]
    tokenized_ngrams.append(processed_tokens)
    print(f"  '{ngram}' -> {processed_tokens}")

    # Build prefix lookup
    for i in range(len(processed_tokens)):
        prefix = tuple(processed_tokens[:i])
        next_token = processed_tokens[i]
        if next_token not in prefix_lookup[prefix]:
            prefix_lookup[prefix].append(next_token)

print("\nPrefix Lookup Structure (Prefix Tuple -> List of Next Tokens):")
for k, v in prefix_lookup.items():
    print(f"  {k}: {v}")
print("-" * 20)


# --- 4. Define the Guidance Program ---

@guidance(stateless=True) # Use stateless=True if feasible for better caching/reuse
def guided_ngram_extraction(lm, user_question, lookup):
    # Few-shot prompt structure
    lm += """You are given a user question, your task is to decompose the user question into contiguous, non-overlapping substrings that can cover different information mentioned in the user question. For each substring, generate n-grams that are the most relevant to the substring.
Example:
User question: what is the full name of the jesus college alumni who graduated in 1960 ?
The relevant keywords are full name | jesus college | alumni | graduated | 1960
The relevant n-grams are full name ( name) | jesus college ( jesus college) | alumni ( alumni, former, university) | graduated (degree, educated, postgraduate) | 1960 ( 1960)

User question: """ + user_question + "\nThe relevant keywords are"

    generated_keywords_text = "" # Keep track for context

    # Loop to extract Keyword | N-grams | ... structure
    while True:
        # Generate the keyword part - stop before '(' or newline
        # We add '|' as a stop char because the model might generate it early
        lm += guidance.gen(name='keyword_part', stop=['(', '\n', '|'], max_tokens=30)
        generated_keywords_text += lm['keyword_part']

        # Check if we stopped because of '(' indicating n-gram alignment start
        # Need to check the *actual* end of the generated text
        # `lm.endswith` might be better if available/reliable in context
        raw_keyword_stop = lm.get('keyword_part', '') # Get the raw generated string
        stop_char = None
        if raw_keyword_stop.endswith('('):
            stop_char = '('
            # Remove trailing '(' captured by gen if stop includes it
            lm = lm[:-1]
            generated_keywords_text = generated_keywords_text[:-1]
        elif raw_keyword_stop.endswith('|'):
             stop_char = '|'
             # Keep the '|' as it's part of the structure
        elif raw_keyword_stop.endswith('\n'):
             stop_char = '\n'

        # Decide if we should continue based on stop character
        if stop_char == '(':
            lm += "(" # Add the opening parenthesis explicitly

            # --- Constrained N-gram Generation within () ---
            current_ngram_prefix_tokens = []
            first_token_in_ngram = True

            while True: # Loop for tokens within a single (...) block
                prefix_key = tuple(current_ngram_prefix_tokens)
                valid_next_tokens = lookup.get(prefix_key, [])

                # Prepare options for `select`
                options = list(valid_next_tokens) # Copy the list

                # Add ',' and ')' as valid options *after* the first token of an n-gram
                # or if the current prefix itself forms a complete n-gram (optional check)
                can_add_delimiters = not first_token_in_ngram
                # More robust: check if current_prefix is a complete tokenized_ngram?
                # prefix_is_complete = any(prefix_key == tuple(tn) for tn in tokenized_ngrams)
                # can_add_delimiters = can_add_delimiters or prefix_is_complete

                if can_add_delimiters:
                    if ")" not in options: options.append(")")
                    if "," not in options: options.append(",")
                    # Add space before comma? Usually handled by select/gen
                    # Let's assume select handles token boundaries correctly


                # If no valid continuations found for the prefix, only allow delimiters
                if not valid_next_tokens and can_add_delimiters:
                     # Only allow delimiters if we are mid-ngram
                     options = [opt for opt in [")", ","] if opt in options] # Keep existing if present
                     if not options: options = [")", ","] # Force if empty
                elif not valid_next_tokens and first_token_in_ngram:
                     # Cannot start any known n-gram, maybe allow just closing? Or error?
                     # Forcing ')' might be safest if the dummy list is comprehensive.
                     options = [")"]


                # Ensure options are not empty
                if not options:
                    print(f"Warning: No valid options found for prefix {prefix_key}. Forcing ')'.")
                    options = [")"] # Fallback

                # Use select to force the choice
                # Need unique option strings if duplicates exist
                unique_options = sorted(list(set(options)))
                lm += guidance.select(unique_options, name='selected_token')

                selected = lm['selected_token']

                if selected == ")":
                    break # Exit the inner n-gram loop
                elif selected == ",":
                    # Reset prefix for the next n-gram within the same ()
                    current_ngram_prefix_tokens = []
                    first_token_in_ngram = True
                    lm += " " # Add space after comma before next n-gram/token
                else:
                    # Append the selected token to the current prefix
                    # Assumes 'selected' is a single token/word as processed earlier
                    current_ngram_prefix_tokens.append(selected)
                    first_token_in_ngram = False
                    # Check if current n-gram exceeds max length (3 tokens) - Optional
                    if len(current_ngram_prefix_tokens) >= 3:
                         # If we hit max length, the only valid next tokens should be delimiters
                         # This logic is implicitly handled if the lookup table is built correctly
                         pass

            # --- End Constrained N-gram Generation ---

        elif stop_char == '|':
             # We stopped at '|', just continue the main loop
             pass
        elif stop_char == '\n' or stop_char is None:
             # Stopped at newline or end of generation for the keyword part
             # Heuristic: Assume generation is finished
             # A more robust check might involve checking if the last generated
             # content looks like a complete structure.
             print(f"\nStopping generation loop due to stop_char: {stop_char}")
             break # Exit the main keyword extraction loop

        # If we finished a () block or just a keyword without (), expect " | " or newline
        # Generate the separator, expecting " | " or potentially just end of line
        # Allow generating the separator only if needed
        # Check if the last character suggests we need a separator
        last_char = str(lm)[-1]
        if last_char == ')': # Need separator after closing paren
             lm += guidance.gen(name='separator', max_tokens=3, regex=r" \| |\n") # Expect space-pipe-space or newline
             generated_keywords_text += lm['separator']
             if lm['separator'] == '\n':
                  break # End loop if newline generated after ngrams
        elif last_char == '|': # Already have pipe, maybe expect space? Or newline?
            pass # Already handled
        elif stop_char is None and last_char != '\n': # Keyword ended without '(', expect separator or newline
            lm += guidance.gen(name='separator', max_tokens=3, regex=r" \| |\n")
            generated_keywords_text += lm['separator']
            if lm['separator'] == '\n':
                  break


        # Safety break if loop runs too long (optional)
        # if len(generated_keywords_text) > 500: break

    return lm

# --- Example Execution ---
user_input = "what is the full name of the jesus college alumni who graduated in 1960 ?"

# Build the lookup table from the dummy n-grams
prefix_lookup = defaultdict(list)
tokenized_ngrams = []
print("\n--- Building Final Prefix Lookup ---")
for ngram in dummy_ngrams:
    tokens = ngram.split(' ')
    processed_tokens = [tokens[0]] + [" " + t for t in tokens[1:]]
    tokenized_ngrams.append(processed_tokens)
    print(f"  Processing: {processed_tokens}")
    for i in range(len(processed_tokens)):
        prefix = tuple(processed_tokens[:i])
        next_token = processed_tokens[i]
        if next_token not in prefix_lookup[prefix]:
            prefix_lookup[prefix].append(next_token)

print("Lookup built:")
# print(prefix_lookup) # Can be verbose

# Execute the guidance program
# Note: Mock LLM will likely just repeat parts of the prompt or follow simple patterns.
# A real LLM is needed for meaningful output based on the instructions.
# We can preload the mock LLM with expected intermediate outputs for testing.

# Example Mock setup for testing the constrained part:
mock_keyword_part = "full name " # Simulate LLM generating keyword and stopping before '('
mock_ngram_choice_1 = " name"     # Simulate LLM choosing ' name' from options [' name', ' full name']
mock_ngram_choice_2 = ","         # Simulate LLM choosing ','
mock_ngram_choice_3 = " alumni"   # Simulate LLM choosing ' alumni'
mock_ngram_choice_4 = ")"         # Simulate LLM choosing ')'
mock_separator = " | "

# This setup is complex for Mock. Let's run it and see what the Mock LLM does.
# It will likely fail without a more sophisticated mock setup or a real LLM.
# For now, we focus on the guidance program structure.

print("\n--- Running Guidance Program (with Mock LLM) ---")
try:
    # For Mock LLM, we often need to guide it more explicitly or it returns empty/prompt
    # Let's try running it as is first.
    program_output = guided_ngram_extraction(
        user_question=user_input,
        lookup=prefix_lookup,
        # llm=guidance.llms.OpenAI("gpt-3.5-turbo") # Swap for real LLM
    )
    print("\n--- Program Output ---")
    print(program_output)

except Exception as e:
    print(f"\n--- Error during Guidance execution ---")
    print(e)
    print("NOTE: Mock LLM execution is limited. Testing with a real LLM (OpenAI, Transformers) is necessary.")
    print("The code defines the structure; its behavior depends heavily on the LLM.")