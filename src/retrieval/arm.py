from src.utils.trie_index import find_next_word_continuations, get_distinct_tokens_from_trie
from src.retrieval.base import RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from typing import List
import guidance
from guidance import models, gen, select, capture
import re
from typing import List, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

"""trie_index_path = "assets/feverous/trie_indexes/ngrams_table_level_1_3.marisa"
prefix = "iver"
continuations = find_next_word_continuations(prefix, trie_index_path)
print(f"\n--- Next word continuations: {continuations}")
all_tokens = get_distinct_tokens_from_trie(trie_index_path)
print(f"\n--- Total distinct tokens in the Trie: {len(all_tokens)} ---")
bm25 = PyseriniBM25Retriever()
dense_index = FaissDenseRetriever()
retrieved_results_bm25 : List[List[RetrievalResult]] = []
retrieved_results_bm25 += bm25.retrieve(
    nlqs=["Formula E"],
    output_folder="assets/feverous/pyserini_indexes/bm25_row_index",
    k=5
)
retrieved_results_dense : List[List[RetrievalResult]] = []
retrieved_results_dense += dense_index.retrieve(
    nlqs=["Formula E"],
    output_folder="assets/feverous/faiss_indexes/dense_row_UAE-Large-V1",
    k=5
)

print("\n--- Retrieved Results with BM25 ---")
for i, result in enumerate(retrieved_results_bm25):
    for res in result:
        print(res.object)
print("\n\n--- Retrieved Results with Dense ---") 
for i, result in enumerate(retrieved_results_dense):
    for res in result:
        print(f"From {res.metadata.{res.object}")
"""




trie_index_path = "assets/feverous/trie_indexes/ngrams_table_level_1_3.marisa"
dense_index = FaissDenseRetriever()
all_tokens = get_distinct_tokens_from_trie(trie_index_path)


@guidance(stateless=False) # Needs to be stateful to use Python logic based on captures
def dynamic_retrieval_guidance(lm, user_query: str):

    # 1. Decompose query into keywords
    lm += f"""You are given a user question, your task is to decompose the user question into contiguous, non-overlapping substrings that can
cover different information mentioned in the user question. For each substring, generate n-grams that are the most relevant to
the substring. Based on the generated relevant n-grams, generate a list of relevant objects, including their names, content, and
connections between these objects. From these candidate objects, you should identify the minimum number of objects that can
be used to answer the user question based on the relevance between the object name, object content and user question as well as
the relevance of the object connections. You should end your response with <>.

User question: What is the birth date of the director of the movie 'Inception' which was released in 2010?
The relevant keywords are birth date | director | movie 'Inception' | released in 2010
The relevant n-grams are birth date (born, date of birth, DOB) | director (directed by, filmmaker, film director) | movie 'Inception' (Inception, film Inception, movie named Inception) | released in 2010 (2010, release year 2010, came out 2010)

Here are the objects that can be relevant:
(...4 objects omitted...)
- Inception was directed by Christopher Nolan.
- Lions live in the jungle.


Here are the objects that are enough to answer the user query:
(...2 objects omitted...)
- Inception was directed by Christopher Nolan.


User question: {user_query}/no_think
The relevant keywords are: """
    lm += gen(name='keywords_str', stop='\n')

    # Parse keywords (Python logic outside LLM generation flow but within the guidance function)
    keywords_str = lm['keywords_str'].strip()
    keywords = [k.strip() for k in keywords_str.split('|')]
    print(f"--- Identified Keywords: {keywords} ---")

    # 2. Rephrase keywords using dynamic constraints
    lm += "\nThe relevant n-grams are"
    rephrased_dict = {}

    for keyword_idx, keyword in enumerate(keywords): # Use enumerate for unique variable names
        lm += f" {keyword} (" # Start the rephrasing block for this keyword

        # ---- START: MODIFIED SECTION ----

        # === Step 1: Generate the VERY FIRST token using all_tokens ===
        print(f"--- Selecting first token for '{keyword}' from all_tokens ---")
        # Use select with the pre-loaded all_tokens list
        # Ensure unique name for capture variable across keywords and steps
        first_token_var = f"rephrase_{keyword_idx}_first_token"
        lm += select(options=all_tokens, name=first_token_var)

        # Initialize current_rephrase with the captured first token
        current_rephrase = lm[first_token_var]
        print(f"--- First token for '{keyword}': '{current_rephrase}' ---")


        # === Step 2: Generate SUBSEQUENT tokens using find_next_word_continuations ===
        max_rephrase_steps = 10 # Limit steps for safety in this example
        for step in range(max_rephrase_steps):
             # Get valid continuations from the Trie for the *current* rephrased string
             print(f"--- Finding continuations for: '{current_rephrase}' (step {step+1}) ---")
             continuations = find_next_word_continuations(current_rephrase, trie_index_path)

             # Add potential terminators for the current rephrase block
             valid_next_strings = continuations + [')', ','] # Now add terminators here
             valid_next_strings = list(set(valid_next_strings)) # Ensure unique

             print(f"--- Options (step {step+1}): {valid_next_strings[:10]}... ---")

             # Decide if we *must* or *can* stop
             can_stop = ')' in valid_next_strings or ',' in valid_next_strings
             must_stop = len(valid_next_strings) == 0 or \
                         (len(valid_next_strings) == 1 and valid_next_strings[0] in [')', ',']) or \
                         (len(valid_next_strings) == 2 and set(valid_next_strings) == {')', ','})

             if must_stop:
                 print(f"--- Must stop for '{keyword}' at '{current_rephrase}' ---")
                 # If only terminators are left, pick one (prefer ')' if both are options)
                 if ')' in valid_next_strings:
                     next_token_str = ')'
                 elif ',' in valid_next_strings:
                      next_token_str = ','
                 else:
                     # Should not happen if must_stop is true based on above logic, but defensive closure
                     next_token_str = ')'

                 lm += next_token_str # Force the stop token
                 # current_rephrase += next_token_str # Optionally add terminator to capture, depends on desired output
                 break # Exit the inner loop for this keyword

             elif can_stop:
                 # Allow selecting a terminator or continuing
                 # Prioritize non-terminators if available? Or let LLM choose? Let LLM choose for now.
                 current_token_var = f"rephrase_{keyword_idx}_token_{step+1}"
                 lm += select(options=valid_next_strings, name=current_token_var)
                 selected_token_str = lm[current_token_var]

                 current_rephrase += selected_token_str
                 if selected_token_str in [')', ',']:
                     print(f"--- Stopping for '{keyword}' because '{selected_token_str}' was selected ---")
                     break # Stop if a terminator is selected
             else:
                 # Cannot stop yet, must select from continuations
                 current_token_var = f"rephrase_{keyword_idx}_token_{step+1}"
                 lm += select(options=valid_next_strings, name=current_token_var) # Pass only continuations
                 selected_token_str = lm[current_token_var]
                 current_rephrase += selected_token_str


             # Safety break if loop runs too long
             if step == max_rephrase_steps - 1:
                 print(f"--- Max rephrase steps reached for '{keyword}', breaking. ---")
                 # Ensure termination if max steps hit
                 if current_rephrase[-1] not in [')', ',']:
                      lm += ")"
                 break


        # Store the final rephrased string (Python logic)
        # Need to parse the generated content between the parentheses correctly.
        # This is tricky as `lm` holds the entire generation history.
        # A more robust way would be to capture *only* the generation within the loop.
        # Let's capture the `current_rephrase` python variable for simplicity here.
        rephrased_dict[keyword] = current_rephrase
        print(f"--- Rephrased '{keyword}' to: '{current_rephrase}' ---")
        lm += " |" # Separator for the next keyword rephrasing block

    # Remove the last " |"
    lm = lm[:-2] # This modification needs care in guidance, might be better to handle formatting differently

    # Store the rephrased keywords in the lm state for potential later use
    lm = lm.set("rephrased_dict", rephrased_dict)

    # 3. Retrieval (Python logic)
    retrieved_results_dense: List[List[RetrievalResult]] = []
    all_retrieved_objects: List[str] = []

    # Prepare nlqs list for retrieval call
    nlqs_for_retrieval = [v for v in rephrased_dict.values() if v] # Use non-empty rephrased terms
    if nlqs_for_retrieval:
        retrieved_results_dense = dense_index.retrieve(
            nlqs=nlqs_for_retrieval, # Pass the list of rephrased keywords/n-grams
            output_folder="assets/feverous/faiss_indexes/dense_row_UAE-Large-V1", # Example path
            k=5
        )
        # Flatten the results and get object strings
        for result_list in retrieved_results_dense:
            for res in result_list:
                all_retrieved_objects.append(res.object)
        all_retrieved_objects = list(set(all_retrieved_objects)) # Make unique
    else:
        print("--- No valid rephrased keywords for retrieval ---")


    # 4. List all retrieved objects (Forced Generation)
    lm += "\n\nHere are the objects that can be relevant:\n"
    if all_retrieved_objects:
        for obj_text in all_retrieved_objects:
            lm += f"- {obj_text}\n" # Force the LLM to output each object string
    else:
        lm += "- (No objects retrieved)\n"

    # Store all retrieved objects text in the lm state
    lm = lm.set("all_retrieved_objects", all_retrieved_objects)

    # 5. Filter relevant objects
    lm += "\nHere are the objects that are enough to answer the user query:\n"
    # Let the LLM generate the filtered list based on the original query and the full list provided above
    lm += gen(name='filtered_objects', stop='<>', max_tokens=500) # Generate until the final stop token

    return lm

# --- Example Usage ---
user_query = "What is the full name of the jesus college alumni who graduated in 1960 ?"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_q = AutoModelForCausalLM.from_pretrained(
    "/data/hdd1/users/akouk/ARM/ARM/assets/cache/models--Qwen--Qwen2.5-32B-Instruct",         # or "./my_model_dir"
    quantization_config=bnb_config,
    device_map="auto"              # distribute across GPUs/CPU
)
tokenizer_q = AutoTokenizer.from_pretrained("/data/hdd1/users/akouk/ARM/ARM/assets/cache/models--Qwen--Qwen2.5-32B-Instruct")
model = models.Transformers(model_q, tokenizer_q, echo=False)

# Execute the guidance program
executed_program = model + dynamic_retrieval_guidance(user_query)

# Print the final generated output
print("\n--- Final LLM Output ---")
print(str(executed_program))