import guidance
from guidance import models, gen, system, user, assistant, capture, select
import transformers
import torch
import re

# --- Configuration (Same as before) ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"# Replace with a more capable model if needed

VALID_KEYWORDS = [
    "highest eligible free rate", "k-12 students", "schools",
    "most populous county", "california", "county population",
    "school districts", "free meal percent", "eligible free",
    "student enrollment"
]

# --- Tokenizer and LLM Loading (Same as before) ---
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.encode('[PAD]')[0]

# Load the model using guidance.models
guidance_model = models.Transformers(MODEL_NAME, echo=True)

# --- Trie Construction (Same as before) ---
def get_ngrams(text, n):
    tokens = tokenizer.encode(text)
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def build_ngram_trie(keywords, max_n=3):
    trie = {}
    all_ngram_tokens = set()
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    trie[ (bos_token_id,) ] = {}

    for keyword in keywords:
        tokens = tokenizer.encode(keyword)
        if not tokens: continue
        current_ngram_tokens = set()
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram_tuple = tuple(tokens[i:i+n])
                current_ngram_tokens.add(ngram_tuple)
                node = trie.setdefault((bos_token_id,), {})
                for j, token_id in enumerate(ngram_tuple):
                    # Decode token for grammar (Guidance grammars often work with strings)
                    token_str = tokenizer.decode([token_id])
                    # Use string keys in the grammar trie part? No, let's stick to token IDs for trie, decode during grammar build.
                    node.setdefault(token_id, {})
                    if j == len(ngram_tuple) - 1:
                        node[token_id]['<end>'] = True
                    node = node[token_id]
        all_ngram_tokens.update(current_ngram_tokens)
    print(f"Built Trie with {len(all_ngram_tokens)} unique N-gram token sequences.")
    return trie, all_ngram_tokens

ngram_trie, all_ngram_token_sequences = build_ngram_trie(VALID_KEYWORDS)

# --- Function to Build Guidance Grammar from Trie ---
# Note: This is complex and might need refinement based on exact grammar behavior.
# Guidance grammars typically operate on strings/bytes, need careful token handling.

# Cache for generated grammar parts to avoid recomputing
grammar_cache = {}

def build_grammar_from_trie_node(node):
    node_id = id(node) # Use node identity for caching key
    if node_id in grammar_cache:
        return grammar_cache[node_id]

    choices = []
    for token_id, next_node_info in node.items():
        if not isinstance(token_id, int): continue # Skip '<end>' marker

        # Decode token_id into its string representation for the grammar
        # Handle potential decoding errors or special tokens carefully
        try:
            # Important: Decoding single tokens might need prefix spaces handled depending on tokenizer
            token_str = tokenizer.decode([token_id])
            # If decode returns empty or special chars, might need alternative (e.g., byte representation)
            if not token_str or token_str == tokenizer.unk_token or token_str == tokenizer.pad_token:
                 # Fallback: Use byte representation? This gets very complex.
                 # Let's skip problematic tokens for now.
                 print(f"Warning: Skipping potentially problematic token ID {token_id} -> '{token_str}'")
                 continue
        except Exception as e:
            print(f"Warning: Error decoding token {token_id}: {e}")
            continue


        # Check if this path can terminate here
        can_end_here = '<end>' in next_node_info

        # Check if this path can continue
        can_continue = any(isinstance(k, int) for k in next_node_info.keys())

        if can_continue:
            # Recursively build the grammar for the next step
            next_grammar = build_grammar_from_trie_node(next_node_info)
            if can_end_here:
                # Option to end here OR continue
                # `select(["", next_grammar])` means choose empty string (end) or continue
                choices.append(token_str + select(["", next_grammar]))
            else:
                # Must continue
                choices.append(token_str + next_grammar)
        elif can_end_here:
            # Must end here
            choices.append(token_str)
        # Else: Node exists but leads nowhere valid? Skip.

    if not choices:
        # If no valid choices from this node, it's effectively an empty path end
        # guidance.epsilon() might be the correct representation, but let's return None
        # to signify no valid continuation, handled by select below.
         # Returning "" instead? Let's try None for clarity.
        grammar_cache[node_id] = None # Cache the result
        return None


    # Combine choices using select. Filter out None results.
    valid_choices = [c for c in choices if c is not None]
    if not valid_choices:
        grammar_cache[node_id] = None
        return None

    result_grammar = select(valid_choices)
    grammar_cache[node_id] = result_grammar # Cache the result
    return result_grammar

# Build the final grammar starting from the root
root_node = ngram_trie.get((tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,), {})
print("Building grammar...")
ngram_grammar = build_grammar_from_trie_node(root_node)
print("Grammar building complete.")

if ngram_grammar is None:
    print("ERROR: Failed to build a valid grammar from the Trie root.")
    exit()
# print("Generated Grammar (partial):", str(ngram_grammar)[:500]) # Debug print

# --- Simulated Retrieval Function (Synchronous - Same as before) ---
def simulate_retrieval(generated_ngrams_string):
    print(f"\n--- Running Simulate Retrieval with N-grams: '{generated_ngrams_string}' ---")
    retrieved = [
        "Dummy Document 1: Mentions K-12 students and free rates.",
        "Dummy Table Snippet A: Shows county populations for California.",
        "Dummy Document 2: Discusses school district policies."
    ]
    print("--- Simulation Complete ---")
    return retrieved

# --- Guidance Program Definition (@guidance function using Grammar) ---
@guidance
def arm_logic(lm, user_question):

    # 1. Analysis
    lm += f"""
Analysis:
{gen(name='analysis', temperature=0.0, max_tokens=100, stop='Relevant N-grams:')}"""
    lm += "Relevant N-grams: (" # Add the trigger explicitly

    # 2. N-gram Generation (Using the generated grammar)
    # We use the grammar object directly. `capture` saves the matched output.
    lm += capture(ngram_grammar, name='ngrams')

    lm += ")" # Add the closing parenthesis

    # 3. External Retrieval Call (Synchronous)
    retrieved_content_list = simulate_retrieval(lm['ngrams'])
    lm = lm.set("retrieved_content", retrieved_content_list)

    # 4. Inject Retrieved Content & Prompt for Verification
    lm += "\n\nSimulated Retrieved Content:\n"
    for item in retrieved_content_list:
        lm += f"- {item}\n"
    lm += "\nVerification and Summary based on Retrieved Content:\n"

    # 5. Final Verification Generation
    lm += gen(name='verification', temperature=0.0, max_tokens=150)

    return lm


def run_arm_test(guidance_model, user_question, test_num):
    print(f"\n--- Starting Guidance Program: Test {test_num} ---")
    print(f"User Question: {user_question}")

    # Construct the initial prompt
    lm = guidance_model
    lm += f"""Act as an AI assistant simulating the ARM retrieval process.
Follow these steps:
1. Analyze the question: {user_question}
2. Extract keywords and align them to N-grams between '(' and ')'. Use *only* N-grams allowed by the provided grammar.
3. Simulate retrieval based on aligned N-grams.
4. Summarize findings based *only* on the simulated retrieved content.

"""
    # Execute the main logic function
    try:
        final_lm_state = lm + arm_logic(user_question=user_question)
        print("\n--- Guidance Program Final State (Output follows) ---")
        # Output is implicitly printed via echo=True
        # print(final_lm_state) # Optional: print state object
    except Exception as e:
        print(f"\n--- ERROR during Guidance execution for Test {test_num} ---")
        print(e)
        import traceback
        traceback.print_exc()

    print(f"--- End of Execution: Test {test_num} ---")


# --- Run Multiple Tests ---
test_questions = [
    "What is the highest eligible free rate for K-12 students in the schools in the most populous county in California?",
    "Show me the population data for california counties.",
    "What percentage of students get free meals in LA schools?" # Note: 'LA' isn't a keyword, model has to generalize/use others
]

for i, q in enumerate(test_questions):
    run_arm_test(guidance_model, q, i+1)
    print("\n" + "="*50 + "\n") # Separator