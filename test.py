import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Assuming train_utils is in a subdirectory relative to this script,
# or that train_utils is in the PYTHONPATH.
# If you place this script at the root of RecLM-cgen, these imports should work.
from train_utils.processor import FastPrefixConstrainedLogitsProcessor, Trie_link
from train_utils.utils import get_ctrl_item

def run_constrained_generation_test():
    # 1. Initialize tokenizer and model
    model_name = "gpt2"  # Using a small, standard model for simplicity
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add and set special tokens
    # <SOI> for Start Of Item, <EOI> for End Of Item
    tokens_to_add = ['<SOI>', '<EOI>']
    if tokenizer.pad_token is None:
        # Add [PAD] token if tokenizer doesn't have one (like gpt2)
        tokens_to_add.append('[PAD]')
        
    tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
    
    if tokenizer.pad_token_id is None: # If [PAD] was just added and not set as pad_token
        tokenizer.pad_token = '[PAD]'
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')


    # Set custom attributes for SOI/EOI token IDs, as expected by Trie_link
    # These must be attributes of the tokenizer instance.
    tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids('<SOI>')
    tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids('<EOI>')
    
    # Resize model embeddings to accommodate new special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Setup device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode

    print(f"\n--- Tokenizer and Model Setup ---")
    print(f"Using device: {device}")
    print(f"SOI token: '{tokenizer.decode(tokenizer.soi_token_id)}', ID: {tokenizer.soi_token_id}")
    print(f"EOI token: '{tokenizer.decode(tokenizer.eoi_token_id)}', ID: {tokenizer.eoi_token_id}")
    print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"Vocabulary size: {len(tokenizer)}")


    # 2. Define in-domain titles (list of 3 words as requested)
    in_domain_titles = ["apple", "banana", "cherry","yepiskoposyan","yaposkibesan"]
    print(f"\n--- In-domain Titles ---")
    print(f"Titles: {in_domain_titles}")
    
    # Encode titles: list of lists of token IDs. 
    # add_special_tokens=False because Trie_link handles adding <SOI> and <EOI> to its internal structure.
    item_ids_list = tokenizer.batch_encode_plus(in_domain_titles, add_special_tokens=False).input_ids
    # print(f"Encoded item IDs (for Trie): {item_ids_list}")
    
    # 3. Define num_beams for the logits processor
    num_beams = 2

    # 4. Create prefix tree
    print("\n--- Prefix Tree and Processor ---")
    print("Building prefix tree...")
    item_prefix_tree = Trie_link(item_ids_list, tokenizer)

    # 5. Create logit processor
    # The constrain_search_list method of Trie_link is the core function for constraining.
    processor = FastPrefixConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=item_prefix_tree.constrain_search_list, 
        num_beams=num_beams
    )
    print("Logits processor created.")

    # 6. Define query (1 "main" word in query, plus <SOI> to trigger constraint)
    # The query "I want an " uses "an" as the single query word in context.
    # "<SOI>" is appended to the prompt to instruct the model to start generating an item.
    query_text = "yepiskoposan <SOI>" 
    input_ids = tokenizer.encode(query_text, return_tensors="pt").to(device)

    # 7. Generate output
    print(f"\n--- Generation ---")
    print(f"Query: {query_text}")
    # print(f"Input token IDs: {input_ids.tolist()}")
    
    with torch.no_grad(): # Disable gradient calculations for inference
        output_sequences = model.generate(
            input_ids=input_ids,
            logits_processor=[processor], # Apply the custom logits processor
            num_beams=num_beams, # MODIFIED: Set to 2
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.eoi_token_id],
            num_return_sequences=num_beams, # MODIFIED: Return all beams
            early_stopping=True # MODIFIED: Recommended for beam search
        )

    # 8. Print and analyze output
    print(f"\n--- Output Analysis (for {output_sequences.shape[0]} beams) ---") # MODIFIED: Updated print statement
    num_input_tokens = input_ids.shape[1]

    for i, beam_output in enumerate(output_sequences): # MODIFIED: Loop through each beam
        print(f"\n--- Beam {i+1} ---")
        generated_token_ids_full = beam_output.tolist()
        full_generated_text = tokenizer.decode(generated_token_ids_full, skip_special_tokens=False)
        print(f"Full generated text (Beam {i+1}): {full_generated_text}")

        generated_part_ids = beam_output[num_input_tokens:]
        generated_part_text = tokenizer.decode(generated_part_ids, skip_special_tokens=False)
        print(f"Newly generated part (Beam {i+1}): {generated_part_text}")

        extracted_items = get_ctrl_item(full_generated_text)
        print(f"Extracted items using get_ctrl_item (Beam {i+1}): {extracted_items}")

        if extracted_items and extracted_items[0] in in_domain_titles:
            print(f"SUCCESS (Beam {i+1}): Constrained generation produced an in-domain item and stopped appropriately.")
        else:
            print(f"NOTE (Beam {i+1}): Constrained generation did not produce an in-domain item as expected or stop correctly. Check logs and setup.")

if __name__ == "__main__":
    run_constrained_generation_test()