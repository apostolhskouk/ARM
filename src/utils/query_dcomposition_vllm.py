import time
from typing import List, Optional, Dict, Any
from vllm import LLM, SamplingParams, EngineArgs
from vllm.utils import FlexibleArgumentParser

# Define a default prompt template for decomposition.
# This is crucial and often requires tuning based on the model used.
# It uses a few-shot approach.
DEFAULT_DECOMPOSITION_PROMPT_TEMPLATE = """
Instruction: Decompose the user query into multiple simpler sub-queries that can be answered independently and then combined to answer the original query. Output each sub-query on a new line, starting with '- '. Do not include the original query in the output. If the query is already simple, output only the original query.

Query: What were the total sales and profit margin for product X in Q4 2023?
Decomposition:
- What were the total sales for product X in Q4 2023?
- What was the profit margin for product X in Q4 2023?

Query: Which European capitals have the highest population density and lowest average rainfall?
Decomposition:
- List European capitals.
- Find the population density for each European capital.
- Find the average rainfall for each European capital.
- Identify the European capital with the highest population density.
- Identify the European capital with the lowest average rainfall.

Query: Tell me about the history of the Eiffel Tower.
Decomposition:
- Tell me about the history of the Eiffel Tower.

Query: {nlq}
Decomposition:
"""

class DecomposerVLLM:
    """
    A class to decompose natural language queries (NLQs) into sub-queries
    using the vLLM library for efficient batch processing and caching.
    """
    def __init__(
        self,
        model_id: str,
        download_dir: Optional[str] = None,
        prompt_template: str = DEFAULT_DECOMPOSITION_PROMPT_TEMPLATE,
        max_new_tokens: int = 150,
        stop_sequences: Optional[List[str]] = None, # e.g., ["\n\n", "Query:"]
        temperature: float = 0.0, # Lower temperature for more deterministic decomposition
        enable_prefix_caching: bool = True,
        quantization: Optional[str] = None, # e.g., "int8" - requires llm-compressor and compatible GPU
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        enforce_eager: bool = False, # Set True to disable CUDA graphs if memory issues arise
        engine_args_override: Optional[Dict[str, Any]] = None,
        **kwargs # Catch-all for other potential LLM/Engine args
    ):
        """
        Initializes the DecomposerVLLM.

        Args:
            model_id (str): The Hugging Face model ID or local path for the LLM.
            prompt_template (str): A format string template for the decomposition prompt.
                                   Must contain '{nlq}' for the natural language query.
            max_new_tokens (int): Maximum number of tokens to generate for the decomposition.
            stop_sequences (Optional[List[str]]): List of strings that will stop generation.
                                                 Crucial for getting structured output.
                                                 If None, defaults might depend on the model.
            temperature (float): Sampling temperature for generation. Lower is more deterministic.
            enable_prefix_caching (bool): Enables vLLM's Automatic Prefix Caching.
            quantization (Optional[str]): Enables quantization (e.g., "int8"). Requires setup.
            tensor_parallel_size (int): Number of GPUs for tensor parallelism.
            gpu_memory_utilization (float): Proportion of GPU memory to reserve.
            max_model_len (Optional[int]): Maximum sequence length for the model.
            enforce_eager (bool): Disable CUDA graphs (can save memory but reduce speed).
            engine_args_override (Optional[Dict[str, Any]]): Dictionary to override specific EngineArgs.
            **kwargs: Additional arguments passed directly to the LLM constructor.
        """
        print(f"Initializing DecomposerVLLM with model: {model_id}")
        self.model_id = model_id
        self.prompt_template = prompt_template
        self._validate_prompt_template()

        # Define Sampling Parameters specific to decomposition
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop_sequences if stop_sequences is not None else [],
            # Add other relevant params like top_p, top_k if needed, but often not for decomposition
            # top_p=0.95,
        )

        # --- vLLM Engine Configuration ---
        # Start with default engine args and allow overrides
        engine_args_dict = {
            "model": model_id,
            "enable_prefix_caching": enable_prefix_caching,
            "quantization": quantization,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            # Add other relevant engine args if needed
            # "dtype": "auto", # Usually auto is fine
            # "max_num_seqs": 256, # Default is usually okay unless hitting limits
        }

        # Apply overrides if provided
        if engine_args_override:
            engine_args_dict.update(engine_args_override)

        # Allow direct LLM kwargs to override engine args dict for convenience
        engine_args_dict.update(kwargs)

        # It's good practice to use EngineArgs for clarity if complex config is needed,
        # but direct kwargs to LLM often suffice for simpler cases.
        # For robustness, let's build EngineArgs explicitly
        # Note: Building EngineArgs requires careful handling of argument sources (CLI vs. direct)
        # For programmatic use like this, passing kwargs directly to LLM is often simpler.
        # Let's stick to passing kwargs directly to LLM constructor:

        llm_kwargs = {
            "model": model_id,
            "enable_prefix_caching": enable_prefix_caching,
            "quantization": quantization,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            **({"download_dir": download_dir} if download_dir is not None else {}),
            **kwargs # Include any extra arguments passed
        }

        # Clean up potential None values if LLM complains, though it's usually robust
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        print(f"vLLM LLM Configuration: {llm_kwargs}")
        if quantization == "int8":
            print("INT8 Quantization enabled. Ensure 'llm-compressor' is installed and GPU compute capability > 7.5.")
            print("You might need to load a model specifically quantized for vLLM INT8 W8A8.")
            print("See: https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415")
        if enable_prefix_caching:
            print("Automatic Prefix Caching enabled.")

        # Initialize the vLLM engine
        self.llm = LLM(**llm_kwargs)
        print("DecomposerVLLM initialized successfully.")

    def _validate_prompt_template(self):
        """Ensures the prompt template is valid."""
        if "{nlq}" not in self.prompt_template:
            raise ValueError("prompt_template must contain '{nlq}' placeholder.")

    def _build_prompts(self, nlqs: List[str]) -> List[str]:
        """Builds the full prompts for the LLM based on the template."""
        return [self.prompt_template.format(nlq=nlq) for nlq in nlqs]

    def _parse_outputs(self, outputs: List['RequestOutput']) -> List[List[str]]:
        """Parses the LLM outputs into lists of sub-queries."""
        decomposed_list = []
        for output in outputs:
            # We usually take the first completion result (index 0)
            # output.outputs is a list because of potential 'n' > 1 in SamplingParams
            if not output.outputs:
                 print(f"Warning: Request {output.request_id} produced no output.")
                 decomposed_list.append([]) # Or handle as error?
                 continue

            generated_text = output.outputs[0].text.strip()

            # Parsing logic: split by newline, remove common prefixes/bullets
            sub_queries = []
            for line in generated_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Remove common prefixes like '- ' or '* ' if they exist
                if line.startswith("- "):
                    line = line[2:].strip()
                elif line.startswith("* "):
                     line = line[2:].strip()
                # Add only non-empty lines
                if line:
                    sub_queries.append(line)

            # Handle cases where parsing results in nothing or just the original query was returned
            if not sub_queries:
                 # Maybe the model failed or the query was simple - return original query?
                 # Find original NLQ from prompt
                 original_nlq = output.prompt.split("Query:")[-1].split("\nDecomposition:")[0].strip()
                 print(f"Warning: No sub-queries parsed for request {output.request_id}. Returning original: '{original_nlq}'")
                 decomposed_list.append([original_nlq]) # Return original query as list
            else:
                decomposed_list.append(sub_queries)

        return decomposed_list

    def decompose(self, nlqs: List[str]) -> List[List[str]]:
        """
        Decomposes a batch of natural language queries into sub-queries.

        Args:
            nlqs (List[str]): A list of natural language queries to decompose.

        Returns:
            List[List[str]]: A list where each element is a list of sub-queries
                             corresponding to the input NLQ at the same index.
        """
        if not nlqs:
            return []

        print(f"Decomposing {len(nlqs)} queries...")
        start_time = time.time()

        # 1. Build prompts for the batch
        prompts = self._build_prompts(nlqs)

        # 2. Generate decompositions using vLLM (handles batching internally)
        # The `generate` method takes prompts and common sampling parameters.
        # request_id is automatically handled internally if not provided.
        outputs = self.llm.generate(prompts, self.sampling_params)

        # 3. Parse the generated outputs
        decomposed_queries = self._parse_outputs(outputs)

        end_time = time.time()
        print(f"Decomposition finished in {end_time - start_time:.2f} seconds.")

        # Sanity check
        if len(decomposed_queries) != len(nlqs):
             print(f"Warning: Number of outputs ({len(decomposed_queries)}) does not match number of inputs ({len(nlqs)}).")
             # Decide how to handle this - potentially raise an error or return partial results?
             # For now, let's return what we have, but this indicates an issue.

        return decomposed_queries

    def __del__(self):
        """Clean up resources if needed (vLLM might handle this internally)."""
        # Explicit cleanup isn't typically required for the LLM object itself,
        # but good practice if managing external resources.
        print("DecomposerVLLM shutting down.")
        # If using LLMEngine directly, you might call engine.shutdown() here.
        # With the LLM class, Python's garbage collection usually suffices.


# --- Example Usage ---
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-32B-Instruct" # Or another suitable model

    # Optional: Enable INT8 quantization if desired and requirements met
    USE_INT8 = False # Set to True to try INT8
    QUANTIZATION_MODE = "int8" if USE_INT8 else None

    # Optional: If using multiple GPUs
    TENSOR_PARALLEL_SIZE = 2

    try:
        # Initialize the decomposer
        decomposer = DecomposerVLLM(
            model_id=MODEL_ID,
            quantization=QUANTIZATION_MODE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            download_dir="assets/cache"
            # You can customize other parameters here:
            # max_new_tokens=200,
            # stop_sequences=["\n\n", "Query:"],
            # gpu_memory_utilization=0.85, # Adjust if needed
        )

        # --- Test Queries ---
        nlqs_to_decompose = [
            "What is the capital of France and its population?",
            "Compare the battery life and camera specs of the latest iPhone and Samsung Galaxy.",
            "Summarize the plot of the movie Inception.",
            "Find restaurants near me that serve vegan pizza and are open late.",
            "What programming language is fastest?", # A potentially simple query
            "Who won the Nobel Prize in Physics in 2021 and what was their contribution?",
        ]

        # Add a duplicate query to test prefix caching effectiveness
        nlqs_to_decompose.append("What is the capital of France and its population?")


        # --- Perform Decomposition ---
        decomposed_results = decomposer.decompose(nlqs_to_decompose)

        # --- Print Results ---
        print("\n--- Decomposition Results ---")
        for i, original_nlq in enumerate(nlqs_to_decompose):
            print(f"Original Query {i+1}: {original_nlq}")
            print(f"Decomposed:")
            if i < len(decomposed_results): # Check if result exists (due to potential mismatch warning)
                 for sub_query in decomposed_results[i]:
                     print(f"  - {sub_query}")
            else:
                 print("  [Error: No result]")
            print("-" * 20)

        # --- Test Prefix Caching (APC) ---
        print("\n--- Testing Prefix Caching ---")
        # Create prompts that share a significant prefix
        common_prefix = DEFAULT_DECOMPOSITION_PROMPT_TEMPLATE.split("{nlq}")[0] # Get the instruction part
        long_context = "Background: The company Acme Corp reported its Q3 earnings yesterday. Revenue was $50M, up 10% YoY. Net income was $5M. The main driver was the new 'Widget Pro' product line launched in Q2. Expenses increased due to marketing for Widget Pro. The CEO highlighted strong international growth, particularly in Europe.\n\n"

        nlqs_with_prefix = [
            long_context + "Query: What was Acme Corp's Q3 revenue and net income?",
            long_context + "Query: Which product line drove Acme Corp's Q3 growth?",
            long_context + "Query: Did Acme Corp's expenses increase in Q3?",
        ]

        # First run
        print("\nRunning decomposition with long context (first time):")
        start_cache_test1 = time.time()
        results1 = decomposer.decompose([nlqs_with_prefix[0]]) # Just one first
        end_cache_test1 = time.time()
        print(f"Time taken (1st query): {end_cache_test1 - start_cache_test1:.2f}s")
        for sub in results1[0]: print(f"  - {sub}")


        # Second run with shared prefix - should be faster due to APC
        print("\nRunning decomposition with long context (cached prefix):")
        start_cache_test2 = time.time()
        # Run the others, or even the first one again + others
        results2 = decomposer.decompose(nlqs_with_prefix)
        end_cache_test2 = time.time()
        print(f"Time taken ({len(nlqs_with_prefix)} queries, cached): {end_cache_test2 - start_cache_test2:.2f}s")
        for i, res_list in enumerate(results2):
            print(f"Query {i+1}:")
            for sub in res_list: print(f"  - {sub}")

        # Compare timings

    except ImportError as e:
        print(f"Error: {e}. Make sure vLLM is installed (`pip install vllm`).")
        if "llmcompressor" in str(e) and USE_INT8:
             print("INT8 quantization requires 'llm-compressor'. Install with `pip install llmcompressor`.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()