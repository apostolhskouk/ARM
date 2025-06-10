import json
import os
import pickle
from typing import List, Dict, Any, Set, Tuple, Callable
from src.retrieval.dense import FaissDenseRetriever

from src.retrieval.base import RetrievalResult
import faiss
import torch
from tqdm.auto import tqdm

import guidance
from guidance import models, gen, select, assistant
import traceback 

REACT_PROMPT_TEMPLATE = """Solve a question answering task with interleaving Thought, Action, Observation steps. A Thought step must be followed by a Action step, an Action step must be followed by an Observation step (unless Action is Finish), and an Observation step must be followed by a Thought step. Thought can reason about the current situation. Action can be of two types:
(1) Search[keyword], which returns relevant documents or tables. Use precise keywords based on your thought process.
(2) Finish[answer], which returns the final answer and finishes the task.

Important: Your primary goal is to find documents that directly answer the question. Continue using the Search action, possibly multiple times with refined keywords, until you have retrieved observations that contain the necessary information. The final answer provided in the Finish action must be explicitly derived from the content of these observations. Do not stop searching prematurely if the current observations are insufficient, ambiguous, or do not directly lead to the answer. It is acceptable to retrieve some irrelevant documents as long as you eventually find the ones that support the answer. Ensure your reasoning in the Thought step clearly justifies why the retrieved information is sufficient before using Finish.

Here are some examples:

User question: what is the full name of the jesus college alumni who graduated in 1960 ?
Thought: I need to search jesus college alumni, find the people that graduated in 1960, then find their full names.
Action: Search[jesus college alumni 1960]
Observation:
page_title: List_of_politicians,_lawyers,_and_civil_servants_educated_at_Jesus_College,_Oxford
source: table_0
object: List_of_politicians,_lawyers,_and_civil_servants_educated_at_Jesus_College,_Oxford [SEP] | Name | M | G | Degree | Notes | [SEP] | Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993 |
page_title: List_of_politicians,_lawyers,_and_civil_servants_educated_at_Jesus_College,_Oxford
source: table_0
object: List_of_politicians,_lawyers,_and_civil_servants_educated_at_Jesus_College,_Oxford [SEP] | Name | M | G | Degree | Notes | [SEP] | Neal Blewett ( HF ) | 1957 | 1959 | BA PPE ( 2nd ) | Member of the Australian House of Representatives ( 1977-1994 ) , Government Minister ( 1983-1994 ) , High Commissioner to the UK ( 1994-1998 ) |
page_title: Jesus_College,_Oxford
source: sentence_5
object: Jesus_College,_Oxford [SEP] Jesus College is one of the constituent colleges of the [[University_of_Oxford|University of Oxford]] in [[England|England]].
page_title: Jesus_College,_Oxford
source: sentence_0
object: Jesus_College,_Oxford [SEP] It was founded by [[Elizabeth_I|Elizabeth I]] on 27 June 1571.
page_title: List_of_Oxford_University_people
source: sentence_12
object: List_of_Oxford_University_people [SEP] Alumni of Jesus College include [[T._E.Lawrence|T. E. Lawrence]] ('Lawrence of Arabia'), [[Harold_Wilson|Harold Wilson]] (twice [[Prime_Minister_of_the_United_Kingdom|Prime Minister of the UK]]), [[Angus_Buchanan(VC)|Angus Buchanan]] [[VC]], and [[Pixley_ka_Isaka_Seme|Pixley ka Isaka Seme]], founder of the [[African_National_Congress|African National Congress]].
Thought: The first result from the table 'List_of_politicians...' lists Lalith Athulathmudali with a 1960 BCL degree. This seems relevant. I need to find his full name. I should search specifically for 'Lalith Athulathmudali' to confirm and get more details.
Action: Search[Lalith Athulathmudali full name]
Observation:
page_title: Lalith_Athulathmudali
source: sentence_0
object: Lalith_Athulathmudali [SEP] Lalith William Samarasekera Athulathmudali, [[Queen's_Counsel|PC]] ({{lang-si|ලලිත් විලියම් සමරසේකර අතුලත්මුදලි}}; 26 November 1936 – 23 April 1993), known as Lalith Athulathmudali, was a [[Sri_Lanka|Sri Lankan]] statesman.
page_title: Lalith_Athulathmudali
source: sentence_1
object: Lalith_Athulathmudali [SEP] He was a prominent member of the [[United_National_Party|United National Party]], who served as [[Minister_of_Trade_(Sri_Lanka)|Minister of Trade and Shipping]] ; [[Minister_of_National_Security_(Sri_Lanka)|Minister National Security]] and [[Deputy_Minister_of_Defence_(Sri_Lanka)|Deputy Minister of Defence]] ; [[Minister_of_Agriculture_(Sri_Lanka)|Minister of Agriculture]], Food and Cooperatives and finally [[Minister_of_Education_(Sri_Lanka)|Minister of Education]].
page_title: Assassination_of_Lalith_Athulathmudali
source: sentence_0
object: Assassination_of_Lalith_Athulathmudali [SEP] The assassination of Lalith Athulathmudali took place on 23 April 1993 in [[Kirulapana]], [[Sri_Lanka|Sri Lanka]].
page_title: Gamini_Dissanayake
source: sentence_5
object: Gamini_Dissanayake [SEP] Dissanayake, along with [[Lalith_Athulathmudali|Lalith Athulathmudali]], spearheaded the [[Impeachment_process_against_Ranasinghe_Premadasa|impeachment motion]] against President [[Ranasinghe_Premadasa|Premadasa]] in 1991.
page_title: Sri_Lanka
source: sentence_20
object: Sri_Lanka [SEP] Notable political figures include [[Sirimavo_Bandaranaike|Sirimavo Bandaranaike]], the world's first female head of government, and individuals like [[Lalith_Athulathmudali|Lalith Athulathmudali]].

Thought: The first result provides the full name: Lalith William Samarasekera Athulathmudali. This information is directly from the retrieved documents and answers the question.
Action: Finish[Lalith William Samarasekera Athulathmudali]

User question: the home stadium of the bray wanderers of 2006 league of ireland is situated behind what station?
Thought: I need to find the home stadium of Bray Wanderers in the 2006 League of Ireland, identify its location, and then find the station behind it.
Action: Search[Bray Wanderers 2006 League of Ireland home stadium location]
Observation:
page_title: Carlisle_Grounds
source: sentence_0
object: Carlisle_Grounds [SEP] Carlisle Grounds is a [[Association_football|football]] stadium in [[Bray,_County_Wicklow|Bray]], [[County_Wicklow|County Wicklow]], [[Ireland|Ireland]].
page_title: Carlisle_Grounds
source: sentence_1
object: Carlisle_Grounds [SEP] It is the home ground of [[League_of_Ireland_Premier_Division|League of Ireland]] club [[Bray_Wanderers_A.F.C.|Bray Wanderers A.F.C.]]
page_title: Bray_Daly_railway_station
source: sentence_0
object: Bray_Daly_railway_station [SEP] Bray Daly railway station ([[Irish_language|Irish]]: Stáisiún Bhré / Uí Dhálaigh) serves Bray in County Wicklow, Ireland.
page_title: Bray_Daly_railway_station
source: sentence_2
object: Bray_Daly_railway_station [SEP] It is located adjacent to the [[Carlisle_Grounds|Carlisle Grounds]] on the Albert Walk.
page_title: Bray_Wanderers_A.F.C.
source: sentence_21
object: Bray_Wanderers_A.F.C. [SEP] In the [[2006_League_of_Ireland_Premier_Division|2006 League of Ireland Premier Division]] season, Bray Wanderers finished 10th under manager [[Pat_Devlin|Pat Devlin]], playing their home games at the Carlisle Grounds.
Thought: The results identify the stadium as Carlisle Grounds in Bray. The Bray_Daly_railway_station page explicitly states it is located adjacent to the Carlisle Grounds. Now I need to confirm the 'behind' relationship, as 'adjacent' is not specific enough. I will search for information about the relative positions.
Action: Search[Carlisle Grounds Bray Daly station relationship behind adjacent]
Observation:
page_title: Bray_Daly_railway_station
source: sentence_2
object: Bray_Daly_railway_station [SEP] It is located adjacent to the [[Carlisle_Grounds|Carlisle Grounds]] on the Albert Walk.
page_title: Carlisle_Grounds
source: sentence_9
object: Carlisle_Grounds [SEP] Directions to Carlisle Grounds: By DART/Train: Take the DART south bound to Bray (terminus). The Carlisle Grounds are situated immediately beside/behind the station platform exit.
page_title: Bray_Seafront
source: sentence_3
object: Bray_Seafront [SEP] Key landmarks include the Bray Head hill, the promenade, Bray Daly Station, and the adjacent Carlisle Grounds.
page_title: Travel_to_Bray_Wanderers
source: sentence_4
object: Travel_to_Bray_Wanderers [SEP] For away fans travelling by train, exit Bray Daly station and the Carlisle Grounds entrance is immediately visible as it sits right next to the station.
page_title: Bray_Town_Council_Minutes
source: sentence_2
object: Bray_Town_Council_Minutes [SEP] Discussion regarding access improvements near Bray Daly Station noted its direct adjacency to the Carlisle Grounds.

Thought: The observations confirm Bray Daly station is adjacent/beside/next to the Carlisle Grounds. Sentence 9 from Carlisle_Grounds explicitly mentions "beside/behind the station platform exit". This retrieved information directly supports that Bray Daly is the station and it's behind (part of beside/behind) the station exit, implying the grounds are behind the station from that perspective.
Action: Finish[Bray Daly railway station]

--- Start of Current Task ---
"""

# --- Guidance Program Definition (Moved Outside Class) ---
@guidance(stateless=False, dedent=False)
def react_guidance_program(lm, question: str, max_rounds: int, search_func: Callable[[str], str], initial_prompt: str):
    lm += initial_prompt
    lm += f"\nUser question: {question}"
    with assistant():
        for i in range(1, max_rounds + 1):
            lm += f'\nThought: {gen(name=f"thought_{i}", stop="Action:", temperature=0.2,max_tokens=512)}'
            lm += f'\nAction: {select(["Search", "Finish"], name=f"act_{i}")}'
            lm += f'[{gen(name=f"arg_{i}", stop="]", max_tokens=128, temperature=0.2)}]'
            current_act = lm.get(f'act_{i}')
            current_arg = lm.get(f'arg_{i}')

            if current_act == 'Finish':
                lm += "\n"
                break
            elif current_act == 'Search' and current_arg is not None and current_arg.strip():
                search_query_to_print = current_arg.strip()
                try:
                    observation = search_func(search_query_to_print)
                except Exception as e:
                    observation = "Observation: Error during search execution."
                lm += f'\nObservation:\n{observation}'
            elif current_act == 'Search':
                 print(f"ReAct Step {i} - Search Query: [EMPTY/INVALID]")
                 lm += '\nObservation: Failed to generate valid search keywords.'
            else:
                action_str = str(current_act) if current_act is not None else "None"
                arg_str = str(current_arg) if current_arg is not None else "None"
                lm += f'\nObservation: An unexpected action type "{action_str}" with argument "{arg_str}" was generated. Continuing to next thought.'

    final_arg = None
    last_act = None
    last_checked_round = i if 'i' in locals() else max_rounds
    for round_num in range(last_checked_round, 0, -1): # Check from last round backwards
        act = lm.get(f'act_{round_num}')
        arg = lm.get(f'arg_{round_num}')
        if act == 'Finish':
            final_arg = arg
            last_act = act
            break
        elif act is not None and last_act is None: # Store the last non-Finish action if no Finish found yet
             last_act = act
    if last_act:
        lm = lm.set('final_action', last_act) 
    if final_arg:
        lm = lm.set('final_argument', final_arg) 

    return lm

    


# --- ReAct Retriever Implementation ---
class ReActRetriever(FaissDenseRetriever):
    """
    Implements the ReAct retrieval strategy using the 'guidance' library.
    Uses an LLM to generate thoughts and search actions via structured guidance,
    and uses the underlying dense retriever for the search steps.
    Collects all retrieved documents across search actions for a query.
    """
    def __init__(self,
                 dense_model_name_or_path: str = "WhereIsAI/UAE-Large-V1",
                 model_path: str = "unsloth/gemma-3-27b-it-bnb-4bit",
                 max_iterations: int = 5,
                 k_react_search: int = 5,
                 llm_n_ctx: int = 32768,
                 ):
        """
        Initializes the ReAct retriever using the guidance library.
        """
        # Initialize the base dense retriever (handles dense model loading)
        super().__init__(model_name_or_path=dense_model_name_or_path)

        self.model_path = model_path
        self.max_iterations = max_iterations
        self.k_react_search = k_react_search
        self.llm_n_ctx = llm_n_ctx

        self.guidance_lm = None 

        # Metrics storage
        self.distinct_retrieved_counts_per_query: List[int] = []
        self.llm_search_calls_per_query: List[int] = []

        if not hasattr(self, 'device'):
             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Initialize Guidance LLM ---
        self._initialize_guidance_llm()

    def _initialize_guidance_llm(self):
        """Initializes the Guidance LLM model."""
        if self.guidance_lm is not None:
            return
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Error: LLM model path '{self.model_path}' not found or not specified.")
            self.guidance_lm = None
            return
        self.guidance_lm = models.Transformers(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            echo=False
        )
        #self.guidance_lm = models.LlamaCpp(
        #    self.model_path,
        #    n_gpu_layers=-1,
        #    n_ctx=self.llm_n_ctx,
        #    repeat_penalty=1.2,
        #    echo=False
        #)



    def _format_observation(self, results: List[RetrievalResult]) -> str:
        """Formats retrieval results into an observation string for the LLM."""
        if not results:
            return "No relevant documents found." # More informative than just "No documents found."

        obs_str = ""
        for res in results:
            # Use get with default values for safety
            page_title = res.metadata.get('page_title', 'Unknown Page')
            source = res.metadata.get('source', 'Unknown Source')
            content = getattr(res, 'object', 'No Content') # Use getattr for safety

            # Ensure content is a string before adding
            if not isinstance(content, str):
                content = str(content) # Convert if not string

            # Format consistent with the examples in REACT_PROMPT_TEMPLATE
            obs_str += f"page_title: {page_title}\n"
            obs_str += f"source: {source}\n"
            obs_str += f"object: {page_title} [SEP] {content.strip()}\n---\n"

        # Strip final separator and whitespace
        return obs_str.strip().strip('---').strip()


    def retrieve(self,
                 nlqs: List[str],
                 output_folder: str,
                 k: int = 0) -> List[List[RetrievalResult]]: # k is not directly used here, but kept for signature consistency
        """
        Overrides the retrieve method to implement the ReAct loop using guidance.
        Collects all unique documents retrieved across all search steps for a query.

        Args:
            nlqs: List of natural language questions.
            output_folder: Path to the folder containing the FAISS index and metadata.
            k: (Unused in this ReAct implementation logic, but part of the base signature).
               The number of results per search step is controlled by `self.k_react_search`.

        Returns:
            A list of lists, where each inner list contains the unique RetrievalResult
            objects collected for the corresponding query across all its ReAct search steps.
        """
        if self.guidance_lm is None:
            print("Error: Guidance LLM not initialized. Cannot perform ReAct retrieval.")
            return [[] for _ in nlqs]

        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"Error: Index ('{index_path}') or metadata ('{metadata_path}') not found.")
            return [[] for _ in nlqs]

        # --- Load FAISS index and metadata ONCE ---
        index_cpu = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            doc_metadata_list: List[Dict[str, Any]] = pickle.load(f)
        num_docs_in_index = index_cpu.ntotal # Use ntotal attribute for count
        if num_docs_in_index != len(doc_metadata_list):
                print("Warning: FAISS index size and metadata list length differ!")

        # --- FAISS Index Resource Management (GPU or CPU) ---
        res = None
        index_to_search = None # Initialize to None
        try:
            if self.device == 'cuda' and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index_to_search = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                 raise RuntimeError("CUDA device requested but not available, or no GPUs found.")
        except Exception as e:
             print(f"Failed to load FAISS index to GPU: {e}. Using CPU index.")
             index_to_search = index_cpu # Fallback to CPU index
             res = None # Ensure res is None if GPU fails
        # --- End FAISS Index Resource Management ---

        # List to store the final accumulated results for each query
        all_final_results: List[List[RetrievalResult]] = []
        # Reset metrics lists for this run
        self.distinct_retrieved_counts_per_query = []
        self.llm_search_calls_per_query = []

        def perform_search_step_for_guidance(keywords: str,
                                             current_collected_results_set: Set[RetrievalResult]
                                             ) -> Tuple[str, int]:
            """
            Performs a single search step using the dense retriever,
            formats the observation, and adds results to the provided set.

            Args:
                keywords: Search terms from the LLM.
                current_collected_results_set: The set to add unique results to.

            Returns:
                A tuple containing:
                - The formatted observation string.
                - The number of search calls made in this step (0 or 1).
            """
            search_calls_increment = 0
            if not keywords or not isinstance(keywords, str) or not keywords.strip():
                print("Warning: Empty keywords received for Search action. Skipping retrieval.")
                return "Observation: No valid keywords provided for search.", search_calls_increment

            search_calls_increment = 1 # We are performing a search now

            # 1. Encode the query
            query_embedding = self.model.encode(
                [keywords.strip()],
                convert_to_numpy=True,
                normalize_embeddings=True # Ensure normalization if model expects it
            )

            # 2. Search the FAISS index
            step_scores, step_indices = index_to_search.search(query_embedding, self.k_react_search)

            step_results: List[RetrievalResult] = []
            # Indices and scores are nested (list of lists), take the first element for the single query
            query_indices = step_indices[0]
            query_scores = step_scores[0]

            # 3. Process results
            for rank, doc_idx in enumerate(query_indices):
                # FAISS can return -1 for invalid indices
                if doc_idx < 0 or doc_idx >= num_docs_in_index:
                    print(f"Warning: Invalid doc_idx {doc_idx} received from FAISS search. Skipping.")
                    continue
                # Basic check for score length mismatch
                if rank >= len(query_scores):
                    print(f"Warning: Score index out of bounds for rank {rank}, doc_idx {doc_idx}. Skipping.")
                    continue

                meta = doc_metadata_list[doc_idx]
                # Ensure '_text' exists and is a non-empty string (or adapt if your key is different)
                text_key = '_text' # Or use self.field_to_index if that's passed/stored
                text = meta.get(text_key)
                if not isinstance(text, str) or not text.strip():
                    print(f"Warning: Skipping doc_idx {doc_idx} due to invalid or empty text content (key: '{text_key}').")
                    continue

                # Create metadata dict excluding the indexed text field
                extra_meta = {k_meta: v_meta for k_meta, v_meta in meta.items() if k_meta != text_key}
                score = float(query_scores[rank]) # Ensure score is float

                # Create the result object (ensure it's hashable)
                result = RetrievalResult(score=score, object=text, metadata=extra_meta)

                # Add to step results and the overall collected set for the query
                step_results.append(result)
                current_collected_results_set.add(result) # Add unique results

            # 4. Format the observation for the LLM
            observation_string = self._format_observation(step_results)
            return observation_string, search_calls_increment

        # --- End search step function definition ---


        # --- Iterate through each natural language query ---
        for nlq_index, nlq in enumerate(tqdm(nlqs, desc="Processing Queries (ReAct with Guidance)")):
            # Set to store unique results collected across ALL search steps for THIS query
            collected_results_this_query: Set[RetrievalResult] = set()
            # Counter for search calls made by the LLM for THIS query
            search_call_counter = {'count': 0} # Use a mutable dict to track calls

            def bound_search_func_with_counter(keywords: str) -> str:
                # Calls the main search logic function defined above
                obs, increment = perform_search_step_for_guidance(
                    keywords,
                    collected_results_this_query # Pass the set for this query
                )
                # Update the counter using the mutable dictionary
                search_call_counter['count'] += increment
                return obs
            # --- End search function closure ---

            executed_state = self.guidance_lm + react_guidance_program(
                # lm is implicitly handled by the '+' operator
                question=nlq,
                max_rounds=self.max_iterations,
                search_func=bound_search_func_with_counter,
                initial_prompt=REACT_PROMPT_TEMPLATE
            )

            if isinstance(executed_state, Callable) and not hasattr(executed_state, 'variables'):
                program_definition = react_guidance_program # Get the function object

                    # Execute the program definition by calling it with the LLM and args
                executed_state = program_definition(
                    lm=self.guidance_lm,
                    question=nlq,
                    max_rounds=self.max_iterations,
                    search_func=bound_search_func_with_counter,
                    initial_prompt=REACT_PROMPT_TEMPLATE
                )

            # --- Store results and metrics for the successfully processed query ---
            # Append collected results (unique across all search steps for this query)
            all_final_results.append(list(collected_results_this_query))
            self.distinct_retrieved_counts_per_query.append(len(collected_results_this_query))
            # Log the total search calls made for this query
            self.llm_search_calls_per_query.append(search_call_counter['count'])

        return all_final_results

    # --- display_metrics remains the same ---
    def display_metrics(self, verbose=True) -> Tuple[float, float]:
        """Prints the collected metrics per query and their averages."""
        avg_distinct = 0.0
        avg_llm_calls = 0.0

        if verbose: print("\n--- Retrieval Metrics (Guidance ReAct) ---")

        query_count = len(self.distinct_retrieved_counts_per_query)
        if query_count > 0:
            avg_distinct = sum(self.distinct_retrieved_counts_per_query) / query_count
            if verbose:
                print(f"Processed {query_count} queries.")
                print("Distinct Retrieved Objects per Query:", self.distinct_retrieved_counts_per_query)
                print(f"Average Distinct Retrieved Objects per Query: {avg_distinct:.2f}")
        elif verbose:
            print("No data for distinct retrieved objects (no queries processed or all failed before completion).")

        # Ensure counts match before calculating average calls
        if len(self.llm_search_calls_per_query) == query_count and query_count > 0:
            avg_llm_calls = sum(self.llm_search_calls_per_query) / query_count
            if verbose:
                print("LLM Search Calls per Query:", self.llm_search_calls_per_query)
                print(f"Average LLM Search Calls per Query: {avg_llm_calls:.2f}")
        elif verbose:
             # Check if the lengths mismatch, indicating a potential issue
            if len(self.llm_search_calls_per_query) != query_count:
                 print(f"Warning: Mismatch between query count ({query_count}) and LLM call count ({len(self.llm_search_calls_per_query)}). Average not calculated.")
            else: # query_count is 0
                 print("No data for LLM search calls.")

        # Add a small separator line if verbose
        if verbose: print("-----------------------------------------")

        return (avg_distinct, avg_llm_calls)


