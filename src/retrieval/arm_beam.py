import os
import pickle
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import torch
import faiss
from sentence_transformers import SentenceTransformer
import textwrap
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM as VLLM_Engine, SamplingParams as VLLM_SamplingParams
import json 
from src.utils.recai_recommender import Trie_link, FastPrefixConstrainedLogitsProcessor
from src.utils.text_precessor import simple_tokenize, generate_ngrams
from nltk.tokenize import word_tokenize 
import random
from collections import Counter
from nltk.corpus import stopwords
from src.utils.mip_solver import MIPSolver

class ARMRetriever(BaseRetriever):
    FAISS_INDEX_FILENAME = "index.faiss"
    FAISS_METADATA_FILENAME = "metadata.pkl"

    def __init__(self,
                vllm_model_path: str,
                faiss_index_path: str,
                bm25_index_path: str,
                ngram_llm_model_path :str,
                embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
                keyword_extraction_beams: int = 5,
                mip_k_select: int = 5,
                compatibility_semantic_weight: float = 0.5,
                compatibility_exact_weight: float = 0.5,
                corpus_ngram_min_len: int = 1, # Renamed from trie_min_n_for_extraction
                corpus_ngram_max_len: int = 3, # Renamed from trie_max_n_for_extraction
                bm25_retriever_instance: Optional[PyseriniBM25Retriever] = None,
                faiss_retriever_instance: Optional[FaissDenseRetriever] = None,
                max_keyword_length: int = 20,
                keyword_alignment :bool = True,
                generate_n_grams:bool = True,
                constrained_id_generation:bool = False,
                expansion_k_compatible: int = 5,
                expansion_steps: int = 1,
                keyword_rephrasing_beams: int = 1,
                vllm_tensor_parallel_size: int = 2,
                vllm_cache_dir: str = "/data/hdd1/vllm_models/",
                vllm_quantization: Optional[str] = None,
                filter_ngrams_against_corpus: bool = True,
                arm_cache_dir: str = "assets/arm/",
                keywords_per_query: int = 10
                ) -> None:

        self.faiss_index_path = faiss_index_path
        self.bm25_index_path = bm25_index_path
        self.embedding_model_name = embedding_model_name
        self.keyword_extraction_beams = keyword_extraction_beams
        self.mip_k_select = mip_k_select
        self.compatibility_semantic_weight = compatibility_semantic_weight
        self.compatibility_exact_weight = compatibility_exact_weight
        self.corpus_ngram_min_len = corpus_ngram_min_len # Renamed
        self.corpus_ngram_max_len = corpus_ngram_max_len
        self.indexed_field: Optional[str] = None
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)


        self.bm25_retriever = bm25_retriever_instance if bm25_retriever_instance else PyseriniBM25Retriever(enable_tqdm=False)
        self.faiss_dense_retriever = faiss_retriever_instance if faiss_retriever_instance else FaissDenseRetriever(model_name_or_path=self.embedding_model_name, enable_tqdm=False)

        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_text_to_idx_map: Optional[Dict[str, int]] = None
        self.max_keyword_length = max_keyword_length
        self.keyword_alignment = keyword_alignment
        self.generate_n_grams = generate_n_grams
        self.current_keywords: List[str] = []
        self.current_ngrams_for_retrieval: List[str] = []

        self.ngram_llm_model_path = ngram_llm_model_path
        self.ngram_llm_tokenizer_path = self.ngram_llm_model_path

        self.ngram_llm_tokenizer: Optional[AutoTokenizer] = None
        self.ngram_llm_model: Optional[AutoModelForCausalLM] = None
        self.ngram_corpus_trie: Optional[Trie_link] = None
        self.ngram_corpus_logits_processor: Optional[FastPrefixConstrainedLogitsProcessor] = None

        self.keyword_rephrasing_beams = keyword_rephrasing_beams

        self.vllm_model_path = vllm_model_path
        self.vllm_engine: Optional[VLLM_Engine] = None
        self.vllm_sampling_params: Optional[VLLM_SamplingParams] = None
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_cache_dir = vllm_cache_dir
        self.vllm_quantization = vllm_quantization
        
        self.valid_corpus_ngrams_set: Set[str] = set() # New: For storing word-based corpus n-grams
        self.filter_ngrams_against_corpus = filter_ngrams_against_corpus # New
        self.arm_cache_dir = arm_cache_dir
        self.stop_words_set = set(stopwords.words('english'))
        self.keywords_per_query = keywords_per_query
        if self.vllm_model_path:
            self._initialize_vllm()
        self.mip_solver = MIPSolver(
            embedding_model=self.embedding_model,
            faiss_dense_retriever=self.faiss_dense_retriever,
            faiss_index_path=self.faiss_index_path,
            device=self.device,
            compatibility_semantic_weight=self.compatibility_semantic_weight,
            compatibility_exact_weight=self.compatibility_exact_weight
        )
        
    def _initialize_vllm(self): 
        tokenizer = AutoTokenizer.from_pretrained(self.vllm_model_path, trust_remote_code=True, cache_dir=self.vllm_cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.vllm_model_path:
            self.vllm_engine = VLLM_Engine(
                model=self.vllm_model_path,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                tokenizer=tokenizer.name_or_path,
                download_dir=self.vllm_cache_dir,
                quantization=self.vllm_quantization,
                gpu_memory_utilization=0.8, 
                max_model_len=2048,
            )
            print("vLLM engine initialized.")
        else:
            print("vLLM model path not provided. Skipping vLLM initialization.")
            
            
                        
    def _initialize_ngram_llm_and_build_trie(self, input_jsonl_path_for_ngrams: str, field_for_ngrams: str):
        if not self.ngram_llm_model_path: return
        self.ngram_llm_tokenizer = AutoTokenizer.from_pretrained(self.ngram_llm_tokenizer_path,cache_dir=self.vllm_cache_dir, trust_remote_code=True)
        self.ngram_llm_model = AutoModelForCausalLM.from_pretrained(self.ngram_llm_model_path,torch_dtype=torch.bfloat16,cache_dir=self.vllm_cache_dir, trust_remote_code=True).to(self.device)
        self.ngram_llm_model.eval()

        self.soi_token_str: str = "<GROUND_SOI>" 
        self.eoi_token_str: str = "<GROUND_EOI>"
        
        special_tokens_dict = {'additional_special_tokens': [self.soi_token_str, self.eoi_token_str]}
        num_added_toks = self.ngram_llm_tokenizer.add_special_tokens(special_tokens_dict)
        
        if num_added_toks > 0:
            self.ngram_llm_model.resize_token_embeddings(len(self.ngram_llm_tokenizer))

        self.ngram_llm_tokenizer.soi_token_id = self.ngram_llm_tokenizer.convert_tokens_to_ids(self.soi_token_str)
        self.ngram_llm_tokenizer.eoi_token_id = self.ngram_llm_tokenizer.convert_tokens_to_ids(self.eoi_token_str)
        
        if self.ngram_llm_tokenizer.pad_token_id is None:
            if self.ngram_llm_tokenizer.eos_token_id is not None:
                self.ngram_llm_tokenizer.pad_token_id = self.ngram_llm_tokenizer.eos_token_id
            else:
                self.ngram_llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.ngram_llm_model.resize_token_embeddings(len(self.ngram_llm_tokenizer))
                self.ngram_llm_tokenizer.pad_token_id = self.ngram_llm_tokenizer.convert_tokens_to_ids('[PAD]')
        os.makedirs(self.arm_cache_dir, exist_ok=True) 
        sanitized_input_basename = os.path.basename(input_jsonl_path_for_ngrams).replace('.', '_').replace(os.sep, '_')
        cache_filename = f"trie_cache_{sanitized_input_basename}_min{self.corpus_ngram_min_len}_max{self.corpus_ngram_max_len}.pkl"
        cache_filepath = os.path.join(self.arm_cache_dir, cache_filename)
        if os.path.exists(cache_filepath): 
            print(f"Loading trie from cache: {cache_filepath}") 
            with open(cache_filepath, 'rb') as f: 
                self.item_prefix_trie = pickle.load(f) 
            self.item_prefix_trie.tokenizer = self.ngram_llm_tokenizer 
            self.logits_processor_instance = FastPrefixConstrainedLogitsProcessor( 
                self.item_prefix_trie.constrain_search_list, 
                num_beams=self.keyword_rephrasing_beams 
            ) 
            print("Trie and logits processor loaded from cache.") 
            return
        
        with open(input_jsonl_path_for_ngrams, 'r', encoding='utf-8') as f:
            data_from_json = [json.loads(line) for line in f if line.strip()]
        
        text_content = ""
        all_text_parts = [] #
        if isinstance(data_from_json, dict): 
            all_text_parts.append(data_from_json.get(field_for_ngrams, "")) 
        elif isinstance(data_from_json, list): 
            for item_in_list in data_from_json: 
                if isinstance(item_in_list, dict) and field_for_ngrams in item_in_list: 
                    all_text_parts.append(str(item_in_list[field_for_ngrams])) 
        text_content = " ".join(all_text_parts) 
                    
        word_tokens = simple_tokenize(text_content)
        self.ngrams_strings = set()
        for ngram_tuple in generate_ngrams(word_tokens, self.corpus_ngram_min_len, self.corpus_ngram_max_len): 
            self.ngrams_strings.add(" ".join(ngram_tuple))
        self.ngram_token_ids_for_trie_build: List[List[int]] = []
        for ngram_str_item in self.ngrams_strings:
            token_ids = self.ngram_llm_tokenizer.encode(ngram_str_item, add_special_tokens=False)
            if token_ids: 
                 self.ngram_token_ids_for_trie_build.append(token_ids)
                 
        self.item_prefix_trie = Trie_link(self.ngram_token_ids_for_trie_build, self.ngram_llm_tokenizer)      
        self.logits_processor_instance = FastPrefixConstrainedLogitsProcessor(
            self.item_prefix_trie.constrain_search_list,
            num_beams=self.keyword_rephrasing_beams 
        )

        with open(cache_filepath, 'wb') as f: 
            pickle.dump(self.item_prefix_trie, f) 
        print(f"Trie saved to cache: {cache_filepath}") 
        

    def _load_faiss_assets(self, faiss_folder_path: str) -> bool:
        index_file = os.path.join(faiss_folder_path, self.FAISS_INDEX_FILENAME)
        metadata_file = os.path.join(faiss_folder_path, self.FAISS_METADATA_FILENAME)

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            self.faiss_index = None
            self.faiss_text_to_idx_map = {}
            return False

        self.faiss_index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            faiss_metadata_list = pickle.load(f)
        
        temp_text_map = {}
        for original_idx, meta_item in enumerate(faiss_metadata_list):
            text_content = meta_item.get('_text') 
            if text_content is not None:
                if text_content not in temp_text_map:
                    temp_text_map[text_content] = original_idx
        
        self.faiss_text_to_idx_map = temp_text_map
        return True

    def index(self,
              input_jsonl_path: str,
              output_folder: str,
              field_to_index: str,
              metadata_fields: List[str]) -> None:
        
        os.makedirs(output_folder, exist_ok=True)
        self.main_output_folder = output_folder
        self.indexed_field = field_to_index

        self.bm25_retriever.index(input_jsonl_path, self.bm25_index_path, field_to_index, metadata_fields)
        self.faiss_dense_retriever.index(input_jsonl_path, self.faiss_index_path, field_to_index, metadata_fields)
        self._load_faiss_assets(self.faiss_index_path)
        if self.generate_n_grams:
            self._initialize_ngram_llm_and_build_trie(input_jsonl_path, self.indexed_field)

    def _ensure_assets_loaded(self, output_folder_from_retrieve: str):
        if self.faiss_index is None or self.faiss_text_to_idx_map is None:
            if self.faiss_index_path and os.path.exists(self.faiss_index_path):
                self._load_faiss_assets(self.faiss_index_path)
            else:
                pass

    def retrieve(self,
                 nlqs: List[str],
                 output_folder: str, 
                 k: int) -> List[List[RetrievalResult]]:
        
        self._ensure_assets_loaded(output_folder)
        
        all_query_results: List[List[RetrievalResult]] = []
        for nlq in tqdm(nlqs, desc="Processing queries", unit="query"):
            self._perform_arm_retrieval_for_query(nlq)
            break
            final_chosen_objects = self.current_llm_selected_objects
            
            query_results: List[RetrievalResult] = []
            
            for obj_data in final_chosen_objects:
                query_results.append(
                    RetrievalResult(
                        score=obj_data.get('relevance_score_R_i', 1.0), 
                        object=obj_data['text'],
                        metadata=obj_data['metadata']
                    )
                )
            
            all_query_results.append(query_results)

        return all_query_results

    
    @staticmethod
    def _get_initial_prompt_string(user_query: str) -> str:
     return textwrap.dedent(f"""
        You are given a user question. Your task is to:
        1. Decompose the user question into contiguous, non-overlapping substrings that can cover different information mentioned in the user question.
        2. For each substring, generate n-grams that are the most relevant to the substring. Prefer lengthy n-grams over short ones.
        3. Based on the generated relevant n-grams, identify and list candidate objects that could be relevant. Each object must be serialized with an `Id`, a `Name`, and `Content`. The `Content` can be a sentence or structured data (e.g., from a table, where `[H]` might denote a header element and `[SEP]` acts as a general separator between the object name and its content, or between parts of the content). The format for each object should be: `Id: 'object_id' Object_Name [SEP] Object_Content`. The object_id must follow the pattern `Title_sentence_index` or `Title_table_index`. Present this list under the heading "Here are the objects that can be relevant to answer the user query:".
        4. From the list of candidate objects you've generated in step 3, you must identify and then list *only the IDs* of the *minimum number* of objects that are *collectively sufficient* to fully answer the user question. Critically evaluate each candidate object: select an object *only if it is essential* for answering a part of the question and its information is not adequately covered by other selected objects. Aim for the smallest possible set of IDs. Present this list of comma-separated IDs under the heading "From the above objects, here are the IDs of those that are enough to answer the query:".
        Your entire response, including all these parts, should end with <>.

        User question: What was the lead single from the album 'Echoes of Tomorrow' by The Cosmic Rays, and in what year was this album released?
        The relevant keywords are lead single | album 'Echoes of Tomorrow' | The Cosmic Rays | year released
        The relevant n-grams are lead single (first single, main track released, promotional single) | album 'Echoes of Tomorrow' (Echoes of Tomorrow LP, record Echoes of Tomorrow, the Echoes of Tomorrow album) | The Cosmic Rays (Cosmic Rays band, group The Cosmic Rays) | year released (release date, album year, publication year)

        Here are the objects that can be relevant to answer the user query:
        Id: 'Echoes of Tomorrow_sentence_0' Echoes of Tomorrow [SEP] The critically acclaimed album 'Echoes of Tomorrow' by The Cosmic Rays featured 'Starlight Serenade' as its lead single, captivating audiences worldwide.
        Id: 'Echoes of Tomorrow_table_0' Echoes of Tomorrow [SEP] [H] Album Details: [H] Artist: The Cosmic Rays , [H] Release Year: 2018 , [H] Genre: Psychedelic Rock , [H] Record Label: Nebula Records
        Id: 'The Cosmic Rays_sentence_0' The Cosmic Rays [SEP] The Cosmic Rays are a band primarily known for their energetic live performances and their earlier hit 'Nebula Blues' from the album 'Celestial Journey'.
        Id: 'Starlight Serenade_sentence_0' Starlight Serenade [SEP] 'Starlight Serenade' is a popular song by The Cosmic Rays, often performed live and was recorded during their 2018 studio sessions.
        Id: 'Celestial Journey_table_0' Celestial Journey [SEP] [H] Album: Celestial Journey , [H] Artist: The Cosmic Rays , [H] Release Year: 2016 , [H] Lead Single: Comet Tail
        Id: 'Music Reviews_sentence_0' Music Reviews [SEP] 'Echoes of Tomorrow' received widespread acclaim, though some critics noted its departure from the band's earlier sound.
        Id: 'The Cosmic Rays_table_0' The Cosmic Rays [SEP] [H] Member: Alex Chen , [H] Role: Vocals, Guitar , [H] Joined: 2015 [SEP] [H] Member: Zara Khan , [H] Role: Drums , [H] Joined: 2015
        Id: 'Echoes of Tomorrow_sentence_1' Echoes of Tomorrow [SEP] The recording sessions for 'Echoes of Tomorrow' took place in early 2018, with the band experimenting with new synthesizers.

        From the above objects, here are the IDs of those that are enough to answer the query:
        Echoes of Tomorrow_sentence_0, Echoes of Tomorrow_table_0
        <>

        User question: {user_query}
        The relevant keywords are: """)
     
    
    def _get_keyword_generation_prompt_vllm(self, user_query: str) -> str:
        return textwrap.dedent(f"""
            From the following user question, extract the most important keywords or keyphrases.
            These keywords should capture the main entities, concepts, and intent.
            Separate each keyword or keyphrase with a pipe symbol (|).

            Example:
            User question: What is the latest album by Taylor Swift and its release date?
            The relevant keywords are: latest album | Taylor Swift | release date

            User question: {user_query}
            The relevant keywords are:""")
    
    
    def rewrite_keyword(self, keyword: str, num_beams: int = 1, max_new_tokens: int = 30,length_penalty: float = 0.0,repetition_penalty: float = 1.0):
        messages = [
            {"role": "system", "content": f"You are an expert assistant. Your task is to rewrite the user's keyword. The rewritten keyword must be one of the allowed phrases (n-grams of 1 to 4 words from a document). **If the original keyword itself is an allowed phrase, please use it directly.** Otherwise, use the shortest, most relevant allowed phrase. Enclose the chosen phrase strictly with {self.soi_token_str} and {self.eoi_token_str} tokens. Example: {self.soi_token_str}an example phrase{self.eoi_token_str}"},
            {"role": "user", "content": f"Rewrite the keyword '{keyword}' using only an allowed phrase from the document."}
        ]
        #lowercase the keyword to ensure consistency
        keyword = keyword.lower().strip()
        prompt_string_from_template = self.ngram_llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_for_model_generation = prompt_string_from_template + self.soi_token_str
        
        input_ids_tensor = self.ngram_llm_tokenizer.encode(prompt_for_model_generation, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids_tensor = self.ngram_llm_model.generate(
                input_ids_tensor,
                logits_processor=[self.logits_processor_instance],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True, 
                eos_token_id=self.ngram_llm_tokenizer.eoi_token_id, 
                pad_token_id=self.ngram_llm_tokenizer.pad_token_id,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                temperature=1.0,
            )
        all_rewritten_keywords = [] 
        all_raw_outputs = [] 
        
        for i in range(output_ids_tensor.shape[0]): 
            generated_ids_only = output_ids_tensor[i][input_ids_tensor.shape[1]:] 
            raw_generated_output_text = self.ngram_llm_tokenizer.decode(generated_ids_only, skip_special_tokens=False)

            final_rewritten_keyword = raw_generated_output_text
            if raw_generated_output_text.endswith(self.eoi_token_str):
                final_rewritten_keyword = raw_generated_output_text[:-len(self.eoi_token_str)].strip()
            elif self.eoi_token_str in raw_generated_output_text:
                final_rewritten_keyword = raw_generated_output_text.split(self.eoi_token_str)[0].strip()
            
            if final_rewritten_keyword.startswith(self.soi_token_str): 
                 final_rewritten_keyword = final_rewritten_keyword[len(self.soi_token_str):].strip()


            all_rewritten_keywords.append(final_rewritten_keyword)
            all_raw_outputs.append(raw_generated_output_text)   

        return all_rewritten_keywords, all_raw_outputs
    
    
    def _perform_arm_retrieval_for_query(self, user_query: str):        
        all_vllm_keywords_set = set()
        keywords_for_rec_lm: List[str] = []
        # 1. Keyword Generation using vLLM with beam search (n=5)
        if self.vllm_engine:
            keyword_prompt = self._get_keyword_generation_prompt_vllm(user_query)
            
            keyword_sampling_params = VLLM_SamplingParams(
                temperature=1.0, 
                max_tokens=150,  
                stop=["\n", "<|eot_id|>", "The relevant n-grams are:"],
                n=self.keyword_extraction_beams
            )
            
            vllm_request_outputs = self.vllm_engine.generate([keyword_prompt], keyword_sampling_params)
            
            for beam_completion_output in vllm_request_outputs[0].outputs: # Iterate through each beam's output
                generated_keywords_text = beam_completion_output.text.strip()
                for stop_seq in keyword_sampling_params.stop:
                    if stop_seq in generated_keywords_text:
                        generated_keywords_text = generated_keywords_text.split(stop_seq, 1)[0].strip()
                #lowercase the generated keywords text
                generated_keywords_text = generated_keywords_text.lower()
                beam_keywords = [k.strip() for k in generated_keywords_text.split('|') if k.strip()]
                all_vllm_keywords_set.update(beam_keywords)
            
            keywords_for_rec_lm = list(all_vllm_keywords_set)
            print(f"vLLM Generated Keywords (deduplicated from 5 beams): {keywords_for_rec_lm}")
        else:
            print("Warning: vLLM engine not initialized. Using full query as a keyword.")
            keywords_for_rec_lm = [user_query] if user_query.strip() else []

        keywords_for_rec_lm = [kw for kw in keywords_for_rec_lm if kw] 

        # 2. N-gram Generation using RecLM-cgen (HF Transformers) for each deduplicated keyword
        all_final_ngrams_for_retrieval_temp: List[str] = []
        if self.generate_n_grams:
            for keyword_text in keywords_for_rec_lm:
                extracted_items, _ = self.rewrite_keyword( # Ignoring extracted_raw as per original
                    keyword_text,
                    num_beams=self.keyword_rephrasing_beams,
                )
                all_final_ngrams_for_retrieval_temp.extend(extracted_items)
        else: 
            all_final_ngrams_for_retrieval_temp = [kw for kw in keywords_for_rec_lm if kw]

        processed_sub_ngrams: List[str] = []
        for phrase in all_final_ngrams_for_retrieval_temp:
            tokens = word_tokenize(phrase.lower())
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words_set]
            
            if not filtered_tokens:
                continue

            for ngram in generate_ngrams(filtered_tokens, self.corpus_ngram_min_len, self.corpus_ngram_max_len):
                processed_sub_ngrams.append(" ".join(ngram))
        
        final_top_ngrams_for_retrieval: List[str] = []
        if processed_sub_ngrams:
            ngram_counts = Counter(processed_sub_ngrams)
            # Sort n-grams by frequency in descending order
            sorted_ngrams_with_counts = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)
            
            # Separate candidates by count > 1 and count == 1
            gt1_candidates = [(ng, count) for ng, count in sorted_ngrams_with_counts if count > 1]
            eq1_candidates = [ng for ng, count in sorted_ngrams_with_counts if count == 1]

            last_added_count = -1
            # Add n-grams with count > 1, keeping all ties for the last added count
            for ngram, count in gt1_candidates:
                if len(final_top_ngrams_for_retrieval) < self.keywords_per_query:
                    final_top_ngrams_for_retrieval.append(ngram)
                    last_added_count = count
                elif count == last_added_count: # Tie with count > 1
                    final_top_ngrams_for_retrieval.append(ngram)
                else: 
                    break 
            
            if len(final_top_ngrams_for_retrieval) < self.keywords_per_query:
                needed = self.keywords_per_query - len(final_top_ngrams_for_retrieval)
                random.shuffle(eq1_candidates) # Shuffle for random selection
                final_top_ngrams_for_retrieval.extend(eq1_candidates[:needed])
        else:
            print("Warning: No sub-ngrams generated after stopword removal and tokenization steps.")

        self.current_keywords = keywords_for_rec_lm
        self.current_ngrams_for_retrieval = final_top_ngrams_for_retrieval + self.current_keywords
        
        print(f"Selected Keywords (displaying first {len(self.current_keywords)} of {len(keywords_for_rec_lm)} unique vLLM keywords): {self.current_keywords}")
        print(f"Final N-grams for retrieval (Top up to {len(self.current_ngrams_for_retrieval)}): {self.current_ngrams_for_retrieval}")
        print(f"--- Keyword and N-gram Generation Complete for query '{user_query}' ---")
        return
    
        
"""
if __name__ == "__main__":
    
    keyword_alignment_choices = [True, False]
    generate_n_grams_choices = [True, False]
    constrained_id_generation_choices = [True, False]
    for keyword_alignment in keyword_alignment_choices:
        for generate_n_grams in generate_n_grams_choices:
            for constrained_id_generation in constrained_id_generation_choices:
                print(f"Running with keyword_alignment={keyword_alignment}, generate_n_grams={generate_n_grams}, constrained_id_generation={constrained_id_generation}")
                arm = ARMRetriever(
                    "/data/hdd1/users/akouk/ARM/ARM/assets/cache/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
                    "assets/feverous/faiss_indexes/dense_row_UAE-Large-V1",
                    "assets/feverous/pyserini_indexes/bm25_row_index",
                    "assets/feverous/trie_indexes",
                    keyword_alignment=keyword_alignment,
                    generate_n_grams=generate_n_grams,
                    constrained_id_generation=constrained_id_generation
                )
                arm.index(
                        "assets/feverous/serialized_output/serialized_row_level.jsonl",
                        output_folder="assets/feverous/arm_indexes/",
                        field_to_index="object",
                        metadata_fields =  ["page_title", "source"]
                    )
                nlqs = [
                    "Aramais Yepiskoposan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991."
                ]
                
                results : List[List[RetrievalResult]] = []
                results = arm.retrieve(
                    nlqs=nlqs,
                    output_folder="assets/feverous/arm_indexes/",
                    k=5
                )
                print(f"Results: {results}")

"""

if __name__ == "__main__":
    arm = ARMRetriever(
        vllm_model_path = "gaunernst/gemma-3-27b-it-int4-awq",
        faiss_index_path = "assets/feverous/faiss_indexes/dense_row_UAE-Large-V1",
        bm25_index_path ="assets/feverous/pyserini_indexes/bm25_row_index",
        ngram_llm_model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        keyword_alignment=False,
        generate_n_grams=True,
        constrained_id_generation=False,
        mip_k_select=10,
        expansion_k_compatible=0,
        expansion_steps=1,
        keyword_rephrasing_beams=10,
        vllm_tensor_parallel_size = 1
    )
    
    arm.index(
        "assets/feverous/serialized_output/serialized_row_level.jsonl",
        output_folder="assets/feverous/arm_indexes/",
        field_to_index="object",
        metadata_fields=["page_title", "source"]
    )
    
    nlqs = [
        "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991."
    ]
    #lowercase all nlqs
    nlqs = [nlq.lower() for nlq in nlqs]
    results : List[List[RetrievalResult]] = []
    results = arm.retrieve(
        nlqs=nlqs,
        output_folder="assets/feverous/arm_indexes/",
        k=5
    )
    print(f"Results: {results}")