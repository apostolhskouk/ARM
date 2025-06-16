import os
import pickle
from typing import List, Dict, Optional, Set, Tuple
from tqdm import tqdm
import torch
import faiss
from sentence_transformers import SentenceTransformer, util
import textwrap
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM as VLLM_Engine, SamplingParams as VLLM_SamplingParams
import json 
from src.utils.recai_recommender import Trie_link, FastPrefixConstrainedLogitsProcessor, Node
from src.utils.text_precessor import simple_tokenize, generate_ngrams
from nltk.tokenize import word_tokenize 
import random
from collections import Counter
from nltk.corpus import stopwords
from src.utils.mip_solver import MIPSolver
import numpy as np
import time
from mxbai_rerank import MxbaiRerankV2
from collections import defaultdict
class ARMRetriever(BaseRetriever):
    FAISS_INDEX_FILENAME = "index.faiss"
    FAISS_METADATA_FILENAME = "metadata.pkl"

    def __init__(self,
                vllm_model_path: str,
                ngram_llm_model_path :str,
                embedding_model_name: str = "BAAI/bge-m3",
                keyword_extraction_beams: int = 5,
                mip_k_select: int = 30,
                compatibility_semantic_weight: float = 0.99,
                compatibility_exact_weight: float = 0.01,
                corpus_ngram_min_len: int = 1, 
                corpus_ngram_max_len: int = 3, 
                generate_n_grams:bool = False,
                expansion_k_compatible: int = 5,
                expansion_steps: int = 1,
                keyword_rephrasing_beams: int = 3,
                vllm_tensor_parallel_size: int = 2,
                vllm_cache_dir: str = "/data/hdd1/vllm_models/",
                vllm_quantization: Optional[str] = None,
                arm_cache_dir: str = "assets/arm/",
                keywords_per_query: int = 20,
                alignment_retrieval_k: int = 20,
                use_reranker_instead_of_mip: bool = False,
                final_llm_selection_beams: int = 1,
                dense_instead_of_sparse : bool = True
                ) -> None:

        self.embedding_model_name = embedding_model_name
        self.keyword_extraction_beams = keyword_extraction_beams
        self.mip_k_select = mip_k_select
        self.compatibility_semantic_weight = compatibility_semantic_weight
        self.compatibility_exact_weight = compatibility_exact_weight
        self.corpus_ngram_min_len = corpus_ngram_min_len 
        self.corpus_ngram_max_len = corpus_ngram_max_len
        self.indexed_field: Optional[str] = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)


        self.bm25_retriever = PyseriniBM25Retriever(enable_tqdm=False)
        self.faiss_dense_retriever = FaissDenseRetriever(model_name_or_path=self.embedding_model_name, enable_tqdm=False)

        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_text_to_idx_map: Optional[Dict[str, int]] = None
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
        
        self.valid_corpus_ngrams_set: Set[str] = set()
        self.arm_cache_dir = arm_cache_dir
        self.stop_words_set = set(stopwords.words('english'))
        self.keywords_per_query = keywords_per_query
        if self.vllm_model_path:
            self._initialize_vllm()
        self.alignment_retrieval_k = alignment_retrieval_k
        self.expansion_k_compatible = expansion_k_compatible
        self.expansion_steps = expansion_steps
        self.all_queries_processed: int = 0
        self.all_items_returned: int = 0
        self.use_reranker_instead_of_mip = use_reranker_instead_of_mip
        self.reranker = None
        self.final_llm_selection_beams = final_llm_selection_beams
        self.dense_instead_of_sparse = dense_instead_of_sparse
    @staticmethod
    def _convert_dicts_to_retrieval_results(objects: List[Dict]) -> List[RetrievalResult]:
        results = []
        if not objects:
            return results
        for obj_data in objects:
            results.append(
                RetrievalResult(
                    score=obj_data.get('relevance_score_R_i', 1.0),
                    object=obj_data['text'],
                    metadata=obj_data['metadata']
                )
            )
        return results

    @staticmethod
    def parse_qwen3_output(raw_output_text):
        """
        Parses the raw output from Qwen3 (potentially with thinking tags)
        to extract the final content.
        """
        think_end_tag = "</think>"
        # Find the *last* occurrence of the closing tag
        last_tag_index = raw_output_text.rfind(think_end_tag)

        if last_tag_index != -1:
            # If tag found, content starts after the tag
            content_start_index = last_tag_index + len(think_end_tag)
            final_content = raw_output_text[content_start_index:]
        else:
            # If no tag found, assume the whole output is the content
            final_content = raw_output_text

        # Strip leading/trailing whitespace and newlines
        return final_content.strip()
    
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
                gpu_memory_utilization=0.72, 
                max_model_len=22048,
            )
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
            with open(cache_filepath, 'rb') as f: 
                self.item_prefix_trie = pickle.load(f) 
            self.item_prefix_trie.tokenizer = self.ngram_llm_tokenizer 
            self.logits_processor_instance = FastPrefixConstrainedLogitsProcessor( 
                self.item_prefix_trie.constrain_search_list, 
                num_beams=self.keyword_rephrasing_beams 
            ) 
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
              output_folder: List[str],
              field_to_index: str,
              metadata_fields: List[str]) -> None:
        self.bm25_index_path = output_folder[0]
        self.faiss_index_path = output_folder[1]
        self.indexed_field = field_to_index
        self.bm25_retriever.index(input_jsonl_path, self.bm25_index_path, field_to_index, metadata_fields)
        self.faiss_dense_retriever.index(input_jsonl_path, self.faiss_index_path, field_to_index, metadata_fields)
        self._load_faiss_assets(self.faiss_index_path)
        if self.generate_n_grams:
            self._initialize_ngram_llm_and_build_trie(input_jsonl_path, self.indexed_field)


    def retrieve(self,
                 nlqs: List[str],
                 output_folder: List[str], 
                 k: int) -> List[List[RetrievalResult]]:
        self.bm25_index_path = output_folder[0]
        self.faiss_index_path = output_folder[1]
        if self.use_reranker_instead_of_mip:
            if self.reranker is None:
                self.reranker = MxbaiRerankV2('mixedbread-ai/mxbai-rerank-large-v2',device='cuda:1')
        self.mip_solver = MIPSolver(
            embedding_model=self.embedding_model,
            faiss_dense_retriever=self.faiss_dense_retriever,
            faiss_index_path=self.faiss_index_path,
            device=self.device,
            compatibility_semantic_weight=self.compatibility_semantic_weight,
            compatibility_exact_weight=self.compatibility_exact_weight
        )
        all_query_results: List[List[RetrievalResult]] = []
        for nlq in tqdm(nlqs, desc="Processing queries", unit="query"):
            final_chosen_objects = self._perform_arm_retrieval_for_query(nlq)
            
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
    def _get_final_selection_prompt(user_query: str, keywords_string: str, serialized_objects: str) -> str:
        return textwrap.dedent(f"""
You are a meticulous research assistant. You are given a user question and a pre-assembled list of candidate objects. Your primary goal is to identify and select **every object** that is relevant to the user's question. Your selection should be **comprehensive** to avoid missing any potentially useful information.

**Guiding Principles:**
1.  **Prioritize Recall over Precision:** It is much better to include a moderately relevant object than to accidentally exclude a crucial one. When in doubt, include the object.
2.  **Value Context:** Select objects that provide useful background or context, even if they don't contain the direct answer. Information that helps build a complete picture is valuable.
3.  **Assume Pre-filtered Candidates:** The list of objects has already been filtered for potential relevance. Your task is to perform the final, careful selection from this list.

**Formatting Rules:**
-   List your chosen IDs separated by commas (e.g., `id1,id2,id3`).
-   Close your response with the terminator `<>` on a new line.

**Example Task:**

User question: What was the lead single from the album 'Echoes of Tomorrow' by The Cosmic Rays, and in what year was this album released?

Here are the objects that can be relevant to answer the user query:
Id: 'sentence_1' : Echoes of Tomorrow [SEP] The critically acclaimed album 'Echoes of Tomorrow' by The Cosmic Rays featured 'Starlight Serenade' as its lead single, captivating audiences worldwide.
Id: 'sentence_2' : The Cosmic Rays [SEP] The Cosmic Rays are a band primarily known for their energetic live performances and their earlier hit 'Nebula Blues' from the album 'Celestial Journey'.
Id: 'table_1' : Echoes of Tomorrow [SEP] [H] Album Details: [H] Artist: The Cosmic Rays , [H] Release Year: 2018 , [H] Genre: Psychedelic Rock , [H] Record Label: Nebula Records
Id: 'sentence_3' : Starlight Serenade [SEP] 'Starlight Serenade' is a popular song by The Cosmic Rays, often performed live and was recorded during their 2018 studio sessions.
Id: 'table_2' : Celestial Journey [SEP] [H] Album: Celestial Journey , [H] Artist: The Cosmic Rays , [H] Release Year: 2016 , [H] Lead Single: Comet Tail
Id: 'sentence_4' : Music Reviews [SEP] 'Echoes of Tomorrow' received widespread acclaim, though some critics noted its departure from the band's earlier sound.
Id: 'table_3' : The Cosmic Rays [SEP] [H] Member: Alex Chen , [H] Role: Vocals, Guitar , [H] Joined: 2015 [SEP] [H] Member: Zara Khan , [H] Role: Drums , [H] Joined: 2015
Id: 'sentence_5' : Echoes of Tomorrow [SEP] The recording sessions for 'Echoes of Tomorrow' took place in early 2018, with the band experimenting with new synthesizers.

Based on the principles of comprehensive selection, here are the IDs of all relevant and contextual objects:
sentence_0,table_0,sentence_2,sentence_3,sentence_4,table_2
<>

**Now it’s your turn.**

User question: {user_query}  

Here are the objects that can be relevant to answer the user query:
{serialized_objects}

From the above objects, here are the IDs of those that are enough to answer the query:
            """)
        
    def _get_keyword_generation_prompt_vllm(self, user_query: str) -> str:
        return textwrap.dedent(f"""
            From the following user question, extract the most important keywords or keyphrases.
            These keywords should capture the main entities, concepts, and intent.
            Separate each keyword or keyphrase with a pipe symbol (|).

            Example:
            User question: What is the latest album by Taylor Swift and its release date?
            The relevant keywords are: latest album | Taylor Swift | release date

            User question: {user_query}
            /no_think
            The relevant keywords are:""")
    
    
    def rewrite_keyword(self, keyword: str, num_beams: int = 1, max_new_tokens: int = 30,repetition_penalty: float = 1.0):
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
                eos_token_id=self.ngram_llm_tokenizer.eoi_token_id, 
                pad_token_id=self.ngram_llm_tokenizer.pad_token_id,
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
        total_start_time = time.time()
        
        # Step 1: Keyword Generation
        step1_start = time.time()
        all_vllm_keywords_set = set()
        keywords_for_rec_lm: List[str] = []
        if self.vllm_engine:
            keyword_prompt = self._get_keyword_generation_prompt_vllm(user_query)
            
            keyword_sampling_params = VLLM_SamplingParams(
                temperature=0.7, 
                max_tokens=150,  
                stop=["\n", "<|eot_id|>", "The relevant n-grams are:"],
                n=self.keyword_extraction_beams
            )
            
            vllm_request_outputs = self.vllm_engine.generate([keyword_prompt], keyword_sampling_params,use_tqdm=False)
            
            for beam_completion_output in vllm_request_outputs[0].outputs:
                generated_keywords_text = self.parse_qwen3_output(beam_completion_output.text) #in case we use the thinking models of qwen
                for stop_seq in keyword_sampling_params.stop:
                    if stop_seq in generated_keywords_text:
                        generated_keywords_text = generated_keywords_text.split(stop_seq, 1)[0].strip()
                generated_keywords_text = generated_keywords_text.lower()
                beam_keywords = [k.strip() for k in generated_keywords_text.split('|') if k.strip()]
                all_vllm_keywords_set.update(beam_keywords)
            
            keywords_for_rec_lm = list(all_vllm_keywords_set)
        else:
            print("Warning: vLLM engine not initialized. Using full query as a keyword.")
            keywords_for_rec_lm = [user_query] if user_query.strip() else []

        keywords_for_rec_lm = [kw for kw in keywords_for_rec_lm if kw] 
        # Step 2: N-gram Generation
        all_final_ngrams_for_retrieval_temp: List[str] = []
        if self.generate_n_grams:
            for keyword_text in keywords_for_rec_lm:
                extracted_items, _ = self.rewrite_keyword(
                    keyword_text,
                    num_beams=self.keyword_rephrasing_beams,
                )
                all_final_ngrams_for_retrieval_temp.extend(extracted_items)
        else: 
            all_final_ngrams_for_retrieval_temp = [kw for kw in keywords_for_rec_lm if kw]
            
        # Step 3: N-gram Processing
        processed_sub_ngrams: List[str] = []
        for phrase in all_final_ngrams_for_retrieval_temp:
            tokens = word_tokenize(phrase.lower())
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words_set]
            
            if not filtered_tokens:
                continue

            for ngram in generate_ngrams(filtered_tokens, self.corpus_ngram_min_len, self.corpus_ngram_max_len):
                processed_sub_ngrams.append(" ".join(ngram))
        
        # Step 4: Top N-gram Selection
        final_top_ngrams_for_retrieval: List[str] = []
        if processed_sub_ngrams:
            ngram_counts = Counter(processed_sub_ngrams)
            sorted_ngrams_with_counts = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)
            
            gt1_candidates = [(ng, count) for ng, count in sorted_ngrams_with_counts if count > 1]
            eq1_candidates = [ng for ng, count in sorted_ngrams_with_counts if count == 1]

            last_added_count = -1
            for ngram, count in gt1_candidates:
                if len(final_top_ngrams_for_retrieval) < self.keywords_per_query:
                    final_top_ngrams_for_retrieval.append(ngram)
                    last_added_count = count
                elif count == last_added_count:
                    final_top_ngrams_for_retrieval.append(ngram)
                else: 
                    break 
            
            if len(final_top_ngrams_for_retrieval) < self.keywords_per_query:
                needed = self.keywords_per_query - len(final_top_ngrams_for_retrieval)
                random.shuffle(eq1_candidates)
                final_top_ngrams_for_retrieval.extend(eq1_candidates[:needed])
        else:
            print("Warning: No sub-ngrams generated after stopword removal and tokenization steps.")

        self.current_keywords = keywords_for_rec_lm
        self.current_ngrams_for_retrieval = final_top_ngrams_for_retrieval + self.current_keywords

        # Step 5: Initial Retrieval (BM25 + FAISS)
        retrieved_docs_by_bm25_nested: List[List[RetrievalResult]] = []
        if self.dense_instead_of_sparse :
            retrieved_docs_by_bm25_nested = self.faiss_dense_retriever.retrieve(
                nlqs=self.current_ngrams_for_retrieval,
                output_folder=self.faiss_index_path, 
                k=self.alignment_retrieval_k
            )
        else:
            retrieved_docs_by_bm25_nested = self.bm25_retriever.retrieve(
                nlqs=self.current_ngrams_for_retrieval, 
                output_folder=self.bm25_index_path, 
                k=self.alignment_retrieval_k
            )
        retrieved_docs_by_faiss: List[List[RetrievalResult]] = []
        retrieved_docs_by_faiss = self.faiss_dense_retriever.retrieve(
            nlqs=[user_query], 
            output_folder=self.faiss_index_path, 
            k=self.alignment_retrieval_k * 4
        )
        all_bm25_results_flat: List[RetrievalResult] = [item for sublist in retrieved_docs_by_bm25_nested for item in sublist]
        faiss_results_for_query: List[RetrievalResult] = retrieved_docs_by_faiss[0] if retrieved_docs_by_faiss else []
        combined_initial_candidates = all_bm25_results_flat + faiss_results_for_query

        # Step 6: Deduplication
        unique_retrieved_objects_map = {}
        for res_item in combined_initial_candidates:
            current_meta = res_item.metadata if isinstance(res_item.metadata, dict) else {}
            doc_id = res_item.object
            if doc_id not in unique_retrieved_objects_map:
                unique_retrieved_objects_map[doc_id] = {'doc': res_item, 'max_score': res_item.score, 'metadata': current_meta}
            else:
                unique_retrieved_objects_map[doc_id]['max_score'] = max(
                    unique_retrieved_objects_map[doc_id]['max_score'], res_item.score
                )

        # Step 7: Global Relevance Scoring
        query_embedding_tensor = self.embedding_model.encode(user_query, convert_to_tensor=True, show_progress_bar=False)
        query_embedding_tensor = query_embedding_tensor.to(self.device)
        if query_embedding_tensor.ndim == 1: query_embedding_tensor = query_embedding_tensor.unsqueeze(0)
        texts_for_global_Ri_calc = set()
        for data_item in unique_retrieved_objects_map.values():
            texts_for_global_Ri_calc.add(data_item['doc'].object)
        unique_texts_list_for_global_Ri = list(texts_for_global_Ri_calc)
        global_relevance_scores_map: Dict[str, float] = {}
        texts_needing_fresh_encoding_for_global_Ri = []
        if self.faiss_index and self.faiss_text_to_idx_map and self.faiss_index.ntotal > 0:
            candidate_faiss_indices_global = []
            map_faiss_idx_to_text_global = {}
            for text_content in unique_texts_list_for_global_Ri:
                if text_content in self.faiss_text_to_idx_map:
                    original_faiss_idx = self.faiss_text_to_idx_map[text_content]
                    if 0 <= original_faiss_idx < self.faiss_index.ntotal:
                        candidate_faiss_indices_global.append(original_faiss_idx)
                        map_faiss_idx_to_text_global[original_faiss_idx] = text_content
                    else:
                        texts_needing_fresh_encoding_for_global_Ri.append(text_content)
                else:
                    texts_needing_fresh_encoding_for_global_Ri.append(text_content)
            if candidate_faiss_indices_global:
                reconstructed_embs_np_global = np.array([self.faiss_index.reconstruct(idx) for idx in candidate_faiss_indices_global]).astype('float32')
                reconstructed_embs_tensor_global = torch.tensor(reconstructed_embs_np_global).to(self.device)
                if query_embedding_tensor.nelement() > 0 and reconstructed_embs_tensor_global.nelement() > 0:
                    sim_scores_tensor_global = util.cos_sim(query_embedding_tensor, reconstructed_embs_tensor_global)[0]
                    for i, original_faiss_idx in enumerate(candidate_faiss_indices_global):
                        obj_text = map_faiss_idx_to_text_global[original_faiss_idx]
                        score = (sim_scores_tensor_global[i].item() + 1) / 2.0
                        global_relevance_scores_map[obj_text] = score
        else:
            texts_needing_fresh_encoding_for_global_Ri.extend(unique_texts_list_for_global_Ri)

        unique_texts_needing_fresh_Ri = list(set(texts_needing_fresh_encoding_for_global_Ri))
        if unique_texts_needing_fresh_Ri:
            obj_embeddings_tensor_global = self.embedding_model.encode(
                unique_texts_needing_fresh_Ri, convert_to_tensor=True, show_progress_bar=False, batch_size=128
            )
            obj_embeddings_tensor_global = obj_embeddings_tensor_global.to(self.device)

            if query_embedding_tensor.nelement() > 0 and obj_embeddings_tensor_global.nelement() > 0:
                sim_scores_tensor_global_fresh = util.cos_sim(query_embedding_tensor, obj_embeddings_tensor_global)[0]
                for i, text_content in enumerate(unique_texts_needing_fresh_Ri):
                    score = (sim_scores_tensor_global_fresh[i].item() + 1) / 2.0
                    global_relevance_scores_map[text_content] = score

        # Step 8: MIP Input Preparation
        initial_mip_input_candidates = []
        for doc_id, data_item in unique_retrieved_objects_map.items():
            doc_object_text = data_item['doc'].object
            relevance_score_R_i = global_relevance_scores_map.get(doc_object_text, 0.0)

            current_metadata = data_item['metadata']
            source_str = current_metadata.get('source', 'unknown_source')
            source_type = 'table' if 'table' in source_str.lower() else 'passage'
            parsed_content_val = None
            if source_type == 'table':
                parsed_content_val = self.mip_solver._parse_table_object_string(doc_object_text)

            initial_mip_input_candidates.append({
                'id': doc_id,
                'text': doc_object_text,
                'source_type': source_type,
                'parsed_content': parsed_content_val,
                'relevance_score_R_i': relevance_score_R_i,
                'metadata': current_metadata
            })
        
        # Step 9: Expansion
        expanded_candidates_dict = {obj['id']: obj for obj in initial_mip_input_candidates}
        if self.expansion_steps > 0:
            current_candidates_to_expand_from = list(initial_mip_input_candidates)
            for step in range(self.expansion_steps):
                newly_added_in_this_step_list = []
                expansion_queries = [base_obj['text'] for base_obj in current_candidates_to_expand_from if base_obj['text']]
                if self.dense_instead_of_sparse:
                    bm25_expansion_results_nested = self.faiss_dense_retriever.retrieve(
                        nlqs=expansion_queries,
                        output_folder=self.faiss_index_path,
                        k=self.expansion_k_compatible
                    )
                else:
                    bm25_expansion_results_nested = self.bm25_retriever.retrieve(
                        nlqs=expansion_queries,
                        output_folder=self.bm25_index_path,
                        k=self.expansion_k_compatible
                    )

                for i, base_obj_text_queried in enumerate(expansion_queries):
                    bm25_retrieved_for_this_base_obj = bm25_expansion_results_nested[i]
                    
                    for bm25_res_item in bm25_retrieved_for_this_base_obj:
                        expanded_content_as_key = bm25_res_item.object 

                        if expanded_content_as_key not in expanded_candidates_dict:
                            expanded_relevance_score_R_i = 0.0
                            if expanded_content_as_key in global_relevance_scores_map:
                                expanded_relevance_score_R_i = global_relevance_scores_map[expanded_content_as_key]
                            else:
                                if query_embedding_tensor.nelement() > 0:
                                    expanded_item_emb_np = self.faiss_dense_retriever.faiss_encode(
                                        [expanded_content_as_key], 
                                        self.faiss_index_path 
                                    )
                                    expanded_item_emb = torch.tensor(expanded_item_emb_np).to(self.device)
                                    if expanded_item_emb.ndim == 1: expanded_item_emb = expanded_item_emb.unsqueeze(0)
                                    if expanded_item_emb.nelement() > 0:
                                        sim_score = util.cos_sim(query_embedding_tensor, expanded_item_emb).item()
                                        expanded_relevance_score_R_i = (sim_score + 1) / 2.0
                                global_relevance_scores_map[expanded_content_as_key] = expanded_relevance_score_R_i

                            temp_meta = bm25_res_item.metadata if isinstance(bm25_res_item.metadata, dict) else {}
                            temp_source_str = temp_meta.get('source', 'unknown_source_bm25_expanded')
                            temp_source_type = 'table' if 'table' in temp_source_str.lower() else 'passage'
                            temp_parsed_content = None
                            if temp_source_type == 'table':
                                temp_parsed_content = self.mip_solver._parse_table_object_string(expanded_content_as_key)

                            mip_formatted_new_candidate = {
                                'id': expanded_content_as_key,
                                'text': expanded_content_as_key,
                                'source_type': temp_source_type,
                                'parsed_content': temp_parsed_content,
                                'relevance_score_R_i': expanded_relevance_score_R_i,
                                'metadata': temp_meta
                            }
                            expanded_candidates_dict[expanded_content_as_key] = mip_formatted_new_candidate
                            newly_added_in_this_step_list.append(mip_formatted_new_candidate)
                
                if not newly_added_in_this_step_list:
                    break
                current_candidates_to_expand_from = newly_added_in_this_step_list
        final_mip_candidates = list(expanded_candidates_dict.values())

        selected_objects_by_mip = []
        if self.use_reranker_instead_of_mip:
            if final_mip_candidates:
                docs_to_rerank = [obj['text'] for obj in final_mip_candidates]
                text_to_candidate_map = {obj['text']: obj for obj in final_mip_candidates}
                try:
                    reranked_results = self.reranker.rank(
                        query=user_query,
                        documents=docs_to_rerank,
                        top_k=self.mip_k_select,  
                        return_documents=True,
                        batch_size=2
                    )
                    for item in reranked_results:
                        doc_text = item.document
                        if doc_text in text_to_candidate_map:
                            original_object = text_to_candidate_map[doc_text]
                            original_object['relevance_score_R_i'] = item.score
                            selected_objects_by_mip.append(original_object)
                        else:
                            print(f"Warning: Reranked document text not found in candidate map: {doc_text}")
                except: #fallback to mip
                    print("Reranker failed, falling back to MIP solver.")
                    selected_objects_by_mip = self.mip_solver._solve_mip_object_selection(
                        candidate_objects=final_mip_candidates,
                        k_select=self.mip_k_select
                    )
        else:
            if final_mip_candidates:
                selected_objects_by_mip = self.mip_solver._solve_mip_object_selection(
                    candidate_objects=final_mip_candidates,
                    k_select=self.mip_k_select
                )
        
        llm_id_to_original_data_map = {}
        serialized_objects = ""
        sentence_counter = 1
        table_counter = 1

        # 1. Group objects by their source (page_title, source_info)
        grouped_objects = defaultdict(list)
        for obj_data_item in selected_objects_by_mip:
            meta = obj_data_item.get('metadata', {})
            # Use a tuple of (page_title, source) as the unique key for a group
            group_key = (
                meta.get('page_title', f"untitled_{obj_data_item['id']}"),
                meta.get('source', f"unknownsrc_{obj_data_item['id']}")
            )
            grouped_objects[group_key].append(obj_data_item)

        # 2. Iterate over the groups to create a single, combined entry for each
        for group_key, object_group in grouped_objects.items():
            if not object_group:
                continue

            # Use the first object in the group to determine the type (table/passage)
            first_obj = object_group[0]
            source_type = first_obj.get('source_type')

            # Determine the simple LLM ID for the entire group
            simple_llm_id = None
            if source_type == 'table':
                simple_llm_id = f"table_{table_counter}"
                table_counter += 1
            elif source_type == 'passage':
                simple_llm_id = f"sentence_{sentence_counter}"
                sentence_counter += 1
            else:
                print(f"Warning: Group with key '{group_key}' has unknown source_type '{source_type}'. Skipping.")
                continue

            # 3. Combine the text of all objects in the group
            combined_text = ""
            if source_type == 'table':
                # For tables, take the title from the first row and combine all row data.
                # This avoids repeating the title part.
                title_part = first_obj['text'].split(' [SEP] ', 1)[0]
                row_parts = [obj['text'].split(' [SEP] ', 1)[1] for obj in object_group if ' [SEP] ' in obj['text']]
                combined_text = f"{title_part} [SEP] {' [SEP] '.join(row_parts)}"
            else:  # 'passage'
                # For sentences, simply join them with a space to form a coherent paragraph.
                combined_text = " ".join([obj['text'] for obj in object_group])

            # 4. Store mapping and build the serialized string for the prompt
            # The map now points from the simple ID to the *list* of original objects
            llm_id_to_original_data_map[simple_llm_id] = object_group
            serialized_objects += f"Id: '{simple_llm_id}' : {combined_text}\n"
            
        serialized_keywords = " | ".join(self.current_keywords)
        selection_prompt = self._get_final_selection_prompt(
            user_query=user_query,
            keywords_string=serialized_keywords,
            serialized_objects=serialized_objects 
        )
        selection_sampling_params = VLLM_SamplingParams(
            n=self.final_llm_selection_beams,
            temperature=0.7,          
            max_tokens=256,
            stop=["<>"],             
        )
        
        vllm_outputs = self.vllm_engine.generate([selection_prompt], selection_sampling_params, use_tqdm=False)
        
        all_selected_ids_from_beams = []
        # Iterate over the outputs from all beams
        for beam_completion_output in vllm_outputs[0].outputs:
            raw_llm_output = self.parse_qwen3_output(beam_completion_output.text)
            # Parse the simple IDs (e.g., "sentence_1", "table_3") from this beam's output
            ids_from_beam = self.parse_selected_ids(raw_llm_output)
            all_selected_ids_from_beams.extend(ids_from_beam)
            
        # Deduplicate the list of IDs while preserving the order of first appearance
        llm_selected_simple_ids = list(dict.fromkeys(all_selected_ids_from_beams))
        final_selected_objects_list = []
        # llm_selected_simple_ids now contains the simple IDs (e.g., "sentence_1", "table_1")
        # that the LLM selected.
        for simple_id_str in llm_selected_simple_ids:
            # Use the new map to look up the original object data using the simple ID.
            if simple_id_str in llm_id_to_original_data_map:
                # Use extend since the map now contains a LIST of original objects for each ID
                final_selected_objects_list.extend(llm_id_to_original_data_map[simple_id_str])
            else:
                # This can happen if LLM hallucinates an ID or if parse_selected_ids is imperfect
                # or if an object was skipped during simple ID generation but LLM still tried to pick it.
                print(f"Warning: LLM-selected simple ID '{simple_id_str}' not found in the mapping. "
                      f"This ID might have been hallucinated by the LLM or was not mappable. Skipping this ID.")
                
        self.all_items_returned += len(final_selected_objects_list)
        self.all_queries_processed += 1
        return final_selected_objects_list # final_mip_candidates, selected_objects_by_mip
    
    def display_metrics(self, verbose=True) -> Tuple[float, float]:
        avg_llm_calls = 1.0
        avg_distinct = self.all_items_returned / self.all_queries_processed
        return avg_distinct, avg_llm_calls
    @staticmethod
    def parse_selected_ids(raw_output: str) -> List[str]:
        # Remove the terminator, newlines & whitespace
        cleaned = raw_output.replace("<>", "").strip()
        # If the model ever writes the heading, drop everything before the first newline
        if "\n" in cleaned:
            cleaned = cleaned.split("\n")[-1].strip()
        # Split on commas and strip whitespace
        return [id_.strip() for id_ in cleaned.split(",") if id_.strip()]
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



if __name__ == "__main__":
    arm = ARMRetriever(
        vllm_model_path = "gaunernst/gemma-3-27b-it-int4-awq",
        faiss_index_path = "assets/feverous/faiss_indexes/dense_row_UAE-Large-V1",
        bm25_index_path ="assets/feverous/pyserini_indexes/bm25_row_index",
        ngram_llm_model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        generate_n_grams=True,
        mip_k_select=30,
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
    
"""