import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/RecAI/RecLM-cgen")))
import pickle
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, util
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, PULP_CBC_CMD, GUROBI_CMD
from pulp.apis.core import PulpSolverError
import re
import textwrap
import time
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from itertools import combinations
from train_utils.processor import FastPrefixConstrainedLogitsProcessor, Trie_link
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_utils.utils import get_ctrl_item 
import json as json_loader

class ARMRetriever(BaseRetriever):
    FAISS_INDEX_FILENAME = "index.faiss"
    FAISS_METADATA_FILENAME = "metadata.pkl"

    def __init__(self,
                llm_model_path: str,
                faiss_index_path: str,
                bm25_index_path: str,
                embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
                llm_n_gpu_layers: int = -1,
                llm_n_ctx: int = 32768,
                main_output_folder: Optional[str] = None,
                max_keywords_per_query: int = 5,
                max_ngrams_per_keyword: int = 3,
                max_ngram_len_chars: int = 50,
                bm25_k_candidates: int = 10,
                mip_k_select: int = 5,
                compatibility_semantic_weight: float = 0.5,
                compatibility_exact_weight: float = 0.5,
                trie_min_n_for_extraction: int = 1, # For extracting n-grams to build Trie_link
                trie_max_n_for_extraction: int = 3, # For extracting n-grams to build Trie_link
                bm25_retriever_instance: Optional[PyseriniBM25Retriever] = None,
                faiss_retriever_instance: Optional[FaissDenseRetriever] = None,
                verbose_llm: bool = False,
                max_rephrases: int = 5,
                max_keyword_length: int = 20,
                keyword_alignment :bool = True,
                generate_n_grams:bool = True,
                constrained_id_generation:bool = False,
                expansion_k_compatible: int = 5,
                expansion_steps: int = 1,
                ngram_llm_model_path: Optional[str] = None, # Path to HF model for n-gram generation
                ngram_llm_tokenizer_path: Optional[str] = None, # Path to its tokenizer
                max_ngrams_generated_per_keyword: int = 5,
                max_new_tokens_for_ngram_gen: int = 100
                ) -> None:

        self.faiss_index_path = faiss_index_path
        self.bm25_index_path = bm25_index_path
        self.embedding_model_name = embedding_model_name
        self.llm_n_gpu_layers = llm_n_gpu_layers
        self.llm_n_ctx = llm_n_ctx
        self.verbose_llm = verbose_llm

        self.main_output_folder = main_output_folder
        self.max_keywords_per_query = max_keywords_per_query
        self.max_ngrams_per_keyword = max_ngrams_per_keyword
        self.max_ngram_len_chars = max_ngram_len_chars
        self.bm25_k_candidates = bm25_k_candidates
        self.mip_k_select = mip_k_select
        self.compatibility_semantic_weight = compatibility_semantic_weight
        self.compatibility_exact_weight = compatibility_exact_weight
        self.trie_min_n_for_extraction = trie_min_n_for_extraction
        self.trie_max_n_for_extraction = trie_max_n_for_extraction
        self.indexed_field: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)


        self.bm25_retriever = bm25_retriever_instance if bm25_retriever_instance else PyseriniBM25Retriever(enable_tqdm=False)
        self.faiss_dense_retriever = faiss_retriever_instance if faiss_retriever_instance else FaissDenseRetriever(model_name_or_path=self.embedding_model_name, enable_tqdm=False)

        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_text_to_idx_map: Optional[Dict[str, int]] = None
        self.max_rephrases = max_rephrases
        self.max_keyword_length = max_keyword_length
        self.keyword_alignment = keyword_alignment
        self.generate_n_grams = generate_n_grams
        self.constrained_id_generation = constrained_id_generation
        self.current_llm_selected_objects: List[Dict[str, Any]] = [] 
        self.expansion_k_compatible = expansion_k_compatible
        self.expansion_steps = expansion_steps

        self.ngram_llm_model_path = ngram_llm_model_path if ngram_llm_model_path else llm_model_path # Default to main LLM if not specified
        self.ngram_llm_tokenizer_path = ngram_llm_tokenizer_path if ngram_llm_tokenizer_path else self.ngram_llm_model_path

        self.ngram_llm_tokenizer: Optional[AutoTokenizer] = None
        self.ngram_llm_model: Optional[AutoModelForCausalLM] = None
        self.ngram_trie: Optional[Trie_link] = None
        self.ngram_logits_processor: Optional[FastPrefixConstrainedLogitsProcessor] = None
        self.max_ngrams_generated_per_keyword = max_ngrams_generated_per_keyword
        self.max_new_tokens_for_ngram_gen = max_new_tokens_for_ngram_gen

        if self.generate_n_grams:
            self._initialize_ngram_llm_components()
            
    
    def _initialize_ngram_llm_components(self):
        self.ngram_llm_tokenizer = AutoTokenizer.from_pretrained(self.ngram_llm_tokenizer_path)
        self.ngram_llm_model = AutoModelForCausalLM.from_pretrained(
            self.ngram_llm_model_path,
            torch_dtype=torch.bfloat16, # Or float16, adjust as per model capability
            device_map=self.device # Or "auto"
        )
        self.ngram_llm_model.eval()

        # Setup special tokens for RecLM-cgen style generation
        special_tokens_dict = {'additional_special_tokens': ['<SOI>', '<EOI>']}
        self.ngram_llm_tokenizer.add_special_tokens(special_tokens_dict)
        
        self.ngram_llm_tokenizer.soi_token = "<SOI>"
        self.ngram_llm_tokenizer.eoi_token = "<EOI>"
        self.ngram_llm_tokenizer.soi_token_id = self.ngram_llm_tokenizer.convert_tokens_to_ids("<SOI>")
        self.ngram_llm_tokenizer.eoi_token_id = self.ngram_llm_tokenizer.convert_tokens_to_ids("<EOI>")

        if self.ngram_llm_tokenizer.pad_token is None:
            # Common practice for Llama models if pad_token is not set
            # For Llama-3, specific pad token might be needed as in RecLM-cgen
            if "llama-3" in self.ngram_llm_tokenizer_path.lower():
                 self.ngram_llm_tokenizer.pad_token = '<|reserved_special_token_250|>'
                 self.ngram_llm_tokenizer.pad_token_id = 128255
                 self.ngram_llm_tokenizer.eos_token = "<|eot_id|>" # Llama-3 specific
                 self.ngram_llm_tokenizer.eos_token_id = self.ngram_llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            else: # General fallback
                self.ngram_llm_tokenizer.pad_token = self.ngram_llm_tokenizer.eos_token
                self.ngram_llm_tokenizer.pad_token_id = self.ngram_llm_tokenizer.eos_token_id
        
        self.ngram_llm_model.resize_token_embeddings(len(self.ngram_llm_tokenizer))
        
        # Initialize new embeddings (optional, from RecLM-cgen's BaseModel)
        with torch.no_grad():
            descriptions = {"<SOI>": "start of an item", "<EOI>": "end of an item"}
            for token_str_val, desc_str_val in descriptions.items():
                token_id_val = self.ngram_llm_tokenizer.convert_tokens_to_ids(token_str_val)
                # Check if token_id_val is within the original vocab size before trying to access its embedding
                if token_id_val < self.ngram_llm_model.get_input_embeddings().weight.size(0) - len(special_tokens_dict['additional_special_tokens']):
                    tokenized_desc = self.ngram_llm_tokenizer.tokenize(desc_str_val)
                    tokenized_ids_desc = self.ngram_llm_tokenizer.convert_tokens_to_ids(tokenized_desc)
                    if not tokenized_ids_desc: continue # Skip if description tokenizes to empty
                    
                    valid_tokenized_ids_desc = [tid for tid in tokenized_ids_desc if tid < self.ngram_llm_model.get_input_embeddings().weight.size(0) - len(special_tokens_dict['additional_special_tokens'])]
                    if not valid_tokenized_ids_desc: continue

                    new_embedding_val = self.ngram_llm_model.get_input_embeddings().weight[valid_tokenized_ids_desc].mean(axis=0)
                    self.ngram_llm_model.get_input_embeddings().weight[token_id_val, :] = new_embedding_val.clone().detach()


    def _extract_ngrams_for_trie(self, input_jsonl_path: str, field_to_index: str) -> List[str]:
        all_ngrams = set()
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting n-grams for Trie_link construction"):
                record = json_loader.loads(line)
                text_content = record.get(field_to_index)
                if isinstance(text_content, str) and text_content.strip():
                    # Simple tokenization, can be improved (e.g., using self.ngram_llm_tokenizer.tokenize)
                    tokens = text_content.lower().split() 
                    for n_val in range(self.trie_min_n_for_extraction, self.trie_max_n_for_extraction + 1):
                        for i in range(len(tokens) - n_val + 1):
                            ngram = " ".join(tokens[i:i+n_val])
                            if ngram: # Ensure non-empty n-gram
                                all_ngrams.add(ngram)
        return list(all_ngrams)



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
        if self.generate_n_grams and self.ngram_llm_tokenizer and self.ngram_llm_model:
            print("Building Trie_link for n-gram generation...")
            unique_ngrams_for_trie = self._extract_ngrams_for_trie(input_jsonl_path, field_to_index)
            if unique_ngrams_for_trie:
                # Filter out empty strings that might result from tokenization if an n-gram was just whitespace
                unique_ngrams_for_trie = [ng for ng in unique_ngrams_for_trie if ng.strip()]
                if unique_ngrams_for_trie: # Check again after stripping
                    tokenized_ngrams_for_trie = self.ngram_llm_tokenizer(
                        unique_ngrams_for_trie, add_special_tokens=False, truncation=True, max_length=32 # Max length for an n-gram
                    ).input_ids
                    # Filter out empty tokenizations
                    tokenized_ngrams_for_trie = [t_ng for t_ng in tokenized_ngrams_for_trie if t_ng]

                    if tokenized_ngrams_for_trie:
                        self.ngram_trie = Trie_link(tokenized_ngrams_for_trie, self.ngram_llm_tokenizer)
                        self.ngram_logits_processor = FastPrefixConstrainedLogitsProcessor(
                            self.ngram_trie.constrain_search_list, num_beams=1
                        )
                        print(f"Trie_link built with {len(unique_ngrams_for_trie)} unique n-grams.")
                    else:
                        print("Warning: No valid tokenized n-grams to build Trie_link.")
                        self.ngram_trie = None
                        self.ngram_logits_processor = None
                else:
                    print("Warning: No non-empty n-grams extracted to build Trie_link.")
                    self.ngram_trie = None
                    self.ngram_logits_processor = None

            else:
                print("Warning: No n-grams extracted to build Trie_link.")
                self.ngram_trie = None
                self.ngram_logits_processor = None
        else:
            if self.generate_n_grams:
                print("Warning: N-gram LLM components not initialized. Skipping Trie_link build.")


    def _ensure_assets_loaded(self, output_folder_from_retrieve: str):

        if self.main_output_folder != output_folder_from_retrieve and output_folder_from_retrieve is not None:
            self.main_output_folder = output_folder_from_retrieve
            self.faiss_index = None 
            self.faiss_text_to_idx_map = None
            self.expansion_compatibility_data = None 

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
    def _parse_table_object_string(table_str: str) -> dict:
        parts = table_str.split(" [SEP] ", 1)
        table_name = parts[0]
        columns_data = {}
        if len(parts) > 1 and parts[1]:
            content_part = parts[1].strip()
            if content_part.startswith("[H] "):
                current_attributes_str = content_part[4:]
            else:
                current_attributes_str = content_part
            attributes = current_attributes_str.split(" , [H] ")
            for attr_str in attributes:
                header_content_split = attr_str.split(":", 1)
                if len(header_content_split) == 2:
                    header, content = header_content_split[0].strip(), header_content_split[1].strip()
                    if header not in columns_data:
                        columns_data[header] = []
                    columns_data[header].append(content)
        return {"name": table_name, "columns": columns_data}

    @staticmethod
    def _get_tokens_cached(text: str, cache: Dict[str, Set[str]]) -> Set[str]:
        if text in cache:
            return cache[text]
        processed_text = text.lower()
        processed_text = re.sub(r'[^\w\s]', '', processed_text)
        tokens = set(processed_text.split())
        cache[text] = tokens
        return tokens

    def _get_tokens(self, text: str) -> Set[str]:
        return ARMRetriever._get_tokens_cached(text, {})

    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2: return 1.0
        if not set1 or not set2: return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _overlap_coefficient(set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2: return 1.0
        if not set1 or not set2: return 0.0
        intersection = len(set1.intersection(set2))
        min_len = min(len(set1), len(set2))
        return intersection / min_len if min_len > 0 else 0.0

    def _get_semantic_similarity_from_embeddings(self, emb1: Optional[torch.Tensor], emb2: Optional[torch.Tensor]) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        
        if isinstance(emb1, np.ndarray):
            if emb1.size == 0: return 0.0
            emb1_tensor = torch.tensor(emb1, dtype=torch.float32)
        elif isinstance(emb1, torch.Tensor):
            if emb1.nelement() == 0: return 0.0
            emb1_tensor = emb1
        else: 
            return 0.0

        if isinstance(emb2, np.ndarray):
            if emb2.size == 0: return 0.0 
            emb2_tensor = torch.tensor(emb2, dtype=torch.float32)
        elif isinstance(emb2, torch.Tensor):
            if emb2.nelement() == 0: return 0.0
            emb2_tensor = emb2
        else: 
            return 0.0
        
        current_emb1 = emb1_tensor.to(self.device) if emb1_tensor.device != self.device else emb1_tensor
        current_emb2 = emb2_tensor.to(self.device) if emb2_tensor.device != self.device else emb2_tensor

        if current_emb1.ndim == 1: current_emb1 = current_emb1.unsqueeze(0)
        if current_emb2.ndim == 1: current_emb2 = current_emb2.unsqueeze(0)
        
        return util.cos_sim(current_emb1, current_emb2).item()
    
    def _get_semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True, show_progress_bar=False)
        return self._get_semantic_similarity_from_embeddings(emb1, emb2)
    def _calculate_table_table_compatibility_optimized(
        self, table1_parsed: dict, table2_parsed: dict, 
        embeddings_map: Dict[str, torch.Tensor], tokens_cache: Dict[str, Set[str]]
    ) -> float:
        max_col_pair_compatibility = 0.0
        table1_headers = list(table1_parsed["columns"].keys())
        table2_headers = list(table2_parsed["columns"].keys())
        if not table1_headers or not table2_headers: return 0.0

        for h1 in table1_headers:
            emb_h1 = embeddings_map.get(h1)
            if emb_h1 is None: continue
            for h2 in table2_headers:
                emb_h2 = embeddings_map.get(h2)
                if emb_h2 is None: continue

                header_sem_sim = self._get_semantic_similarity_from_embeddings(emb_h1, emb_h2)
                
                col1_value_tokens = set()
                for v_val in table1_parsed["columns"].get(h1, []): 
                    col1_value_tokens.update(ARMRetriever._get_tokens_cached(v_val, tokens_cache))
                col2_value_tokens = set()
                for v_val in table2_parsed["columns"].get(h2, []): 
                    col2_value_tokens.update(ARMRetriever._get_tokens_cached(v_val, tokens_cache))
                
                exact_val_sim = self._jaccard_similarity(col1_value_tokens, col2_value_tokens)
                col_pair_comp = (self.compatibility_semantic_weight * header_sem_sim +
                                 self.compatibility_exact_weight * exact_val_sim)
                if col_pair_comp > max_col_pair_compatibility:
                    max_col_pair_compatibility = col_pair_comp
        return max_col_pair_compatibility

    def _calculate_table_passage_compatibility_optimized(
        self, table_parsed: dict, passage_text: str, passage_embedding: Optional[torch.Tensor], 
        embeddings_map: Dict[str, torch.Tensor], tokens_cache: Dict[str, Set[str]]
    ) -> float:
        max_cell_sentence_compatibility = 0.0
        if not table_parsed["columns"] or passage_embedding is None: return 0.0
        
        passage_tokens = ARMRetriever._get_tokens_cached(passage_text, tokens_cache)

        for header, cells in table_parsed["columns"].items():
            for cell_content in cells:
                emb_cell = embeddings_map.get(cell_content)
                if emb_cell is None: continue

                cell_sem_sim = self._get_semantic_similarity_from_embeddings(emb_cell, passage_embedding)
                
                cell_tokens = ARMRetriever._get_tokens_cached(cell_content, tokens_cache)
                exact_val_sim = self._overlap_coefficient(cell_tokens, passage_tokens)
                cell_sentence_comp = (self.compatibility_semantic_weight * cell_sem_sim +
                                      self.compatibility_exact_weight * exact_val_sim)
                if cell_sentence_comp > max_cell_sentence_compatibility:
                    max_cell_sentence_compatibility = cell_sentence_comp
        return max_cell_sentence_compatibility

    def _calculate_passage_passage_compatibility_optimized(
        self, passage1_text: str, passage1_embedding: Optional[torch.Tensor], 
        passage2_text: str, passage2_embedding: Optional[torch.Tensor],
        tokens_cache: Dict[str, Set[str]]
    ) -> float:
        if passage1_embedding is None or passage2_embedding is None: return 0.0
        
        sem_sim = self._get_semantic_similarity_from_embeddings(passage1_embedding, passage2_embedding)
        
        passage1_tokens = ARMRetriever._get_tokens_cached(passage1_text, tokens_cache)
        passage2_tokens = ARMRetriever._get_tokens_cached(passage2_text, tokens_cache)
        
        exact_val_sim = self._overlap_coefficient(passage1_tokens, passage2_tokens)
        return (self.compatibility_semantic_weight * sem_sim +
                self.compatibility_exact_weight * exact_val_sim)
        
        
    def _calculate_table_table_compatibility(self, table1_parsed: dict, table2_parsed: dict) -> float:
        max_col_pair_compatibility = 0.0
        table1_headers = list(table1_parsed["columns"].keys())
        table2_headers = list(table2_parsed["columns"].keys())
        if not table1_headers or not table2_headers: return 0.0
        for h1 in table1_headers:
            for h2 in table2_headers:
                header_sem_sim = self._get_semantic_similarity(h1, h2)
                col1_value_tokens = set()
                for v_val in table1_parsed["columns"].get(h1, []): col1_value_tokens.update(self._get_tokens(v_val))
                col2_value_tokens = set()
                for v_val in table2_parsed["columns"].get(h2, []): col2_value_tokens.update(self._get_tokens(v_val))
                exact_val_sim = self._jaccard_similarity(col1_value_tokens, col2_value_tokens)
                col_pair_comp = (self.compatibility_semantic_weight * header_sem_sim +
                                 self.compatibility_exact_weight * exact_val_sim)
                if col_pair_comp > max_col_pair_compatibility:
                    max_col_pair_compatibility = col_pair_comp
        return max_col_pair_compatibility

    def _calculate_table_passage_compatibility(self, table_parsed: dict, passage_text: str) -> float:
        max_cell_sentence_compatibility = 0.0
        passage_tokens = self._get_tokens(passage_text)
        if not table_parsed["columns"]: return 0.0
        for header, cells in table_parsed["columns"].items():
            for cell_content in cells:
                cell_sem_sim = self._get_semantic_similarity(cell_content, passage_text)
                cell_tokens = self._get_tokens(cell_content)
                exact_val_sim = self._overlap_coefficient(cell_tokens, passage_tokens)
                cell_sentence_comp = (self.compatibility_semantic_weight * cell_sem_sim +
                                      self.compatibility_exact_weight * exact_val_sim)
                if cell_sentence_comp > max_cell_sentence_compatibility:
                    max_cell_sentence_compatibility = cell_sentence_comp
        return max_cell_sentence_compatibility

    def _calculate_passage_passage_compatibility(self, passage1_text: str, passage2_text: str) -> float:
        sem_sim = self._get_semantic_similarity(passage1_text, passage2_text)
        passage1_tokens = self._get_tokens(passage1_text)
        passage2_tokens = self._get_tokens(passage2_text)
        exact_val_sim = self._overlap_coefficient(passage1_tokens, passage2_tokens)
        return (self.compatibility_semantic_weight * sem_sim +
                self.compatibility_exact_weight * exact_val_sim)

    def _solve_mip_object_selection(self, candidate_objects: list[dict], k_select: int) -> list[dict]:
        num_objects = len(candidate_objects)
        if num_objects == 0 or k_select <= 0: return []
        
        effective_k_select = min(k_select, num_objects)
        
        table_indices = [i for i, obj in enumerate(candidate_objects) if obj['source_type'] == 'table']
        passage_indices = [i for i, obj in enumerate(candidate_objects) if obj['source_type'] == 'passage']
        
        min_tables_needed = 0
        min_passages_needed = 0
        if effective_k_select > 0:
            num_available_tables = len(table_indices)
            num_available_passages = len(passage_indices)
            if num_available_tables + num_available_passages < effective_k_select:
                 effective_k_select = num_available_tables + num_available_passages
            ideal_tables = effective_k_select // 2
            ideal_passages = effective_k_select - ideal_tables
            min_tables_needed = min(ideal_tables, num_available_tables)
            min_passages_needed = min(ideal_passages, num_available_passages)
            if min_tables_needed + min_passages_needed < effective_k_select:
                if min_tables_needed < ideal_tables and num_available_tables > min_tables_needed:
                    min_tables_needed = min(num_available_tables, min_tables_needed + (effective_k_select - (min_tables_needed + min_passages_needed)))
                if min_passages_needed < ideal_passages and num_available_passages > min_passages_needed:
                     min_passages_needed = min(num_available_passages, min_passages_needed + (effective_k_select - (min_tables_needed + min_passages_needed)))
        
        mip_setup_start_time = time.time()
        R = [obj['relevance_score_R_i'] for obj in candidate_objects]
        C = np.zeros((num_objects, num_objects))

        all_texts_for_embedding = set()
        for obj in candidate_objects:
            if obj['source_type'] == 'table' and obj['parsed_content']:
                for header in obj['parsed_content']['columns'].keys():
                    all_texts_for_embedding.add(header)
                for cell_list in obj['parsed_content']['columns'].values():
                    for cell_content in cell_list:
                        all_texts_for_embedding.add(cell_content)
            elif obj['source_type'] == 'passage':
                all_texts_for_embedding.add(obj['text'])
        
        unique_texts_list = list(all_texts_for_embedding)
        text_to_embedding_map: Dict[str, torch.Tensor] = {}

        if unique_texts_list:
            all_embeddings = self.faiss_dense_retriever.faiss_encode(unique_texts_list,self.faiss_index_path)
            for text, embedding_tensor in zip(unique_texts_list, all_embeddings):
                text_to_embedding_map[text] = embedding_tensor

        tokens_cache_for_mip: Dict[str, Set[str]] = {}

        tt_time = tp_time = pp_time = 0.0
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                obj_i = candidate_objects[i]; obj_j = candidate_objects[j]
                comp_val = 0.0
                
                if obj_i['source_type'] == 'table' and obj_j['source_type'] == 'table':
                    if obj_i['parsed_content'] and obj_j['parsed_content']:
                        t0 = time.time()
                        comp_val = self._calculate_table_table_compatibility_optimized(
                            obj_i['parsed_content'], obj_j['parsed_content'], 
                            text_to_embedding_map, tokens_cache_for_mip)
                        tt_time += time.time() - t0
                elif obj_i['source_type'] == 'table' and obj_j['source_type'] == 'passage':
                    if obj_i['parsed_content']:
                        t0 = time.time()
                        passage_j_embedding = text_to_embedding_map.get(obj_j['text'])
                        comp_val = self._calculate_table_passage_compatibility_optimized(
                            obj_i['parsed_content'], obj_j['text'], passage_j_embedding, 
                            text_to_embedding_map, tokens_cache_for_mip)
                        tp_time += time.time() - t0
                elif obj_i['source_type'] == 'passage' and obj_j['source_type'] == 'table':
                    if obj_j['parsed_content']:
                        t0 = time.time()
                        passage_i_embedding = text_to_embedding_map.get(obj_i['text'])
                        comp_val = self._calculate_table_passage_compatibility_optimized(
                            obj_j['parsed_content'], obj_i['text'], passage_i_embedding, 
                            text_to_embedding_map, tokens_cache_for_mip)
                        tp_time += time.time() - t0
                elif obj_i['source_type'] == 'passage' and obj_j['source_type'] == 'passage':
                    t0 = time.time()
                    passage_i_embedding = text_to_embedding_map.get(obj_i['text'])
                    passage_j_embedding = text_to_embedding_map.get(obj_j['text'])
                    comp_val = self._calculate_passage_passage_compatibility_optimized(
                        obj_i['text'], passage_i_embedding, 
                        obj_j['text'], passage_j_embedding,
                        tokens_cache_for_mip)
                    pp_time += time.time() - t0
                C[i, j] = C[j, i] = comp_val
        
        prob = LpProblem("ObjectSelectionMIP", LpMaximize)
        b = [LpVariable(f"b_{i}", cat=LpBinary) for i in range(num_objects)]
        c_vars = {}
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                if C[i][j] > 1e-6: 
                     c_vars[(i,j)] = LpVariable(f"c_{i}_{j}", cat=LpBinary)

        objective = lpSum(R[i] * b[i] for i in range(num_objects))
        if c_vars: 
            objective += lpSum(C[i][j] * c_vars[(i,j)] for i in range(num_objects) for j in range(i+1, num_objects) if (i,j) in c_vars)
        prob += objective

        prob += lpSum(b[i] for i in range(num_objects)) == effective_k_select, "TotalObjectsConstraint"
        if c_vars:
            if effective_k_select > 1:
                 prob += lpSum(c_vars[(i,j)] for i in range(num_objects) for j in range(i+1, num_objects) if (i,j) in c_vars) <= 2 * (effective_k_select - 1), "MaxConnectionsConstraint"
            elif effective_k_select == 1: 
                 prob += lpSum(c_vars[(i,j)] for i in range(num_objects) for j in range(i+1, num_objects) if (i,j) in c_vars) == 0, "NoConnectionsForK1Constraint"
            
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    if (i,j) in c_vars: 
                        prob += 2 * c_vars[(i,j)] <= b[i] + b[j], f"ConnectionIntegrity_{i}_{j}"
        
        if table_indices and min_tables_needed > 0:
            prob += lpSum(b[i] for i in table_indices) >= min_tables_needed, "MinTablesConstraint"
        if passage_indices and min_passages_needed > 0:
            prob += lpSum(b[i] for i in passage_indices) >= min_passages_needed, "MinPassagesConstraint"

        solver_used = None
        # Try Gurobi first
        try :
            prob.solve(GUROBI_CMD(msg=0, timeLimit=30)) 
            if prob.status == 1:
                solver_used = "Gurobi"
            else:
                print(f"Gurobi solver ran but did not find an optimal solution (status: {prob.status}).")
        except PulpSolverError as e:
            pass
        if solver_used is None and prob.status != 1:
            prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
            if prob.status == 1:
                solver_used = "CBC"
            else:
                print(f"CBC solver ran but did not find an optimal solution (status: {prob.status}).")
        
        selected_objects_indices = []
        if prob.status == 1: 
            selected_objects_indices = [i for i, var_b in enumerate(b) if var_b.value() is not None and var_b.value() > 0.5]
        
        return [candidate_objects[i] for i in selected_objects_indices]
    
    @staticmethod
    def configure_list_size(string_list, n):
        if not string_list:
            return []
        if len(string_list) <= n:
            return string_list
        max_s_len = 0
        for s_item in string_list:
            if len(s_item) > max_s_len:
                max_s_len = len(s_item)
        optimal_trunc_len = 0
        low = 0
        high = max_s_len

        while low <= high:
            mid_len = low + (high - low) // 2
            
            truncated_set = {s_val[:mid_len] for s_val in string_list}
            
            if len(truncated_set) <= n:
                optimal_trunc_len = mid_len
                low = mid_len + 1 
            else:
                high = mid_len - 1
                
        final_truncated_set = {s_val[:optimal_trunc_len] for s_val in string_list}
        return list(final_truncated_set)

    
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
     
    
    def _perform_arm_retrieval_for_query(self, lm, user_query: str): # lm is the guidance model state
        pass
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
        "/data/hdd1/users/akouk/ARM/ARM/assets/cache/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "assets/feverous/faiss_indexes/dense_row_UAE-Large-V1",
        "assets/feverous/pyserini_indexes/bm25_row_index",
        "assets/feverous/trie_indexes",
        keyword_alignment=True,
        generate_n_grams=False,
        constrained_id_generation=True,
        mip_k_select = 10,
        expansion_k_compatible=0,
        expansion_steps=1
    )
    
    arm.index(
        "assets/feverous/serialized_output/serialized_row_level.jsonl",
        output_folder="assets/feverous/arm_indexes/",
        field_to_index="object",
        metadata_fields=["page_title", "source"]
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