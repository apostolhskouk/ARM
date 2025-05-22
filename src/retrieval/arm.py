import os
import pickle
from dataclasses import field
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
import numpy as np
import torch
import faiss
import marisa_trie
import guidance
from guidance import models, gen, select
from sentence_transformers import SentenceTransformer, util
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, HiGHS_CMD, PULP_CBC_CMD, GUROBI_CMD
from pulp.apis.core import PulpSolverError
import re
import textwrap
import time
from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
from src.utils.trie_index import build_marisa_trie_index, simple_tokenize,find_next_word_continuations
from itertools import combinations

class ARMRetriever(BaseRetriever):
    FAISS_INDEX_FILENAME = "index.faiss"
    FAISS_METADATA_FILENAME = "metadata.pkl"

    def __init__(self,
                llm_model_path: str,
                faiss_index_path: str,
                bm25_index_path: str,
                trie_index_path: str,
                embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
                llm_n_gpu_layers: int = -1,
                llm_n_ctx: int = 32768,
                main_output_folder: Optional[str] = None,
                max_keywords_per_query: int = 5,
                max_ngrams_per_keyword: int = 3,
                max_ngram_len_chars: int = 50,
                max_trie_continuations: int = 2000,
                bm25_k_candidates: int = 10,
                mip_k_select: int = 5,
                compatibility_semantic_weight: float = 0.5,
                compatibility_exact_weight: float = 0.5,
                trie_min_n: int = 1,
                trie_max_n: int = 3,
                bm25_retriever_instance: Optional[PyseriniBM25Retriever] = None,
                faiss_retriever_instance: Optional[FaissDenseRetriever] = None,
                verbose_llm: bool = False,
                max_rephrases: int = 5,
                max_keyword_length: int = 20,
                keyword_alignment :bool = True,
                generate_n_grams:bool = True,
                constrained_id_generation:bool = False,
                expansion_k_compatible: int = 6,
                expansion_steps: int = 1) -> None:

        self.llm_model_path = llm_model_path
        self.faiss_index_path = faiss_index_path
        self.bm25_index_path = bm25_index_path
        self.trie_files_directory = trie_index_path
        self.embedding_model_name = embedding_model_name
        self.llm_n_gpu_layers = llm_n_gpu_layers
        self.llm_n_ctx = llm_n_ctx
        self.verbose_llm = verbose_llm

        self.main_output_folder = main_output_folder
        self.max_keywords_per_query = max_keywords_per_query
        self.max_ngrams_per_keyword = max_ngrams_per_keyword
        self.max_ngram_len_chars = max_ngram_len_chars
        self.max_trie_continuations = max_trie_continuations
        self.bm25_k_candidates = bm25_k_candidates
        self.mip_k_select = mip_k_select
        self.compatibility_semantic_weight = compatibility_semantic_weight
        self.compatibility_exact_weight = compatibility_exact_weight
        self.trie_min_n = trie_min_n
        self.trie_max_n = trie_max_n
        self.indexed_field: Optional[str] = None
        self.actual_trie_file_path: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.llm_model = models.LlamaCpp(
            self.llm_model_path,
            n_gpu_layers=self.llm_n_gpu_layers,
            n_ctx=self.llm_n_ctx,
            echo=self.verbose_llm
        )
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)


        self.bm25_retriever = bm25_retriever_instance if bm25_retriever_instance else PyseriniBM25Retriever()
        self.faiss_dense_retriever = faiss_retriever_instance if faiss_retriever_instance else FaissDenseRetriever(model_name_or_path=self.embedding_model_name)

        self.faiss_index: Optional[faiss.Index] = None
        self.faiss_text_to_idx_map: Optional[Dict[str, int]] = None
        self.trie: Optional[marisa_trie.Trie] = None
        self.max_rephrases = max_rephrases
        self.max_keyword_length = max_keyword_length
        self.keyword_alignment = keyword_alignment
        self.generate_n_grams = generate_n_grams
        self.constrained_id_generation = constrained_id_generation
        self.current_llm_selected_objects: List[Dict[str, Any]] = [] 
        self.expansion_k_compatible = expansion_k_compatible
        self.expansion_steps = expansion_steps

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

    def _load_trie(self, trie_path: str) -> bool:
        if not os.path.exists(trie_path):
            self.trie = None
            return False
        self.trie = marisa_trie.Trie()
        self.trie.load(trie_path)
        return True

    def _get_trie_filename(self, field_to_index: str) -> str:
        return f"ngrams_{field_to_index}_{self.trie_min_n}_{self.trie_max_n}.marisa"

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

        trie_filename = self._get_trie_filename(field_to_index)
        self.actual_trie_file_path = os.path.join(self.trie_files_directory, trie_filename)
        if not os.path.exists(self.actual_trie_file_path):
            os.makedirs(self.trie_files_directory, exist_ok=True)
            build_marisa_trie_index(input_jsonl_path, self.actual_trie_file_path, self.trie_min_n, self.trie_max_n, field_to_index)
        self._load_trie(self.actual_trie_file_path)

    def _ensure_assets_loaded(self, output_folder_from_retrieve: str):
        
        if self.main_output_folder != output_folder_from_retrieve and output_folder_from_retrieve is not None:
            self.main_output_folder = output_folder_from_retrieve
            
            if self.indexed_field:
                trie_filename = self._get_trie_filename(self.indexed_field)
                self.actual_trie_file_path = os.path.join(self.trie_files_directory, trie_filename)
            else:
                self.actual_trie_file_path = None
                
            self.faiss_index = None 
            self.faiss_text_to_idx_map = None
            self.trie = None
            self.expansion_compatibility_data = None 

        if self.faiss_index is None or self.faiss_text_to_idx_map is None:
            if self.faiss_index_path and os.path.exists(self.faiss_index_path):
                self._load_faiss_assets(self.faiss_index_path)
            else:
                pass

        if self.trie is None:
            if self.actual_trie_file_path and os.path.exists(self.actual_trie_file_path):
                self._load_trie(self.actual_trie_file_path)

    def retrieve(self,
                 nlqs: List[str],
                 output_folder: str, 
                 k: int) -> List[List[RetrievalResult]]:
        
        self._ensure_assets_loaded(output_folder)
        
        all_query_results: List[List[RetrievalResult]] = []
        for nlq in tqdm(nlqs, desc="Processing queries", unit="query"):
            executed_program = self.llm_model + self._perform_arm_retrieval_for_query(nlq)
            
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
        if emb1 is None or emb2 is None or emb1.nelement() == 0 or emb2.nelement() == 0:
            return 0.0
        
        current_emb1 = emb1.to(self.device) if emb1.device != self.device else emb1
        current_emb2 = emb2.to(self.device) if emb2.device != self.device else emb2

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

    def R(self, original_prefix: str) -> List[str]:
        if self.trie is None: return []
        
        prefix_tokens = simple_tokenize(original_prefix)
        search_prefix = " ".join(prefix_tokens)
        results_set = set()

        if not search_prefix:
            all_keys = self.trie.keys()
            for key in all_keys:
                if ' ' not in key and key: results_set.add(key)
            return sorted(list(results_set))

        continuations = self.trie.keys(prefix=search_prefix)
        len_search_prefix = len(search_prefix)
        for cont in continuations:
            if len(cont) <= len_search_prefix: continue
            remainder = cont[len_search_prefix:]
            if not remainder: continue
            if remainder.startswith(' '):
                next_token = remainder[1:].split(' ', 1)[0]
                if next_token: results_set.add(" " + next_token)
            else:
                if ' ' not in remainder: results_set.add(remainder)
        return sorted(list(results_set))
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
     
    
    @guidance(stateless=False)
    def _perform_arm_retrieval_for_query(self, lm, user_query: str): # lm is the guidance model state
        start_total_time = time.time()
        lm += self._get_initial_prompt_string(user_query)
        
        kw_gen_start_time = time.time()
        lm += gen(name='keywords_str', stop='\n', max_tokens=150)
        keywords_str = lm['keywords_str'].strip()
        keywords = [k.strip() for k in keywords_str.split('|') if k.strip()][:self.max_keywords_per_query]

        lm += "\nThe relevant n-grams are:"
        all_parsed_ngrams_for_bm25_queries = []
        if self.generate_n_grams:
            if self.keyword_alignment:
                for keyword_idx, keyword in enumerate(keywords):
                    if keyword_idx > 0: lm += " |"
                    lm += f" {keyword} ("
                    ngrams_for_this_keyword_list = []
                    current_ngram_being_built_in_python = ""
                    keyword_ngram_generation_finished = False
                    for rephrase_num in range(self.max_rephrases):
                        first_token_var = f"rephrase_{keyword_idx}_{rephrase_num}_first_token"
                        lm += gen(name=first_token_var, max_tokens=1, stop=[')', ','])
                        
                        if lm[first_token_var] == ')':
                            keyword_ngram_generation_finished = True; break
                        if lm[first_token_var] == ',':
                            if current_ngram_being_built_in_python.strip(): ngrams_for_this_keyword_list.append(current_ngram_being_built_in_python.strip())
                            current_ngram_being_built_in_python = ""
                            if rephrase_num < self.max_rephrases -1 : lm += " "
                            break 

                        current_ngram_being_built_in_python = lm[first_token_var]
                        
                        if lm[first_token_var] not in [')', ',']:
                            for token_step in range(self.max_keyword_length -1): 
                                continuations = find_next_word_continuations(current_ngram_being_built_in_python, self.actual_trie_file_path)
                                valid_next_strings = continuations + [')', ', '] 
                                if ' ' not in continuations and (not current_ngram_being_built_in_python or current_ngram_being_built_in_python[-1] != ' '):
                                    valid_next_strings.append(' ') 
                                
                                valid_next_strings = list(set(s for s in valid_next_strings if s and s != '|')) 
                                if not valid_next_strings: valid_next_strings = [')', ', ', ' '] 
                                
                                valid_next_strings = self.configure_list_size(valid_next_strings, self.max_trie_continuations)
                                if not valid_next_strings: valid_next_strings = [')'] 

                                current_token_var = f"rephrase_{keyword_idx}_{rephrase_num}_token_{token_step}"
                                lm += select(options=valid_next_strings, name=current_token_var)
                                selected_token_str = lm[current_token_var]

                                if selected_token_str == ')':
                                    if current_ngram_being_built_in_python.strip(): ngrams_for_this_keyword_list.append(current_ngram_being_built_in_python.strip())
                                    current_ngram_being_built_in_python = ""; keyword_ngram_generation_finished = True; break 
                                elif selected_token_str == ', ':
                                    if current_ngram_being_built_in_python.strip(): ngrams_for_this_keyword_list.append(current_ngram_being_built_in_python.strip())
                                    current_ngram_being_built_in_python = ""
                                    if rephrase_num < self.max_rephrases -1 : lm += " " 
                                    break 
                                else:
                                    current_ngram_being_built_in_python += selected_token_str
                        
                        if keyword_ngram_generation_finished: break 
                    
                    if current_ngram_being_built_in_python.strip(): 
                        ngrams_for_this_keyword_list.append(current_ngram_being_built_in_python.strip())
                    
                    if ngrams_for_this_keyword_list:
                        all_parsed_ngrams_for_bm25_queries.append(" ".join(ngrams_for_this_keyword_list))
                    
                    if not keyword_ngram_generation_finished and keyword_idx < len(keywords) -1 : # This was missing a check for last keyword
                         lm += ")"
                    elif keyword_ngram_generation_finished and keyword_idx < len(keywords) -1 and lm.current_text[-1] != ')': # Ensure ')' if loop broke early
                         lm += ")"

            else: 
                for keyword_idx, keyword in enumerate(keywords):
                    if keyword_idx > 0: lm += " |"
                    lm += f" {keyword} ("
                    
                    max_gen_tokens = (self.max_rephrases * self.max_keyword_length * 2) + self.max_rephrases
                    if max_gen_tokens <= 0: 
                        max_gen_tokens = 50 
                    unconstrained_ngrams_var = f"unconstrained_ngrams_{keyword_idx}"
                    lm += gen(
                        name=unconstrained_ngrams_var, 
                        stop=[')'],  
                        max_tokens=max_gen_tokens,
                        temperature=0.0
                    )
                    
                    generated_ngrams_str = lm[unconstrained_ngrams_var].strip()
                    
                    ngrams_for_this_keyword_list = []
                    if generated_ngrams_str:
                        parsed_ngrams = [s.strip() for s in generated_ngrams_str.split(',') if s.strip()]
                        ngrams_for_this_keyword_list = parsed_ngrams[:self.max_rephrases]

                    if ngrams_for_this_keyword_list:
                        all_parsed_ngrams_for_bm25_queries.append(" ".join(ngrams_for_this_keyword_list))
                    lm += ")"
        else: # not self.generate_n_grams
            if keywords:
                for keyword_idx, keyword_text in enumerate(keywords):
                    if keyword_idx > 0:
                        lm += " |"
                    lm += f" {keyword_text} ({keyword_text})" 
            else:
                lm += " No keywords available to use as n-grams."
            
            all_parsed_ngrams_for_bm25_queries = [kw for kw in keywords if kw.strip()]
        
        retrieved_docs_by_bm25_nested: List[List[RetrievalResult]] = []
        if all_parsed_ngrams_for_bm25_queries and self.bm25_index_path and os.path.exists(self.bm25_index_path):
            retrieved_docs_by_bm25_nested = self.bm25_retriever.retrieve(
                nlqs=all_parsed_ngrams_for_bm25_queries,
                output_folder=self.bm25_index_path, 
                k=self.bm25_k_candidates
            )
        else: 
            retrieved_docs_by_bm25_nested = [[] for _ in all_parsed_ngrams_for_bm25_queries]

        retrieved_docs_by_faiss: List[List[RetrievalResult]] = []
        if self.faiss_index_path and os.path.exists(self.faiss_index_path): 
            retrieved_docs_by_faiss = self.faiss_dense_retriever.retrieve(
                nlqs=[user_query], 
                output_folder=self.faiss_index_path, 
                k=self.bm25_k_candidates 
            )
        else: 
             retrieved_docs_by_faiss = [[]]

        all_bm25_results_flat: List[RetrievalResult] = [item for sublist in retrieved_docs_by_bm25_nested for item in sublist]
        faiss_results_for_query: List[RetrievalResult] = retrieved_docs_by_faiss[0] if retrieved_docs_by_faiss else []
        
        combined_initial_candidates = all_bm25_results_flat + faiss_results_for_query
        
        unique_retrieved_objects_map = {}
        for res_item in combined_initial_candidates:
            current_meta = res_item.metadata if isinstance(res_item.metadata, dict) else {}
            doc_id = current_meta.get('id', res_item.object) # Prefer 'id' from metadata
            
            if doc_id not in unique_retrieved_objects_map:
                unique_retrieved_objects_map[doc_id] = {'doc': res_item, 'max_score': res_item.score, 'metadata': current_meta}
            else: 
                unique_retrieved_objects_map[doc_id]['max_score'] = max(
                    unique_retrieved_objects_map[doc_id]['max_score'], res_item.score
                )
        
        if not unique_retrieved_objects_map:
            lm += "\nHere are the objects that can be relevant to answer the user query:\nNo relevant objects found.\n"
            lm += "From the above objects, here are the IDs of those that are enough to answer the query:\nNo objects to select from.\n<>"
            self.current_llm_selected_objects = []
            return lm

        query_embedding_tensor = self.embedding_model.encode(user_query, convert_to_tensor=True, show_progress_bar=False)
        query_embedding_tensor = query_embedding_tensor.to(self.device)
        if query_embedding_tensor.ndim == 1: query_embedding_tensor = query_embedding_tensor.unsqueeze(0)
        
        texts_for_global_Ri_calc = set()
        for data_item in unique_retrieved_objects_map.values():
            texts_for_global_Ri_calc.add(data_item['doc'].object)

        if self.expansion_compatibility_data and self.expansion_steps > 0:
            potential_expansion_obj_ids_processed = set()
            for data_item in unique_retrieved_objects_map.values():
                base_obj_id = data_item['metadata'].get('id', data_item['doc'].object)
                if base_obj_id in self.expansion_compatibility_data:
                    for comp_obj_info in self.expansion_compatibility_data[base_obj_id]:
                        if comp_obj_info['id'] not in potential_expansion_obj_ids_processed:
                            texts_for_global_Ri_calc.add(comp_obj_info['text'])
                            potential_expansion_obj_ids_processed.add(comp_obj_info['id'])
        
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
        
        initial_mip_input_candidates = []
        for doc_id, data_item in unique_retrieved_objects_map.items():
            doc_object_text = data_item['doc'].object
            relevance_score_R_i = global_relevance_scores_map.get(doc_object_text, 0.0) 

            current_metadata = data_item['metadata']
            source_str = current_metadata.get('source', 'unknown_source')
            source_type = 'table' if 'table' in source_str.lower() else 'passage'
            parsed_content_val = None
            if source_type == 'table':
                parsed_content_val = self._parse_table_object_string(doc_object_text)
            
            initial_mip_input_candidates.append({
                'id': doc_id, # This is the crucial ID
                'text': doc_object_text,
                'source_type': source_type,
                'parsed_content': parsed_content_val,
                'relevance_score_R_i': relevance_score_R_i,
                'metadata': current_metadata
            })

        # Expansion Step
        expanded_candidates_dict = {obj['id']: obj for obj in initial_mip_input_candidates}
        if self.expansion_compatibility_data and self.expansion_steps > 0:
            current_candidates_to_expand_from = list(initial_mip_input_candidates)
            for step in range(self.expansion_steps):
                newly_added_in_this_step_list = []
                for base_obj in current_candidates_to_expand_from:
                    base_obj_id = base_obj['id']
                    if base_obj_id in self.expansion_compatibility_data:
                        for comp_obj_info_template in self.expansion_compatibility_data[base_obj_id]:
                            comp_obj_id = comp_obj_info_template['id']
                            if comp_obj_id not in expanded_candidates_dict:
                                new_candidate = comp_obj_info_template.copy() 
                                new_candidate['relevance_score_R_i'] = global_relevance_scores_map.get(new_candidate['text'], 0.0)
                                expanded_candidates_dict[comp_obj_id] = new_candidate
                                newly_added_in_this_step_list.append(new_candidate)
                
                if not newly_added_in_this_step_list:
                    break 
                current_candidates_to_expand_from = newly_added_in_this_step_list
        
        final_mip_candidates = list(expanded_candidates_dict.values())
        
        selected_objects_by_mip = []
        if final_mip_candidates:
            selected_objects_by_mip = self._solve_mip_object_selection(
                candidate_objects=final_mip_candidates,
                k_select=self.mip_k_select
            )
        
        lm += "\nHere are the objects that can be relevant to answer the user query:\n"
        object_ids_for_llm_selection = []
        mip_objects_map_for_final_selection = {} 

        if selected_objects_by_mip:
            for idx, obj_data_item in enumerate(selected_objects_by_mip):
                meta = obj_data_item.get('metadata', {})
                page_title = meta.get('page_title', f"untitled_{obj_data_item['id']}") # Use obj_data_item['id'] for more uniqueness
                source_info = meta.get('source', f"unknownsrc_{obj_data_item['id']}")
                object_llm_id = f"{page_title}_{source_info}".replace(" ", "_").replace(":", "_").replace("/", "_") # Sanitize more

                object_ids_for_llm_selection.append(object_llm_id)
                mip_objects_map_for_final_selection[object_llm_id] = obj_data_item
                lm += f"Id: '{object_llm_id}' {obj_data_item['text']}\n"
        else:
            lm += "No specific objects were selected by the solver as most relevant.\n"
        
        lm += "\nFrom the above objects, here are the IDs of those that are enough to answer the query:\n"
        
        raw_ids_string_from_llm = ""
        
        if self.constrained_id_generation and object_ids_for_llm_selection: # Add check for non-empty list
            all_possible_id_combinations_strs = []
            for i in range(1, len(object_ids_for_llm_selection) + 1):
                for subset_tuple in combinations(object_ids_for_llm_selection, i):
                    all_possible_id_combinations_strs.append(",".join(subset_tuple))
            if all_possible_id_combinations_strs: # Ensure options are not empty
                 lm += select(options=all_possible_id_combinations_strs, name='final_selected_ids_list')
            else: # Fallback if no combinations (e.g. object_ids_for_llm_selection was empty or had one item and combinations logic failed)
                 lm += gen(name='final_selected_ids_list', stop="\n<>", max_tokens=1024)
        else:
            lm += gen(name='final_selected_ids_list', stop="\n<>", max_tokens=1024) 
            
        raw_ids_string_from_llm = lm['final_selected_ids_list']
        lm += "<>" 

        llm_chosen_final_objects = []
        
        selected_ids_list_from_llm = [id_str.strip().strip("'\"") for id_str in raw_ids_string_from_llm.split(',') if id_str.strip()]
        for llm_id in selected_ids_list_from_llm:
            if llm_id in mip_objects_map_for_final_selection:
                llm_chosen_final_objects.append(mip_objects_map_for_final_selection[llm_id])

        self.current_llm_selected_objects = llm_chosen_final_objects
        
        total_arm_time = time.time() - start_total_time
        print(f"Total _perform_arm_retrieval_for_query time: {total_arm_time:.2f}s")
        print(f"Final llm output : {lm}")
        return lm
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
        constrained_id_generation=True
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