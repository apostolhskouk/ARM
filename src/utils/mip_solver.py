from typing import Dict, Optional, Set, Any
import numpy as np
import torch
from sentence_transformers import util
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, PULP_CBC_CMD, GUROBI_CMD
from pulp.apis.core import PulpSolverError
import re
import time
from src.retrieval.dense import FaissDenseRetriever


class MIPSolver:
    def __init__(self,
                 embedding_model: Any,  
                 faiss_dense_retriever: FaissDenseRetriever, 
                 faiss_index_path: str,
                 device: str, 
                 compatibility_semantic_weight: float,
                 compatibility_exact_weight: float ):
        
        self.embedding_model = embedding_model
        self.faiss_dense_retriever = faiss_dense_retriever
        self.faiss_index_path = faiss_index_path
        self.device = torch.device(device)  
        self.compatibility_semantic_weight = compatibility_semantic_weight
        self.compatibility_exact_weight = compatibility_exact_weight
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
        return MIPSolver._get_tokens_cached(text, {})

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
                    col1_value_tokens.update(MIPSolver._get_tokens_cached(v_val, tokens_cache))
                col2_value_tokens = set()
                for v_val in table2_parsed["columns"].get(h2, []): 
                    col2_value_tokens.update(MIPSolver._get_tokens_cached(v_val, tokens_cache))
                
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
        
        passage_tokens = MIPSolver._get_tokens_cached(passage_text, tokens_cache)

        for header, cells in table_parsed["columns"].items():
            for cell_content in cells:
                emb_cell = embeddings_map.get(cell_content)
                if emb_cell is None: continue

                cell_sem_sim = self._get_semantic_similarity_from_embeddings(emb_cell, passage_embedding)
                
                cell_tokens = MIPSolver._get_tokens_cached(cell_content, tokens_cache)
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
        
        passage1_tokens = MIPSolver._get_tokens_cached(passage1_text, tokens_cache)
        passage2_tokens = MIPSolver._get_tokens_cached(passage2_text, tokens_cache)
        
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
