from src.retrieval.base import BaseRetriever, RetrievalResult
import json
import os
import pickle
from typing import List, Dict, Any
from tqdm.auto import tqdm
import torch 
from pylate import models as pylate_models
from pylate import indexes as pylate_indexes
from pylate import retrieve as pylate_retrieve
import math


class PylateColbertRetriever(BaseRetriever):
    INDEX_SUBFOLDER = "pylate_voyager_index"
    INDEX_NAME_INTERNAL = "colbert_voyager_idx" # Internal name for Voyager index files
    METADATA_FILENAME = "pylate_colbert_metadata.pkl"

    def __init__(self,
                 model_name_or_path: str = "lightonai/Reason-ModernColBERT",
                 enable_tqdm: bool = True,
                 batch_size_encode: int = 16):

        self.model_name_or_path = model_name_or_path
        self.enable_tqdm = enable_tqdm
        self.batch_size_encode = batch_size_encode

        self.pylate_model = pylate_models.ColBERT(
            model_name_or_path=self.model_name_or_path,
        )

    def index(self,
              input_jsonl_path: str,
              output_folder: str,
              field_to_index: str,
              metadata_fields: List[str]) -> None:

        os.makedirs(output_folder, exist_ok=True)
        voyager_index_base_path = os.path.join(output_folder, self.INDEX_SUBFOLDER)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)
        
        voyager_key_file_path = os.path.join(
            voyager_index_base_path,
            self.INDEX_NAME_INTERNAL,
            "index.voyager"
        )
        if os.path.exists(metadata_path) and os.path.exists(voyager_key_file_path):
            print(f"PyLate ColBERT index and metadata appear to exist in '{output_folder}', skipping.")
            return

        texts, current_metadata_list = [], []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                data = json.loads(line.strip())
                text_content = data[field_to_index]
                texts.append(text_content)
                entry = {'_text': text_content, '_original_id': str(line_number)} 
                for field in metadata_fields:
                    if field in data:
                        entry[field] = data[field]
                current_metadata_list.append(entry)

        os.makedirs(voyager_index_base_path, exist_ok=True) 

        if not texts:
            print(f"No texts found in '{input_jsonl_path}' for field '{field_to_index}'. Creating empty index structure.")
            _ = pylate_indexes.Voyager(
                index_folder=voyager_index_base_path,
                index_name=self.INDEX_NAME_INTERNAL,
                override=True, 
            )
            with open(metadata_path, 'wb') as f:
                pickle.dump([], f)
            return

        document_ids = [str(i) for i in range(len(texts))]
        self.target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        pool = self.pylate_model.start_multi_process_pool(target_devices=self.target_devices)
        num_processes = len(pool["processes"])
        chunk_size = math.ceil(len(texts) / (num_processes * 20))
        chunk_size = max(32, min(chunk_size, 5000))
        
        documents_embeddings_list = self.pylate_model.encode_multi_process(
            texts,
            pool,
            batch_size=self.batch_size_encode, 
            chunk_size=chunk_size,
            is_query=False,
            show_progress_bar=self.enable_tqdm
        )
        self.pylate_model.stop_multi_process_pool(pool)
        voyager_index = pylate_indexes.Voyager(
            index_folder=voyager_index_base_path,
            index_name=self.INDEX_NAME_INTERNAL,
            override=True,
            embedding_size=self.pylate_model.get_sentence_embedding_dimension() 
        )

        voyager_index.add_documents(
            documents_ids=document_ids,
            documents_embeddings=documents_embeddings_list
        )

        with open(metadata_path, 'wb') as f:
            pickle.dump(current_metadata_list, f)
            
        del documents_embeddings_list
        del texts
        del current_metadata_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def retrieve(self,
                 nlqs: List[str],
                 output_folder: str,
                 k: int,
                 k_token_for_rerank: int = 100) -> List[List[RetrievalResult]]: # Added k_token_for_rerank

        # --- Start of Profiling Block 1: Setup & Metadata Load ---
        # start_time_setup = time.time()
        voyager_index_base_path = os.path.join(output_folder, self.INDEX_SUBFOLDER)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)
        
        voyager_key_file_path = os.path.join(
            voyager_index_base_path,
            self.INDEX_NAME_INTERNAL,
            "index.voyager"
        )

        if not os.path.exists(metadata_path) or not os.path.exists(voyager_key_file_path):
            print(f"PyLate ColBERT index or metadata not found in '{output_folder}'. Ensure indexing was completed.")
            return [[] for _ in nlqs]

        with open(metadata_path, 'rb') as f:
            doc_metadata_list: List[Dict[str, Any]] = pickle.load(f)

        if not nlqs:
            print("PyLateColbertRetriever.retrieve: No queries provided.")
            return [] 
        
        if not doc_metadata_list:
            print(f"PyLate ColBERT index in '{output_folder}' (metadata) is empty. Returning no results.")
            return [[] for _ in nlqs] 

        voyager_index = pylate_indexes.Voyager(
            index_folder=voyager_index_base_path,
            index_name=self.INDEX_NAME_INTERNAL,
            override=False, 
            # You could expose ef_search here if L1 becomes a bottleneck
            # ef_search=200 # Default in PyLate's Voyager
        )

        retriever_engine = pylate_retrieve.ColBERT(index=voyager_index)
        # end_time_setup = time.time()
        # print(f"DEBUG: Setup and metadata load time: {end_time_setup - start_time_setup:.4f}s")
        
        cuda_available = torch.cuda.is_available()
        num_cuda_devices = torch.cuda.device_count()
        
        print(f"PyLateColbertRetriever.retrieve: CUDA Available: {cuda_available}, Num Devices: {num_cuda_devices}")

        # --- Start of Profiling Block 2: Query Encoding ---
        # start_time_encoding = time.time()
        target_devices_for_encoding = [f"cuda:{i}" for i in range(num_cuda_devices)] if cuda_available and num_cuda_devices > 0 else ["cpu"]
        print(f"PyLateColbertRetriever.retrieve: Target devices for query encoding pool: {target_devices_for_encoding}")
        
        pool = self.pylate_model.start_multi_process_pool(target_devices=target_devices_for_encoding)
        
        num_worker_processes = len(pool["processes"]) if pool and "processes" in pool and pool["processes"] is not None else 0
        print(f"PyLateColbertRetriever.retrieve: Number of worker processes for encoding: {num_worker_processes}")

        all_queries_embeddings_list = []

        if num_worker_processes > 0 and self.batch_size_encode > 0:
            external_macro_chunk_size = self.batch_size_encode * num_worker_processes * 10 
        else: 
            external_macro_chunk_size = max(1, self.batch_size_encode * 10) 
        
        external_macro_chunk_size = min(external_macro_chunk_size, len(nlqs) if nlqs else self.batch_size_encode)
        external_macro_chunk_size = max(1, external_macro_chunk_size)

        nlqs_iterator_source = [nlqs[i:i + external_macro_chunk_size] for i in range(0, len(nlqs), external_macro_chunk_size)]
        
        actual_iterator = nlqs_iterator_source
        if self.enable_tqdm and len(nlqs_iterator_source) > 1:
             actual_iterator = tqdm(nlqs_iterator_source, desc="Encoding query macro-batches")
        
        for nlq_macro_batch in actual_iterator:
            if not nlq_macro_batch:
                continue
            
            batch_embeddings_list = self.pylate_model.encode_multi_process(
                nlq_macro_batch,
                pool,
                batch_size=self.batch_size_encode, 
                chunk_size=None, 
                is_query=True
            )
            all_queries_embeddings_list.extend(batch_embeddings_list)
        
        self.pylate_model.stop_multi_process_pool(pool)
        # end_time_encoding = time.time()
        # print(f"DEBUG: Query encoding time: {end_time_encoding - start_time_encoding:.4f}s for {len(nlqs)} queries.")
        
        device_for_reranking = "cuda" if cuda_available and num_cuda_devices > 0 else "cpu"
        print(f"PyLateColbertRetriever.retrieve: Device for PyLate ColBERT reranking step: {device_for_reranking}")
        print(f"PyLateColbertRetriever.retrieve: Number of encoded queries for retrieval engine: {len(all_queries_embeddings_list)}")
        print(f"PyLateColbertRetriever.retrieve: Reranking top {k} from {k_token_for_rerank} L1 candidates.")


        # --- Start of Profiling Block 3: L1 Retrieval & L2 Reranking ---
        # start_time_retrieval_rerank = time.time()
        pylate_scores_and_ids = retriever_engine.retrieve(
            queries_embeddings=all_queries_embeddings_list,
            k=k, # Final k results
            k_token=k_token_for_rerank, # Candidates from L1 to L2
            device=device_for_reranking, 
            batch_size=256 
        )
        # end_time_retrieval_rerank = time.time()
        # print(f"DEBUG: L1 retrieval & L2 rerank time: {end_time_retrieval_rerank - start_time_retrieval_rerank:.4f}s")

        all_batches_results: List[List[RetrievalResult]] = []

        # --- Start of Profiling Block 4: Result Post-Processing ---
        # start_time_post_processing = time.time()
        iterable_pylate_results = pylate_scores_and_ids
        if self.enable_tqdm and len(all_queries_embeddings_list) > 0 :
             iterable_pylate_results = tqdm(pylate_scores_and_ids, desc="Final processing of retrieved results", total=len(all_queries_embeddings_list))

        for query_idx, query_results_list_of_dicts in enumerate(iterable_pylate_results):
            batch_results: List[RetrievalResult] = []
            for result_item_dict in query_results_list_of_dicts: 
                doc_id_str = result_item_dict['id']
                score_val = result_item_dict['score']
                
                try:
                    doc_idx = int(doc_id_str)
                    if not (0 <= doc_idx < len(doc_metadata_list)):
                        print(f"Warning: Retrieved document index {doc_idx} is out of bounds for doc_metadata_list (len {len(doc_metadata_list)}) for query '{nlqs[query_idx] if query_idx < len(nlqs) else 'UNKNOWN QUERY INDEX'}'. Skipping.")
                        continue
                    meta = doc_metadata_list[doc_idx]
                    text_content = meta.get('_text') 
                    if not isinstance(text_content, str) or not text_content: 
                        print(f"Warning: '_text' field is not a valid string or is missing for doc_idx {doc_idx} (original id: {meta.get('_original_id', 'N/A')}). Skipping.")
                        continue
                except ValueError:
                    print(f"Warning: Could not convert document ID '{doc_id_str}' to integer for query '{nlqs[query_idx] if query_idx < len(nlqs) else 'UNKNOWN QUERY INDEX'}'. Skipping.")
                    continue
                except IndexError: 
                    print(f"Warning: Document index {doc_id_str} (parsed as {doc_idx if 'doc_idx' in locals() else 'PARSE_FAILED'}) out of range for query '{nlqs[query_idx] if query_idx < len(nlqs) else 'UNKNOWN QUERY INDEX'}'. Skipping.")
                    continue
                
                extra_metadata = {key: val for key, val in meta.items() if key != '_text'}

                batch_results.append(
                    RetrievalResult(
                        score=float(score_val), 
                        object=text_content,
                        metadata=extra_metadata
                    )
                )
            all_batches_results.append(batch_results)
        # end_time_post_processing = time.time()
        # print(f"DEBUG: Result post-processing time: {end_time_post_processing - start_time_post_processing:.4f}s")

        if cuda_available:
            torch.cuda.empty_cache()
            
        return all_batches_results