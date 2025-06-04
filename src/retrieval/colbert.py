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
                 batch_size_encode: int = 1024):

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
                 k: int) -> List[List[RetrievalResult]]:

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
            return [] 
        
        if not doc_metadata_list:
            print(f"PyLate ColBERT index in '{output_folder}' (metadata) is empty. Returning no results.")
            return [[] for _ in nlqs] 

        voyager_index = pylate_indexes.Voyager(
            index_folder=voyager_index_base_path,
            index_name=self.INDEX_NAME_INTERNAL,
            override=False, 
        )

        retriever = pylate_retrieve.ColBERT(index=voyager_index)

        queries_embeddings = self.pylate_model.encode(
            nlqs,
            batch_size=self.batch_size_encode,
            is_query=True,
            show_progress_bar=self.enable_tqdm,
        )

        pylate_scores_and_ids = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=k,
        )

        all_batches_results: List[List[RetrievalResult]] = []

        iterable_results = pylate_scores_and_ids
        if self.enable_tqdm:
            iterable_results = tqdm(pylate_scores_and_ids, desc="Retrieving with PyLate ColBERT", total=len(nlqs))

        for query_results in iterable_results:
            batch_results: List[RetrievalResult] = []
            for doc_id_str, score_val in query_results:
                doc_idx = int(doc_id_str) 
                
                meta = doc_metadata_list[doc_idx]
                text_content = meta['_text'] 
                
                extra_metadata = {key: val for key, val in meta.items() if key != '_text'}

                batch_results.append(
                    RetrievalResult(
                        score=float(score_val), 
                        object=text_content,
                        metadata=extra_metadata
                    )
                )
            all_batches_results.append(batch_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return all_batches_results