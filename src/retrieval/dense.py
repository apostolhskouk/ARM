import json
import os
import pickle
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch
from src.retrieval.base import BaseRetriever, RetrievalResult
import numpy as np
class FaissDenseRetriever(BaseRetriever):
    """
    Dense Retriever implementation using Sentence Transformers and FAISS.
    Indexes using FAISS CPU, retrieves using FAISS GPU if available.
    """
    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.pkl"

    def __init__(self, model_name_or_path: str = "WhereIsAI/UAE-Large-V1", enable_tqdm: bool = True):
        """
        Initializes the retriever with a Sentence Transformer model.

        Args:
            model_name_or_path: The name/path of the Sentence Transformer model to use.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name_or_path, device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.num_gpus = torch.cuda.device_count()

        self.faiss_index_cpu: Optional[faiss.Index] = None
        self.doc_metadata_list: Optional[List[Dict[str, Any]]] = None
        self.text_to_id_map: Optional[Dict[str, int]] = None
        self._loaded_output_folder: Optional[str] = None
        self.enable_tqdm = enable_tqdm

    def index(self,
              input_jsonl_path: str,
              output_folder: str,
              field_to_index: str,
              metadata_fields: List[str]) -> None:
        """
        Indexes documents by embedding text and storing in a FAISS CPU index.
        Also saves metadata associated with each document.
        """
        os.makedirs(output_folder, exist_ok=True)
        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Index and metadata exist in '{output_folder}', skipping.")
            return

        texts, metadata_list = [], []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get(field_to_index)
                if not text or not isinstance(text, str):
                    continue
                texts.append(text)
                entry = {'_text': text}
                for field in metadata_fields:
                    if field in data:
                        entry[field] = data[field]
                metadata_list.append(entry)

        if not texts:
            print("No valid documents to index.")
            return
        target_devices = [f'cuda:{i}' for i in range(self.num_gpus)] if self.num_gpus > 0 else ['cpu']
        pool = self.model.start_multi_process_pool(target_devices=target_devices)
        encode_batch_size = 4
        embeddings = self.model.encode_multi_process(
            texts,
            pool=pool,
            batch_size=encode_batch_size,
            show_progress_bar=True,
        )
        self.model.stop_multi_process_pool(pool)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_list, f)
        del embeddings
        del texts
        del metadata_list
        torch.cuda.empty_cache() 

    def retrieve(self,
                 nlqs: List[str],
                 output_folder: str,
                 k: int) -> List[List[RetrievalResult]]:
        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"Index or metadata not found in '{output_folder}'")
            return [[] for _ in nlqs] # Return list of empty lists

        # Load once
        try:
            index_cpu = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                doc_metadata_list: List[Dict[str, Any]] = pickle.load(f)
        except Exception as e:
            print(f"Error loading index/metadata from {output_folder}: {e}")
            return [[] for _ in nlqs]

        # Move to GPU if available
        index_to_search = index_cpu # Default to CPU
        res = None # Keep track of GPU resource
        if faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                index_to_search = faiss.index_cpu_to_gpu(res, 0, index_cpu)
                print("Using FAISS GPU index.")
            except Exception as e:
                print(f"GPU move failed ({e}), using CPU index.")
                # No need to assign index_to_search = index_cpu again, it's the default
        else:
             print("Using FAISS CPU index.")


        # Batch encode queries
        try:
            query_embeddings = self.model.encode(
                nlqs,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            faiss.normalize_L2(query_embeddings)
        except Exception as e:
            print(f"Error encoding queries: {e}")
            return [[] for _ in nlqs]

        try:
            scores, indices = index_to_search.search(query_embeddings, k)
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return [[] for _ in nlqs]

        # Build results
        all_batches: List[List[RetrievalResult]] = []
        num_docs_in_index = len(doc_metadata_list)

        for qi, nlq in enumerate(tqdm(nlqs, desc="Retrieving with FAISS", disable=not self.enable_tqdm)):
            batch_results: List[RetrievalResult] = []
            if qi >= len(indices) or qi >= len(scores):
                 print(f"Warning: Mismatch in FAISS search results length for query index {qi}. Skipping.")
                 all_batches.append(batch_results)
                 continue

            query_indices = indices[qi]
            query_scores = scores[qi]

            for rank, doc_idx in enumerate(query_indices):
                # Check for invalid index or rank
                if doc_idx < 0 or doc_idx >= num_docs_in_index or rank >= len(query_scores):
                    continue

                # --- Safely get score ---
                raw_score = query_scores[rank]
                current_score: Optional[float] = None
                try:
                    if np.isnan(raw_score): current_score = -float('inf')
                    elif np.isinf(raw_score): current_score = float(1e12) if raw_score > 0 else -float('inf')
                    else: current_score = float(raw_score)
                except Exception: current_score = -float('inf') # Fallback on any conversion error
                # --- End Score ---

                # --- Get metadata and ensure text is string ---
                meta = doc_metadata_list[doc_idx]
                text_obj = meta.get('_text') # Get potential text

                # Ensure text is a usable string, otherwise skip this result
                if not isinstance(text_obj, str) or not text_obj:
                    print(f"Warning: Retrieved object is not a valid string for doc_idx {doc_idx}. Skipping result.")
                    continue
                text = text_obj # Now we know it's a valid string

                extra = {key: val for key, val in meta.items() if key != '_text'}
                # --- End Metadata ---

                batch_results.append(
                    RetrievalResult(
                        score=current_score, # Should always be a float 
                        object=text,         # Should always be a non-empty string
                        metadata=extra
                    )
                )
            all_batches.append(batch_results)
        torch.cuda.empty_cache()
        if res:
            del res  
        return all_batches        


    def faiss_encode(self, sentences: List[str], output_folder: str, batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        if self.text_to_id_map is None:
            index_path = os.path.join(output_folder, self.INDEX_FILENAME)
            metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)

            self.faiss_index_cpu = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.doc_metadata_list = pickle.load(f)
            
            self.text_to_id_map = {
                meta['_text']: i 
                for i, meta in enumerate(self.doc_metadata_list) 
                if isinstance(meta.get('_text'), str)
            }
            
        
        if not sentences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        
        results_list = [None] * len(sentences)
        indices_to_encode_newly = []
        sentences_to_encode_newly_map = {} 

        for i, sentence in enumerate(sentences):
            if sentence in self.text_to_id_map:
                doc_id = self.text_to_id_map[sentence]
                embedding = self.faiss_index_cpu.reconstruct(doc_id).copy()
                results_list[i] = embedding
            else:
                indices_to_encode_newly.append(i)
                sentences_to_encode_newly_map[i] = sentence
        if sentences_to_encode_newly_map:
            ordered_sentences_to_encode = [sentences_to_encode_newly_map[i] for i in indices_to_encode_newly]
            
            new_embeddings = self.model.encode(
                ordered_sentences_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            
            if new_embeddings.ndim == 1 and len(ordered_sentences_to_encode) == 1:
                new_embeddings = np.expand_dims(new_embeddings, axis=0)
            
            faiss.normalize_L2(new_embeddings)

            for i, original_idx in enumerate(indices_to_encode_newly):
                results_list[original_idx] = new_embeddings[i]
        
        return np.array(results_list, dtype=np.float32)