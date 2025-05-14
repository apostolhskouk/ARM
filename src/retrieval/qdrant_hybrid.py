import json
# import faiss # No longer needed for this class
import pickle # Still potentially needed if helper functions load pickled data elsewhere
import uuid
import os
import gc # For garbage collection
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import QueryResponse
from accelerate import Accelerator

from src.retrieval.base import BaseRetriever, RetrievalResult

class QdrantBGEHybridRetriever(BaseRetriever):
    """
    Hybrid Retriever using FlagEmbedding BGE-M3 model (dense, sparse, colbert)
    and Qdrant. Embeddings generated locally, storage/retrieval via remote Qdrant.
    """
    # Constants for vector names used in Qdrant
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"
    COLBERT_VECTOR_NAME = "colbert"

    # BGE-M3 specific dimension
    BGE_M3_DIMENSION = 1024

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        qdrant_host: str = "195.251.63.238",
        qdrant_port: int = 6334,
        collection_name: str = "bge_m3_hybrid_docs", # New default name
        prefer_grpc: bool = True,
        use_fp16: bool = True
    ):
        """
        Initializes the retriever with a BGE-M3 model and Qdrant client config
        for hybrid search (dense, sparse, colbert).

        Args:
            model_name_or_path: Path/name of the BGE model.
            qdrant_host: Hostname/IP of the Qdrant server.
            qdrant_port: Port (gRPC if prefer_grpc=True, else HTTP) of Qdrant.
            collection_name: Name for the Qdrant collection configured for hybrid search.
            prefer_grpc: Whether to use gRPC interface.
            use_fp16: Whether to load the model in float16 precision.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        effective_use_fp16 = use_fp16 and self.device == 'cuda'
        if use_fp16 and not effective_use_fp16:
            print("Warning: use_fp16=True but no CUDA device detected. Loading model in fp32.")

        print(f"Loading BGE model: {model_name_or_path} (fp16: {effective_use_fp16}) for Hybrid Search...")
        try:
            self.model = BGEM3FlagModel(model_name_or_path, use_fp16=effective_use_fp16)
            print("BGE model loaded successfully.")
        except Exception as e:
             print(f"FATAL: Error loading BGE model: {e}")
             raise

        self.embedding_dim = self.BGE_M3_DIMENSION

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.prefer_grpc = prefer_grpc

        print(f"Initializing Qdrant client for HYBRID search: {self.qdrant_host}:{self.qdrant_port} (gRPC: {self.prefer_grpc})")
        self.client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port if not self.prefer_grpc else None,
            grpc_port=self.qdrant_port if self.prefer_grpc else None,
            prefer_grpc=self.prefer_grpc,
            timeout=60
        )
        print(f"Qdrant client configured for collection '{self.collection_name}'.")

    def _create_sparse_vector_from_bge(self, sparse_data: Dict[str, float]) -> models.SparseVector:
        """
        Converts BGE-M3 sparse output (lexical_weights) to Qdrant sparse vector format.
        Handles string keys and ensures only positive values are included.
        """
        sparse_indices = []
        sparse_values = []
        if not sparse_data: # Handle empty sparse data case
            return models.SparseVector(indices=[], values=[])

        for key, value in sparse_data.items():
            # BGE-M3 lexical_weights uses string keys for token IDs
            processed_key = None
            if isinstance(key, str) and key.isdigit():
                 processed_key = int(key)
            elif isinstance(key, int):
                 processed_key = key
            else:
                continue # Skip non-integer keys

            # Only include positive weights as per typical sparse vector usage
            float_value = float(value)
            if float_value > 0:
                sparse_indices.append(processed_key)
                sparse_values.append(float_value)

        return models.SparseVector(indices=sparse_indices, values=sparse_values)

    def _ensure_collection_exists(self):
        """
        Checks if the collection exists and is configured correctly for
        BGE-M3 hybrid search (dense, sparse, colbert). Creates/recreates if needed.
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' found.")

            # --- Validate vector configurations ---
            valid_dense = False
            valid_sparse = False
            valid_colbert = False

            # Check Dense
            if self.DENSE_VECTOR_NAME in collection_info.vectors_config.params_map:
                dense_params = collection_info.vectors_config.params_map[self.DENSE_VECTOR_NAME].params
                if dense_params.size == self.embedding_dim and dense_params.distance == models.Distance.COSINE:
                    valid_dense = True
            # Check Sparse
            if self.SPARSE_VECTOR_NAME in collection_info.sparse_vectors_config.map:
                # Basic check, could add index param checks if needed
                 valid_sparse = True
            # Check ColBERT
            if self.COLBERT_VECTOR_NAME in collection_info.vectors_config.params_map:
                colbert_params = collection_info.vectors_config.params_map[self.COLBERT_VECTOR_NAME].params
                if (colbert_params.size == self.embedding_dim and
                    colbert_params.distance == models.Distance.COSINE and
                    colbert_params.multivector_config and
                    colbert_params.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM):
                    valid_colbert = True

            if valid_dense and valid_sparse and valid_colbert:
                print("Collection configuration is compatible for hybrid search.")
                return # Config is valid
            else:
                 print("Warning: Collection configuration mismatch. Recreating for hybrid search.")
                 # Fall through to recreate logic

        except Exception as e:
            # If collection not found or other error occurs during check, try to create/recreate
            if "not found" in str(e).lower() or "status_code=404" in str(e).lower():
                 print(f"Collection '{self.collection_name}' not found. Creating for hybrid search...")
            else:
                 print(f"Warning: Error checking collection '{self.collection_name}' ({e}). Attempting to recreate...")
            # Fall through to recreate logic

        # --- Recreate Collection Logic ---
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                # Configure named vectors
                vectors_config={
                    self.DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    ),
                    self.COLBERT_VECTOR_NAME: models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE, # Important for MAX_SIM
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                },
                # Configure named sparse vectors
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True) # As per notebook
                    )
                }
            )
            print(f"Collection '{self.collection_name}' created/recreated successfully for hybrid search.")
        except Exception as e:
             print(f"FATAL: Failed to create/recreate collection '{self.collection_name}': {e}")
             raise # Stop if collection setup fails


    def index(self,
              input_jsonl_path: str,
              field_to_index: str,
              metadata_fields: List[str],
              output_folder: str = None,
              
             ) -> None:
        """
        Reads data, generates BGE-M3 dense, sparse, and colbert embeddings locally,
        formats sparse, normalizes dense, and upserts points with named vectors
        (ID, vector_dict, payload) to the remote Qdrant collection.

        Args:
            input_jsonl_path: Path to the input JSONL file.
            field_to_index: Key in JSON containing the main text to embed.
            metadata_fields: List of keys from JSON to store as Qdrant payload.
            batch_size: How many texts to encode with BGE model at once.
            qdrant_batch_size: How many points to send to Qdrant in one API call.
        """
        self._ensure_collection_exists() # Ensure collection is ready for hybrid

        # --- Check if collection is already populated ---
        try:
            count_result = self.client.count(collection_name=self.collection_name, exact=True)
            if count_result.count > 0:
                print(f"Collection '{self.collection_name}' already exists and contains {count_result.count} points. Skipping indexing.")
                return
        except Exception as e:
            print(f"Warning: Could not retrieve point count for collection '{self.collection_name}': {e}")
            print("Aborting indexing to prevent potential duplicates.")
            return

        batch_size = 8192
        qdrant_batch_size = 64 
        all_texts = []
        all_payloads = []
        all_ids = []
        ids_generated = 0

        print(f"Reading data from {input_jsonl_path}...")
        try:
            with open(input_jsonl_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing JSONL",disable=False):
                    try:
                        data = json.loads(line.strip())
                        text = data.get(field_to_index)
                        if not text or not isinstance(text, str):
                            continue

                        # Prepare payload *first*, including original text
                        payload = {'_text': text}
                        for field in metadata_fields:
                            if field in data:
                                payload[field] = data[field]
                        all_payloads.append(payload)

                        doc_id = str(uuid.uuid4())
                        ids_generated += 1
                        all_ids.append(doc_id)

                        # Append text for batch encoding
                        # Consider pre-processing/formatting text like in the notebook if needed
                        # processed_text = f"Product: {data.get('Name', '')}\\nDescription: {text}"
                        processed_text = text # Using raw text for now
                        all_texts.append(processed_text)

                    except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line.")
                    except Exception as e: print(f"Warning: Error processing line: {e}")

        except FileNotFoundError: print(f"Error: Input file not found at {input_jsonl_path}"); return
        except Exception as e: print(f"Error reading input file: {e}"); return

        if not all_texts: print("No valid documents found to index."); return

        # --- Process in batches ---
        points_to_upsert = []
        num_docs = len(all_texts)

        print(f"Generating BGE-M3 hybrid embeddings for {num_docs} documents...")
        for i in tqdm(range(0, num_docs, batch_size), desc="Encoding Batches",disable=False):
            batch_texts_slice = all_texts[i : i + batch_size]
            batch_ids_slice = all_ids[i : i + batch_size]
            batch_payloads_slice = all_payloads[i : i + batch_size]

            if not batch_texts_slice: continue

            try:
                # Generate all three embedding types for the batch
                output = self.model.encode(
                    batch_texts_slice,
                    batch_size=len(batch_texts_slice),
                    max_length=1024,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )

                # Extract results for the batch
                batch_dense_vecs = output['dense_vecs']
                batch_sparse_weights = output['lexical_weights']
                batch_colbert_vecs = output['colbert_vecs']

                # Ensure numpy arrays where expected
                if not isinstance(batch_dense_vecs, np.ndarray): batch_dense_vecs = np.array(batch_dense_vecs)
                # sparse_weights is a list of dicts
                # colbert_vecs is a list of numpy arrays

                # Normalize dense vectors
                norms = np.linalg.norm(batch_dense_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                batch_dense_normalized = batch_dense_vecs / norms

                # Process each item in the batch result
                for j in range(len(batch_texts_slice)):
                    doc_id = batch_ids_slice[j]
                    doc_payload = batch_payloads_slice[j]
                    doc_dense = batch_dense_normalized[j]
                    doc_sparse_data = batch_sparse_weights[j]
                    doc_colbert = batch_colbert_vecs[j] # This is already a list/array of vectors

                    # Convert sparse data
                    qdrant_sparse = self._create_sparse_vector_from_bge(doc_sparse_data)

                    # Create the PointStruct with named vectors
                    point = models.PointStruct(
                        id=doc_id,
                        payload=doc_payload,
                        vector={
                            self.DENSE_VECTOR_NAME: doc_dense.tolist(),
                            self.SPARSE_VECTOR_NAME: qdrant_sparse,
                            # ColBERT vectors might need tolist() if they are numpy arrays
                            self.COLBERT_VECTOR_NAME: doc_colbert.tolist() if isinstance(doc_colbert, np.ndarray) else doc_colbert
                        }
                    )
                    points_to_upsert.append(point)

            except Exception as e:
                print(f"\nError encoding or processing batch starting at document index {i}: {e}")
                print("Skipping this batch.")
                continue # Skip to the next batch on error

        if not points_to_upsert:
            print("No points were successfully processed for upserting.")
            return

        # --- Upsert to Qdrant in batches ---
        print(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{self.collection_name}'...")
        try:
            for start in range(0, len(points_to_upsert), qdrant_batch_size):
                end = start + qdrant_batch_size
                batch = points_to_upsert[start:end]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            print("Upsert completed successfully.")
        except Exception as e:
            print(f"Error during Qdrant upsert (batch starting at {start}): {e}")
            print("Indexing may be incomplete.")

        # --- Cleanup ---
        del all_texts, all_payloads, all_ids, points_to_upsert
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("Indexing finished.")


    def retrieve(self,
                nlqs: List[str],
                k: int,
                output_folder: str = None # Not used in this method,
                ) -> List[List[RetrievalResult]]:
        """
        Performs hybrid search for multiple queries using BGE-M3 embeddings.
        1. Encodes queries to get dense, sparse, colbert vectors.
        2. Uses Qdrant's query_points with prefetch (dense+sparse) and
           ColBERT reranking.

        Args:
            nlqs: List of natural language queries.
            k: Final number of results to retrieve for each query after reranking.
            prefetch_limit: Number of candidates to fetch using dense/sparse before reranking.
            query_batch_size: How many queries to encode in one go.

        Returns:
            A list of lists, containing top 'k' RetrievalResult objects for each query.
        """
        if not nlqs:
            print("Warning: No queries provided."); return []
        prefetch_limit: int = 10
        query_batch_size: int = 16
        num_queries = len(nlqs)
        all_results: List[List[RetrievalResult]] = [[] for _ in range(num_queries)]

        print(f"Performing hybrid search for {num_queries} queries...")
        # Process queries in batches for encoding efficiency
        for i in tqdm(range(0, num_queries, query_batch_size), desc="Processing Query Batches", disable=False):
            batch_nlqs_slice = nlqs[i : i + query_batch_size]
            if not batch_nlqs_slice: continue

            try:
                # Encode the batch of queries
                output = self.model.encode(
                    batch_nlqs_slice,
                    batch_size=len(batch_nlqs_slice),
                    max_length=1024, # Use appropriate max length for queries
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )

                # Extract batch results
                batch_dense_vecs = output['dense_vecs']
                batch_sparse_weights = output['lexical_weights']
                batch_colbert_vecs = output['colbert_vecs']

                # Ensure numpy arrays
                if not isinstance(batch_dense_vecs, np.ndarray): batch_dense_vecs = np.array(batch_dense_vecs)
                # ColBERT vecs is list of numpy arrays

                # Normalize dense query vectors
                norms = np.linalg.norm(batch_dense_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                batch_dense_normalized = batch_dense_vecs / norms

                # --- Perform Qdrant search for each query in the batch ---
                for j in range(len(batch_nlqs_slice)):
                    query_index = i + j # Original index in nlqs list
                    q_dense = batch_dense_normalized[j]
                    q_sparse_data = batch_sparse_weights[j]
                    q_colbert = batch_colbert_vecs[j] # Array of vectors for the query

                    # Convert query sparse data
                    qdrant_sparse_query = self._create_sparse_vector_from_bge(q_sparse_data)

                    # Define Prefetch queries (fetch candidates using dense and sparse)
                    prefetch_queries = [
                        models.Prefetch(
                            query=q_dense.tolist(),
                            using=self.DENSE_VECTOR_NAME,
                            limit=prefetch_limit
                        ),
                         models.Prefetch(
                            query=qdrant_sparse_query,
                            using=self.SPARSE_VECTOR_NAME,
                            limit=prefetch_limit
                         )
                    ]

                    # Perform the reranking query using ColBERT vectors
                    search_result: QueryResponse = self.client.query_points(
                        collection_name=self.collection_name,
                        prefetch=prefetch_queries,
                        # The main query uses the ColBERT vector for reranking
                        query=q_colbert.tolist() if isinstance(q_colbert, np.ndarray) else q_colbert,
                        using=self.COLBERT_VECTOR_NAME, # Rerank based on ColBERT similarity
                        limit=k, # Get the final top k results
                        with_payload=True # We need the payload for results
                    )

                    # Process results for this query
                    batch_query_results: List[RetrievalResult] = []
                    for hit in search_result.points: # Access points attribute of QueryResponse
                        payload = hit.payload or {}
                        text = payload.get('_text', '')
                        if not isinstance(text, str) or not text:
                            print(f"Warning: Retrieved point (ID: {hit.id}) missing valid '_text' for query {query_index}. Skipping.")
                            continue
                        extra_metadata = {key: v for key, v in payload.items() if key != '_text'}
                        batch_query_results.append(
                            RetrievalResult(score=hit.score, object=text, metadata=extra_metadata)
                        )
                    all_results[query_index] = batch_query_results # Store results at correct index

            except Exception as e:
                 print(f"\nError processing query batch starting at index {i}: {e}")
                 # Leave the results for this batch as empty lists

        # --- Cleanup ---
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("Hybrid retrieval finished.")
        return all_results

