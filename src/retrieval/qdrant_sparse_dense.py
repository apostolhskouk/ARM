import json
import pickle # Keep for potential use elsewhere, though not directly used here
import uuid
import os
import gc # For garbage collection
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np
from qdrant_client import QdrantClient, models
# Import necessary Qdrant models
from qdrant_client.http.models import (
    Batch, Distance, VectorParams, PointStruct, SparseVectorParams,
    SparseIndexParams, SparseVector, ScoredPoint # Search returns ScoredPoint
)

# Assuming these are defined correctly in your project structure
from src.retrieval.base import BaseRetriever, RetrievalResult

class QdrantBGEDenseSparseRetriever(BaseRetriever):
    """
    Dense + Sparse Retriever using FlagEmbedding BGE-M3 model and Qdrant.
    Generates dense and sparse embeddings locally, stores and retrieves
    via a remote Qdrant server using combined search.
    Keeps the same index/retrieve signature as previous dense retrievers.
    """
    # Constants for vector names used in Qdrant
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    # BGE-M3 specific dimension
    BGE_M3_DIMENSION = 1024

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        qdrant_host: str = "195.251.63.238",
        qdrant_port: int = 6334,
        collection_name: str = "bge_m3_dense_sparse_docs", # New default name
        prefer_grpc: bool = True,
        use_fp16: bool = True
    ):
        """
        Initializes the retriever with a BGE-M3 model and Qdrant client config
        for dense + sparse search.

        Args:
            model_name_or_path: Path/name of the BGE model.
            qdrant_host: Hostname/IP of the Qdrant server.
            qdrant_port: Port (gRPC if prefer_grpc=True, else HTTP) of Qdrant.
            collection_name: Name for the Qdrant collection configured for dense/sparse search.
            prefer_grpc: Whether to use gRPC interface.
            use_fp16: Whether to load the model in float16 precision.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        effective_use_fp16 = use_fp16 and self.device == 'cuda'
        if use_fp16 and not effective_use_fp16:
            print("Warning: use_fp16=True but no CUDA device detected. Loading model in fp32.")

        print(f"Loading BGE model: {model_name_or_path} (fp16: {effective_use_fp16}) for Dense+Sparse Search...")
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

        print(f"Initializing Qdrant client for DENSE+SPARSE search: {self.qdrant_host}:{self.qdrant_port} (gRPC: {self.prefer_grpc})")
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
            processed_key = None
            if isinstance(key, str) and key.isdigit():
                 processed_key = int(key)
            elif isinstance(key, int):
                 processed_key = key
            else:
                continue # Skip non-integer keys

            float_value = float(value)
            if float_value > 0:
                sparse_indices.append(processed_key)
                sparse_values.append(float_value)

        return models.SparseVector(indices=sparse_indices, values=sparse_values)

    def _ensure_collection_exists(self):
        """
        Checks if the collection exists and is configured correctly for
        BGE-M3 dense + sparse search. Creates/recreates if needed.
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' found.")

            valid_dense = False
            valid_sparse = False

            if self.DENSE_VECTOR_NAME in collection_info.vectors_config.params_map:
                dense_params = collection_info.vectors_config.params_map[self.DENSE_VECTOR_NAME].params
                if dense_params.size == self.embedding_dim and dense_params.distance == models.Distance.COSINE:
                    valid_dense = True
            if self.SPARSE_VECTOR_NAME in collection_info.sparse_vectors_config.map:
                 valid_sparse = True

            if valid_dense and valid_sparse:
                print("Collection configuration is compatible for dense+sparse search.")
                return
            else:
                 print("Warning: Collection configuration mismatch. Recreating for dense+sparse search.")

        except Exception as e:
            if "not found" in str(e).lower() or "status_code=404" in str(e).lower():
                 print(f"Collection '{self.collection_name}' not found. Creating for dense+sparse search...")
            else:
                 print(f"Warning: Error checking collection '{self.collection_name}' ({e}). Attempting to recreate...")

        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )
            print(f"Collection '{self.collection_name}' created/recreated successfully for dense+sparse search.")
        except Exception as e:
             print(f"FATAL: Failed to create/recreate collection '{self.collection_name}': {e}")
             raise


    # =======================================================================
    # INDEX METHOD - Signature matches previous Dense Retriever
    # =======================================================================
    def index(self,
              input_jsonl_path: str,
              output_folder: str, # Argument accepted but NOT USED internally
              field_to_index: str,
              metadata_fields: List[str]
             ) -> None:
        """
        Reads data, generates BGE-M3 dense and sparse embeddings locally,
        formats sparse, normalizes dense, and upserts points with named vectors
        (ID, vector_dict{'dense':..., 'sparse':...}, payload) to Qdrant.

        Args:
            input_jsonl_path: Path to the input JSONL file.
            output_folder: Path to an output folder (ARGUMENT IGNORED for Qdrant).
            field_to_index: Key in JSON containing the main text to embed.
            metadata_fields: List of keys from JSON to store as Qdrant payload.
        """
        # --- Internal Configuration ---
        encoding_batch_size = 96 # Set encoding batch size internally
        qdrant_upsert_batch_size = 64 # Set Qdrant upsert batch size internally

        if output_folder: # Acknowledge the argument even if unused
            print(f"Note: 'output_folder' argument ('{output_folder}') is ignored by Qdrant retrievers during indexing.")

        self._ensure_collection_exists() # Ensure collection is ready for dense+sparse

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

        # --- Data Reading ---
        all_texts = []
        all_payloads = []
        all_ids = []

        print(f"Reading data from {input_jsonl_path}...")
        try:
            with open(input_jsonl_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing JSONL"):
                    try:
                        data = json.loads(line.strip())
                        text = data.get(field_to_index)
                        if not text or not isinstance(text, str): continue

                        payload = {'_text': text}
                        for field in metadata_fields:
                            if field in data: payload[field] = data[field]
                        all_payloads.append(payload)

                        # Always generate UUIDs in this version
                        doc_id = str(uuid.uuid4())
                        all_ids.append(doc_id)

                        processed_text = text
                        all_texts.append(processed_text)

                    except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line.")
                    except Exception as e: print(f"Warning: Error processing line: {e}")

        except FileNotFoundError: print(f"Error: Input file not found at {input_jsonl_path}"); return
        except Exception as e: print(f"Error reading input file: {e}"); return

        if not all_texts: print("No valid documents found to index."); return

        # --- Encoding and Processing in Batches ---
        points_to_upsert = []
        num_docs = len(all_texts)

        print(f"Generating BGE-M3 dense+sparse embeddings for {num_docs} documents...")
        for i in tqdm(range(0, num_docs, encoding_batch_size), desc="Encoding Batches"): # Use internal batch size
            batch_texts_slice = all_texts[i : i + encoding_batch_size]
            batch_ids_slice = all_ids[i : i + encoding_batch_size]
            batch_payloads_slice = all_payloads[i : i + encoding_batch_size]

            if not batch_texts_slice: continue

            try:
                output = self.model.encode(
                    batch_texts_slice,
                    batch_size=len(batch_texts_slice),
                    max_length=8192,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False # Explicitly False
                )

                batch_dense_vecs = output['dense_vecs']
                batch_sparse_weights = output['lexical_weights']

                if not isinstance(batch_dense_vecs, np.ndarray): batch_dense_vecs = np.array(batch_dense_vecs)

                norms = np.linalg.norm(batch_dense_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                batch_dense_normalized = batch_dense_vecs / norms

                for j in range(len(batch_texts_slice)):
                    doc_id = batch_ids_slice[j]
                    doc_payload = batch_payloads_slice[j]
                    doc_dense = batch_dense_normalized[j]
                    doc_sparse_data = batch_sparse_weights[j]

                    qdrant_sparse = self._create_sparse_vector_from_bge(doc_sparse_data)

                    point = models.PointStruct(
                        id=doc_id,
                        payload=doc_payload,
                        vector={
                            self.DENSE_VECTOR_NAME: doc_dense.tolist(),
                            self.SPARSE_VECTOR_NAME: qdrant_sparse
                        }
                    )
                    points_to_upsert.append(point)

            except Exception as e:
                print(f"\nError encoding or processing batch starting at document index {i}: {e}")
                print("Skipping this batch.")
                continue

        if not points_to_upsert:
            print("No points were successfully processed for upserting.")
            return

        # --- Upsert to Qdrant ---
        print(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{self.collection_name}'...")
        try:
             self.client.upsert(
                 collection_name=self.collection_name,
                 points=points_to_upsert,
                 batch_size=qdrant_upsert_batch_size, # Use internal batch size
                 wait=True
             )
             print("Upsert completed successfully.")
        except Exception as e:
            print(f"Error during Qdrant batch upsert: {e}")
            print("Indexing may be incomplete.")

        # --- Cleanup ---
        del all_texts, all_payloads, all_ids, points_to_upsert
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("Indexing finished.")


\
    def retrieve(self,
                 nlqs: List[str],
                 k: int,
                 output_folder: str = None # Argument accepted but NOT USED internally
                ) -> List[List[RetrievalResult]]:
        """
        Performs dense + sparse search for multiple queries using BGE-M3 embeddings.
        1. Encodes queries to get dense and sparse vectors.
        2. Uses Qdrant's standard search combining dense and sparse vectors.

        Args:
            nlqs: List of natural language queries.
            k: Final number of results to retrieve for each query.
            output_folder: Path to output folder (ARGUMENT IGNORED for Qdrant retrieval).


        Returns:
            A list of lists, containing top 'k' RetrievalResult objects for each query.
        """

        if not nlqs:
            print("Warning: No queries provided."); return []

        # --- Internal Configuration ---
        query_encoding_batch_size: int = 16 # Set query encoding batch size internally

        num_queries = len(nlqs)
        all_results: List[List[RetrievalResult]] = [[] for _ in range(num_queries)]

        print(f"Performing dense+sparse search for {num_queries} queries...")
        # Process queries in batches for encoding efficiency
        for i in tqdm(range(0, num_queries, query_encoding_batch_size), desc="Processing Query Batches"): # Use internal batch size
            batch_nlqs_slice = nlqs[i : i + query_encoding_batch_size]
            if not batch_nlqs_slice: continue

            try:
                output = self.model.encode(
                    batch_nlqs_slice,
                    batch_size=len(batch_nlqs_slice),
                    max_length=8192,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False # Explicitly False
                )

                batch_dense_vecs = output['dense_vecs']
                batch_sparse_weights = output['lexical_weights']

                if not isinstance(batch_dense_vecs, np.ndarray): batch_dense_vecs = np.array(batch_dense_vecs)

                norms = np.linalg.norm(batch_dense_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                batch_dense_normalized = batch_dense_vecs / norms

                # --- Perform Qdrant search for each query in the batch ---
                for j in range(len(batch_nlqs_slice)):
                    query_index = i + j
                    q_dense = batch_dense_normalized[j]
                    q_sparse_data = batch_sparse_weights[j]

                    qdrant_sparse_query = self._create_sparse_vector_from_bge(q_sparse_data)

                    # Use standard search with multiple named query vectors
                    search_result: List[ScoredPoint] = self.client.search(
                        collection_name=self.collection_name,
                        query_vector={
                            self.DENSE_VECTOR_NAME: q_dense.tolist(),
                            self.SPARSE_VECTOR_NAME: qdrant_sparse_query
                        },
                        limit=k,
                        with_payload=True
                    )

                    batch_query_results: List[RetrievalResult] = []
                    for hit in search_result:
                        payload = hit.payload or {}
                        text = payload.get('_text', '')
                        if not isinstance(text, str) or not text:
                            print(f"Warning: Retrieved point (ID: {hit.id}) missing valid '_text' for query {query_index}. Skipping.")
                            continue
                        extra_metadata = {key: v for key, v in payload.items() if key != '_text'}
                        batch_query_results.append(
                            RetrievalResult(score=hit.score, object=text, metadata=extra_metadata)
                        )
                    all_results[query_index] = batch_query_results

            except Exception as e:
                 print(f"\nError processing query batch starting at index {i}: {e}")

        # --- Cleanup ---
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("Dense+sparse retrieval finished.")
        return all_results

