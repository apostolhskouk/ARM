import json
import faiss
import pickle
import uuid
import os
from typing import List,Dict,Any
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel # ADDED
import torch
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Batch

from src.retrieval.base import BaseRetriever, RetrievalResult


class QdrantBGEDenseRetriever(BaseRetriever): # Renamed class for clarity
    """
    Dense Retriever using FlagEmbedding BGE-M3 model and Qdrant.
    Generates ONLY dense embeddings locally (Server 2), stores and retrieves
    via a remote Qdrant server (Server 1).
    """

    # BGE-M3 specific dimension
    BGE_M3_DIMENSION = 1024

    def __init__(
        self,
        # Default to BGE-M3 model
        model_name_or_path: str = "BAAI/bge-m3",
        # --- Qdrant Connection Details ---
        qdrant_host: str = "195.251.63.238", # Your Server 1 IP Address
        qdrant_port: int = 6334,            # Your Qdrant gRPC Port
        collection_name: str = "bge_m3_documents", # Choose a name
        prefer_grpc: bool = True,
        use_fp16: bool = True # Use fp16 for speed/memory if GPU available (as per notebook)
    ):
        """
        Initializes the retriever with a BGE-M3 model and Qdrant client config.

        Args:
            model_name_or_path: Path to the BGE model (default: BAAI/bge-m3).
            qdrant_host: Hostname/IP of the Qdrant server.
            qdrant_port: Port (gRPC if prefer_grpc=True, else HTTP) of Qdrant.
            collection_name: Name for the Qdrant collection.
            prefer_grpc: Whether to use gRPC interface.
            use_fp16: Whether to load the model in float16 precision (requires GPU).
        """
        # --- Model Setup (Locally on Server 2 using FlagEmbedding) ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Determine if fp16 can be used
        effective_use_fp16 = use_fp16 and self.device == 'cuda'
        if use_fp16 and not effective_use_fp16:
            print("Warning: use_fp16=True but no CUDA device detected. Loading model in fp32.")
        try:
            # BGEM3FlagModel doesn't take device argument directly, handles internally
            self.model = BGEM3FlagModel(model_name_or_path, use_fp16=effective_use_fp16)
        except Exception as e:
             print(f"Error loading BGE model: {e}")
             print("Ensure 'FlagEmbedding' and potentially 'accelerate' are installed.")
             raise

        self.embedding_dim = self.BGE_M3_DIMENSION # Use known dimension for BGE-M3

        # --- Qdrant Client Setup (Connecting to Server 1) ---
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.prefer_grpc = prefer_grpc

        print(f"Initializing Qdrant client for {self.qdrant_host}:{self.qdrant_port} (gRPC: {self.prefer_grpc})")
        self.client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port if not self.prefer_grpc else None,
            grpc_port=self.qdrant_port if self.prefer_grpc else None,
            prefer_grpc=self.prefer_grpc,
            timeout=60
        )
        print(f"Qdrant client configured.")

    def _ensure_collection_exists(self):
        """Checks if the collection exists and creates it for BGE-M3 dense vectors if not."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            # Validate existing collection's DEFAULT vector parameters for dense vectors
            if 'default' in collection_info.vectors_config.params_map:
                existing_params = collection_info.vectors_config.params_map['default'].params
                if existing_params.size != self.embedding_dim or existing_params.distance != models.Distance.COSINE:
                    print(f"Warning: Collection default vector params mismatch "
                          f"(Size: {existing_params.size} vs {self.embedding_dim}, "
                          f"Dist: {existing_params.distance} vs {models.Distance.COSINE}). Recreating...")
                    self.client.recreate_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
                    )
                    print(f"Collection '{self.collection_name}' recreated.")
                else:
                     print("Collection default vector parameters are compatible.")
            else:
                 # If no default config exists, recreate it
                 print("Warning: No default vector configuration found. Recreating collection...")
                 self.client.recreate_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
                 )
                 print(f"Collection '{self.collection_name}' recreated.")

        except Exception as e:
            if "not found" in str(e).lower() or "status_code=404" in str(e).lower():
                 print(f"Collection '{self.collection_name}' not found. Creating...")
                 self.client.recreate_collection(
                    collection_name=self.collection_name,
                    # Define ONLY the default dense vector configuration
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE # Use Cosine for normalized BGE embeddings
                    )
                    # No sparse_vectors_config or other vector names needed
                 )
                 print(f"Collection '{self.collection_name}' created.")
            else:
                print(f"Error checking or creating collection '{self.collection_name}': {e}")
                raise

    def index(self,
              input_jsonl_path: str,
              output_folder: str,
              field_to_index: str,
              metadata_fields: List[str]) -> None:
        """
        Reads data, generates BGE-M3 dense embeddings locally, normalizes,
        and upserts points (ID, vector, payload) to the remote Qdrant collection.

        Args:
            input_jsonl_path: Path to the input JSONL file.
            field_to_index: Key in JSON containing text to embed.
            metadata_fields: List of keys from JSON to store as Qdrant payload.
        """
        try:
            count_result = self.client.count(collection_name=self.collection_name, exact=True)
            if count_result.count > 0:
                print(f"Collection '{self.collection_name}' already exists and contains {count_result.count} points. Skipping indexing.")
                return
        except:
            pass
        self._ensure_collection_exists()
        batch_size =32
        qdrant_batch_size = 128
        texts_to_embed = []
        payloads = []
        ids = []

        try:
            with open(input_jsonl_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing JSONL"):
                    try:
                        data = json.loads(line.strip())
                        text = data.get(field_to_index)
                        if not text or not isinstance(text, str):
                            continue

                        processed_text = text 

                        texts_to_embed.append(processed_text)
                        payload = {'_text': text} # Store original text
                        for field in metadata_fields:
                            if field in data:
                                payload[field] = data[field]
                        payloads.append(payload)
                        ids.append(str(uuid.uuid4()))
                    except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line.")
                    except Exception as e: print(f"Warning: Error processing line: {e}")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_jsonl_path}"); return
        except Exception as e:
            print(f"Error reading input file: {e}"); return

        if not texts_to_embed:
            print("No valid documents found to index."); return
        all_embeddings = []
        # Process in batches for memory efficiency during encoding
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Encoding Batches"):
            batch_texts = texts_to_embed[i:i + batch_size]
            try:
                # Use FlagEmbedding's encode method for dense vectors only
                output = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts), # Encode the current batch
                    max_length=8192, # BGE-M3 supports longer sequences
                    return_dense=True,
                    return_sparse=False, # Explicitly false
                    return_colbert_vecs=False # Explicitly false
                )
                batch_embeddings = output['dense_vecs']
                # Ensure numpy array
                if not isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = np.array(batch_embeddings)

                all_embeddings.append(batch_embeddings)

            except Exception as e:
                print(f"Error encoding batch starting at index {i}: {e}")
                return

        if not all_embeddings:
            print("Embedding generation failed.")
            return

        # Concatenate embeddings from all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        del all_embeddings # Free memory

        # --- Normalize embeddings ---
        try:
             norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
             norms[norms == 0] = 1e-9 # Avoid division by zero
             embeddings = embeddings / norms
        except Exception as e:
             print(f"Error during embedding normalization: {e}"); return

        try:
            for i in tqdm(range(0, len(ids), qdrant_batch_size), desc="Upserting to Qdrant"):
                batch_ids = ids[i : i + qdrant_batch_size]
                batch_embeddings_normalized = embeddings[i : i + qdrant_batch_size]
                batch_payloads = payloads[i : i + qdrant_batch_size]
                batch_vectors = [vector.tolist() for vector in batch_embeddings_normalized]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=Batch(
                         ids=batch_ids,
                         vectors=batch_vectors, # Use the default vector field
                         payloads=batch_payloads
                     ),
                    wait=True
                )
        except Exception as e:
            print(f"Error during Qdrant upsert: {e}")
            print("Indexing may be incomplete.")

        del embeddings, texts_to_embed, payloads, ids
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    def retrieve(self,
                 nlqs: List[str],
                 k: int,
                 output_folder: str = None,
                ) -> List[List[RetrievalResult]]:
        """
        Generates BGE-M3 dense query embeddings locally, normalizes,
        and searches the remote Qdrant collection using the default vector index.

        Args:
            nlqs: List of natural language queries.
            k: Number of results to retrieve for each query.
        Returns:
            A list of lists, containing RetrievalResult objects for each query.
        """
        if not nlqs:
            print("Warning: No queries provided."); return []
        query_batch_size = 64
        all_query_embeddings = []
        try:
             for i in tqdm(range(0, len(nlqs), query_batch_size), desc="Encoding Queries"):
                  batch_nlqs = nlqs[i:i+query_batch_size]
                  output = self.model.encode(
                     batch_nlqs,
                     batch_size=len(batch_nlqs),
                     max_length=8192,
                     return_dense=True,
                     return_sparse=False,
                     return_colbert_vecs=False
                  )
                  batch_q_embeddings = output['dense_vecs']
                  if not isinstance(batch_q_embeddings, np.ndarray):
                       batch_q_embeddings = np.array(batch_q_embeddings)
                  all_query_embeddings.append(batch_q_embeddings)

        except Exception as e:
             print(f"Error generating query embeddings: {e}")
             return [[] for _ in nlqs]

        if not all_query_embeddings:
             print("Query embedding generation failed.")
             return [[] for _ in nlqs]

        query_embeddings = np.concatenate(all_query_embeddings, axis=0)
        del all_query_embeddings

        try:
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            query_embeddings = query_embeddings / norms
        except Exception as e:
            print(f"Error normalizing query embeddings: {e}")
            return [[] for _ in nlqs]

        all_results: List[List[RetrievalResult]] = []

        # --- Query Qdrant (Server 1) ---
        for i, query_vector in enumerate(tqdm(query_embeddings, desc="Querying Qdrant")):
            batch_results: List[RetrievalResult] = []
            try:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(), # Use the normalized dense vector
                    limit=k,
                )

                for hit in search_result:
                    payload = hit.payload or {}
                    text = payload.get('_text', '')
                    if not isinstance(text, str) or not text:
                        print(f"Warning: Retrieved point (ID: {hit.id}) missing valid '_text'. Skipping.")
                        continue
                    extra_metadata = {k: v for k, v in payload.items() if k != '_text'}
                    batch_results.append(
                        RetrievalResult(score=hit.score, object=text, metadata=extra_metadata)
                    )
                all_results.append(batch_results)

            except Exception as e:
                 print(f"Error searching Qdrant for query {i} ('{nlqs[i][:50]}...'): {e}")
                 all_results.append([])

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Retrieval finished.")
        return all_results


    def index_from_faiss(
        self,
        faiss_folder: str,
        qdrant_batch_size: int = 256
    ) -> None:
        """
        Loads vectors and metadata from a previously created Faiss index folder
        (containing 'index.faiss' and 'metadata.pkl') and upserts them
        into the configured Qdrant collection without re-calculating embeddings.

        Assumes the Faiss index was created with vectors corresponding
        sequentially to the entries in the metadata list, and that vectors
        were normalized before being added to the Faiss index (as done
        in the original script using IndexFlatIP).

        Args:
            faiss_folder: Path to the directory containing 'index.faiss'
                          and 'metadata.pkl'.
            qdrant_batch_size: How many points to send to Qdrant in one API call.
        """
        try:
            count_result = self.client.count(collection_name=self.collection_name, exact=True)
            if count_result.count > 0:
                print(f"Collection '{self.collection_name}' already exists and contains {count_result.count} points. Skipping indexing.")
                return
        except:
            pass
        faiss_index_path = os.path.join(faiss_folder, "index.faiss")
        metadata_path = os.path.join(faiss_folder, "metadata.pkl")

        # --- Input Validation ---
        if not os.path.isdir(faiss_folder):
            print(f"Error: Faiss folder not found at '{faiss_folder}'")
            return
        if not os.path.exists(faiss_index_path):
            print(f"Error: Faiss index file not found at '{faiss_index_path}'")
            return
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata file not found at '{metadata_path}'")
            return

        print(f"Loading data from Faiss folder: {faiss_folder}")

        try:
            # --- Load Metadata ---
            print("Loading metadata...")
            with open(metadata_path, 'rb') as f:
                # Assuming metadata_list is a list of dictionaries as saved previously
                payloads: List[Dict[str, Any]] = pickle.load(f)
            num_metadata = len(payloads)
            print(f"Loaded {num_metadata} metadata entries.")
            if num_metadata == 0:
                print("Error: Metadata file is empty. Cannot proceed.")
                return

            # --- Load Faiss Index ---
            print("Loading Faiss index...")
            index = faiss.read_index(faiss_index_path)
            num_vectors = index.ntotal
            faiss_dim = index.d
            print(f"Loaded Faiss index with {num_vectors} vectors of dimension {faiss_dim}.")

            # --- Consistency Checks ---
            if num_vectors != num_metadata:
                print(f"Error: Mismatch between number of vectors in Faiss ({num_vectors})"
                      f" and number of metadata entries ({num_metadata}). Aborting.")
                return
            if faiss_dim != self.embedding_dim:
                print(f"Error: Mismatch between dimension in Faiss index ({faiss_dim})"
                      f" and retriever's expected dimension ({self.embedding_dim}). Aborting.")
                # This might happen if the Faiss index was built with a different model.
                return
            
            self._ensure_collection_exists()
            
            # --- Extract Vectors from Faiss ---
            # Reconstruct all vectors. Assumes they fit in memory.
            # If the index is huge, might need to reconstruct in batches.
            print("Reconstructing vectors from Faiss index...")
            vectors_np = index.reconstruct_n(0, num_vectors)
            print(f"Reconstructed {vectors_np.shape[0]} vectors.")

            # --- Prepare for Qdrant Upsert ---
            # Generate unique IDs for Qdrant points
            ids = [str(uuid.uuid4()) for _ in range(num_vectors)]

            # Vectors are assumed to be pre-normalized (as per original Faiss script)
            # No need to re-normalize here if using Cosine distance in Qdrant

            print(f"Upserting {num_vectors} points from Faiss to Qdrant collection '{self.collection_name}'...")
            # --- Upsert to Qdrant in batches ---
            for i in tqdm(range(0, num_vectors, qdrant_batch_size), desc="Upserting Faiss data to Qdrant"):
                batch_ids = ids[i : i + qdrant_batch_size]
                batch_vectors_np = vectors_np[i : i + qdrant_batch_size]
                batch_payloads = payloads[i : i + qdrant_batch_size]

                # Convert numpy vectors to lists
                batch_vectors_list = [v.tolist() for v in batch_vectors_np]

                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=Batch(
                            ids=batch_ids,
                            vectors=batch_vectors_list, # Default vector field
                            payloads=batch_payloads
                        ),
                        wait=True
                    )
                except Exception as e:
                    print(f"\nError during Qdrant upsert in batch starting at index {i}: {e}")
                    print("Stopping further upserts for this Faiss index.")
                    # Optionally add retry logic or skip the batch
                    return # Stop the process on error

            print(f"Successfully indexed {num_vectors} points from Faiss folder '{faiss_folder}' into Qdrant collection '{self.collection_name}'.")

        except faiss.FaissException as e:
             print(f"Faiss error: {e}")
        except pickle.UnpicklingError as e:
             print(f"Error reading metadata file (pickle): {e}. Try installing/using 'pickle5'.")
        except MemoryError:
             print("Memory Error: Failed to load or reconstruct all Faiss vectors into memory.")
             print("Consider reconstructing and upserting vectors in smaller batches if the index is very large.")
        except Exception as e:
             print(f"An unexpected error occurred: {e}")
        finally:
            # Clean up potentially large objects
            del payloads, index, vectors_np, ids
            import gc
            gc.collect() # Force garbage collection