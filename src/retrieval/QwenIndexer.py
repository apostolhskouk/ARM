import json
import os
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm
from vllm import LLM

from src.retrieval.base import BaseRetriever, RetrievalResult


class QwenIndexer(BaseRetriever):
    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.pkl"

    def __init__(self,
                 model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
                 task_instruction: str = 'Given a natural language query, retrieve relevant passages or rows from tables',
                 enable_tqdm: bool = True):

        self.model_name_or_path = model_name_or_path
        self.task_instruction = task_instruction
        self.enable_tqdm = enable_tqdm
        self.num_gpus = torch.cuda.device_count()

        self.model = LLM(
            model=self.model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=self.num_gpus,
            task="embed",
            enforce_eager=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=8192,
        )
        self.embedding_dim = self.model.llm_engine.model_config.get_hidden_size()
        self.max_length = self.model.llm_engine.model_config.max_model_len

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_instruction}\nQuery:{query}'

    def _encode(self, texts: List[str]) -> np.ndarray:
        request_outputs = self.model.embed(
            texts,
            truncate_prompt_tokens=self.max_length
        )
        embeddings = [output.outputs.embedding for output in request_outputs]
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        return embeddings_np

    def index(self,
              input_jsonl_path: str,
              output_folder: str,
              field_to_index: str,
              metadata_fields: List[str]) -> None:

        os.makedirs(output_folder, exist_ok=True)
        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Index and metadata exist in '{output_folder}', skipping.")
            return
        texts_to_index = []
        doc_metadata_list = []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading documents", disable=not self.enable_tqdm):
                data = json.loads(line.strip())
                text = data[field_to_index]
                texts_to_index.append(text)

                metadata_entry = {'_text': text}
                for field in metadata_fields:
                    if field in data:
                        metadata_entry[field] = data[field]
                doc_metadata_list.append(metadata_entry)

        embeddings_np = self._encode(texts_to_index)

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings_np)

        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(doc_metadata_list, f)

        del embeddings_np
        del texts_to_index
        del doc_metadata_list
        torch.cuda.empty_cache()

    def retrieve(self,
                 nlqs: List[str],
                 output_folder: str,
                 k: int) -> List[List[RetrievalResult]]:

        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)

        index_cpu = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            doc_metadata_list = pickle.load(f)

        index_to_search = faiss.index_cpu_to_all_gpus(index_cpu)

        formatted_queries = [self._get_detailed_instruct(q) for q in nlqs]
        query_embeddings = self._encode(formatted_queries)

        scores, indices = index_to_search.search(query_embeddings, k)

        all_results: List[List[RetrievalResult]] = []
        for i in range(len(nlqs)):
            query_results = []
            for rank, doc_id in enumerate(indices[i]):
                if doc_id == -1:
                    continue

                score = float(scores[i][rank])
                metadata = doc_metadata_list[doc_id]
                text = metadata['_text']
                extra_metadata = {key: val for key, val in metadata.items() if key != '_text'}

                result = RetrievalResult(
                    score=score,
                    object=text,
                    metadata=extra_metadata
                )
                query_results.append(result)
            all_results.append(query_results)

        del index_to_search
        torch.cuda.empty_cache()

        return all_results