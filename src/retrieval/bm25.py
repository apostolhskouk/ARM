from pyserini.search.lucene import LuceneSearcher
import abc
import json
import os
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.retrieval.base import BaseRetriever, RetrievalResult
from tqdm import tqdm
class PyseriniBM25Retriever(BaseRetriever):
    """
    BM25 Retriever implementation using Pyserini.
    """

    def index(self,
              input_jsonl_path: str,
              output_folder: str,
              field_to_index: str,
              metadata_fields: List[str]) -> None:
        """
        Indexes documents using Pyserini's Lucene indexer.
        Converts input JSONL to the format Pyserini expects and stores
        metadata within the raw document payload.
        """

        if os.path.exists(output_folder) :
            print(f"Index already exist in '{output_folder}'. Skipping indexing.")
            return
        temp_dir = None
        try:
            # Pyserini's JsonCollection requires input to be a directory
            temp_dir = tempfile.mkdtemp()
            prepared_jsonl_path = os.path.join(temp_dir, "prepared_data.jsonl")
            line_count = 0
            with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
                 open(prepared_jsonl_path, 'w', encoding='utf-8') as outfile:

                for line in infile:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        original_data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line {line_count + 1} in {input_jsonl_path}")
                        continue

                    text_content = original_data.get(field_to_index)
                    if not text_content or not isinstance(text_content, str):
                        print(f"Skipping line {line_count + 1}: Missing or invalid field '{field_to_index}'")
                        continue

                    pyserini_doc = {
                        "id": f"doc_{line_count}", # Simple sequential ID
                        "contents": text_content
                    }
                    # Add requested metadata fields to the dict that will be stored
                    for meta_field in metadata_fields:
                        if meta_field in original_data:
                            pyserini_doc[meta_field] = original_data[meta_field]

                    outfile.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
                    line_count += 1

            if line_count == 0:
                print("No valid documents found to index.")
                return # No point running Pyserini if no data

            os.makedirs(output_folder, exist_ok=True)

            num_threads = os.cpu_count() or 1
            cmd = [
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", temp_dir, 
                "--index", output_folder,
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", str(num_threads),
                "--storePositions", 
                "--storeDocvectors", 
                "--storeRaw" #
            ]

            try:
                process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

            except subprocess.CalledProcessError as e:
                print(f"Pyserini indexing failed with {e}")
                raise e

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def retrieve(self,
                nlqs: List[str],
                output_folder: str,
                k: int) -> List[List[RetrievalResult]]:
        """
        Retrieves documents using a Pyserini LuceneSearcher with batch search.
        """
        if not os.path.exists(output_folder):
            print(f"Index directory not found: {output_folder}")
            return [[] for _ in nlqs] # Return empty list for each query

        try:
            # Initialize searcher once
            searcher = LuceneSearcher(output_folder)
            # Prepare arguments for batch search
            nlq_list = [nlq for nlq in nlqs] # Ensure it's a list
            k_list = [k] * len(nlqs)
            # Use available threads, default to 1 if not detectable
            num_threads = os.cpu_count() or 1

            # Perform batch search
            batch_hits = searcher.batch_search(
                queries=nlq_list,
                qids=[f"q{i}" for i in range(len(nlqs))], 
                k=k,
                threads=num_threads
            )

        except Exception as e:
            print(f"Pyserini search failed: {e}")
            return [[] for _ in nlqs]

        all_batches: List[List[RetrievalResult]] = []
        # Process results for each query based on the original order
        for i, nlq in enumerate(tqdm(nlqs, desc="Retrieving with BM25")):
            qid = f"q{i}"
            hits = batch_hits.get(qid, []) # Get hits for the current query id
            results: List[RetrievalResult] = []
            for hit in hits:
                try:
                    if hasattr(hit, 'raw_doc'):
                        raw_doc_str = hit.raw_doc
                    elif hasattr(hit, 'lucene_document'):
                         raw_doc_str = hit.lucene_document.get("raw")
                    else:
                         raw_doc_str = None 

                    if raw_doc_str:
                        stored_data = json.loads(raw_doc_str)
                        retrieved_object = stored_data.get("contents", "")
                        retrieved_metadata = {
                            key: value for key, value in stored_data.items()
                            if key not in [ "contents"]
                        }
                        results.append(RetrievalResult(
                            score=hit.score,
                            object=retrieved_object,
                            metadata=retrieved_metadata
                        ))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode stored JSON for a hit in query {qid}")
                except Exception as e:
                    print(f"Warning: Error processing hit for query {qid}: {e}")

            # Pyserini batch_search might not guarantee order, but we sort by score anyway
            results.sort(key=lambda r: r.score, reverse=True)
            all_batches.append(results[:k]) # Ensure we don't exceed k

        return all_batches

