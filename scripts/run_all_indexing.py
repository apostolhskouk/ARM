import os
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
import time
from src.retrieval.arm import ARMRetriever
EMBEDDING_MODEL = "BAAI/bge-m3"
INPUT_FOLDER = "assets/all_data/serialized_data"
OUTPUT_FOLDER_BM25 = "assets/all_data/indexes/bm25"
OUTPUT_FOLDER_DENSE = "assets/all_data/indexes/dense_bge_m3"
METADATA_FIELDS_TO_INDEX = ["page_title", "source"]
FIELD_TO_INDEX = "object"
ARM_VLLM_MODEL_PATH = "gaunernst/gemma-3-27b-it-int4-awq"
ARM_NGRAM_LLM_MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"

def main():
    bm25_retriever = PyseriniBM25Retriever()
    dense_retriever = FaissDenseRetriever(EMBEDDING_MODEL)
    arm_retriever = ARMRetriever(
        vllm_model_path=ARM_VLLM_MODEL_PATH,
        ngram_llm_model_path=ARM_NGRAM_LLM_MODEL_PATH,
        embedding_model_name=EMBEDDING_MODEL,
        vllm_tensor_parallel_size=1,
        expansion_steps=0
    )
    for file_name in os.listdir(INPUT_FOLDER):
        print(f"Processing file: {file_name}")
        input_file_path = os.path.join(INPUT_FOLDER, file_name)
        file_name = file_name.split('.')[0]
        bm25_output_folder = os.path.join(OUTPUT_FOLDER_BM25, file_name)
        dense_output_folder = os.path.join(OUTPUT_FOLDER_DENSE, file_name)
        bm_25_time = time.time()
        bm25_retriever.index(input_file_path, bm25_output_folder,FIELD_TO_INDEX,METADATA_FIELDS_TO_INDEX)
        bm_25_time = time.time() - bm_25_time
        dense_time = time.time()
        dense_retriever.index(input_file_path, dense_output_folder, FIELD_TO_INDEX, METADATA_FIELDS_TO_INDEX)
        dense_time = time.time() - dense_time
        arm_time = time.time()
        arm_retriever.index(input_file_path, [bm25_output_folder,dense_output_folder], FIELD_TO_INDEX, METADATA_FIELDS_TO_INDEX)
        arm_time = time.time() - arm_time
        print(f"BM25 indexing time: {bm_25_time:.2f} seconds")
        print(f"Dense indexing time: {dense_time:.2f} seconds")
        print(f"ARM indexing time: {arm_time:.2f} seconds")
        
if __name__ == "__main__":
    main()
    print("Finished these dumb things, going for some beers now.")