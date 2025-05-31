import os
from src.retrieval.bm25 import PyseriniBM25Retriever
from src.retrieval.dense import FaissDenseRetriever
import time
EMBEDDING_MODEL = "infly/inf-retriever-v1-1.5b"
INPUT_FOLDER = "assets/all_data/serialized_data"
OUTPUT_FOLDER_BM25 = "assets/all_data/indexes/bm25"
OUTPUT_FOLDER_DENSE = "assets/all_data/indexes/dense_infly_v1_1.5b"
METADATA_FIELDS_TO_INDEX = ["page_title", "source"]
FIELD_TO_INDEX = "object"
def main():
    bm25_retriever = PyseriniBM25Retriever()
    dense_retriever = FaissDenseRetriever(EMBEDDING_MODEL)
    #iterate over all files in the input folder
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
        dense_retriever.index(input_file_path, dense_output_folder, FIELD_TO_INDEX, EMBEDDING_MODEL)
        dense_time = time.time() - dense_time
        print(f"BM25 indexing time: {bm_25_time:.2f} seconds")
        print(f"Dense indexing time: {dense_time:.2f} seconds")
        
if __name__ == "__main__":
    main()
    print("Finished these dumb things, going for some beers now.")