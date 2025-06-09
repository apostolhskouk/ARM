import time
from src.retrieval.dense import FaissDenseRetriever
import os 
from time import time
INDEXES_BASE_DIR_TEMPLATE = "assets/all_data/indexes/{embedding_folder_key}/"
INPUT_FOLDER = "assets/all_data/serialized_data"

EMBEDDING_MODELS_CONFIG = {
  "Alibaba-NLP/gte-modernbert-base": "dense_alibaba_gte_modernbert",
  "Alibaba-NLP/gte-multilingual-base": "dense_alibaba_gte_multilingual",
  "Snowflake/snowflake-arctic-embed-l-v2.0": "dense_arctic_embed_l_v2",
  "Snowflake/snowflake-arctic-embed-s": "dense_artic_embed_s",
  "BAAI/bge-m3": "dense_bge_m3",
  "intfloat/multilingual-e5-large-instruct": "dense_e5_large_instruct",
  "infly/inf-retriever-v1-1.5b": "dense_infly_v1_1.5b",
  "jinaai/jina-embeddings-v3": "dense_jina_v3",
  "nomic-ai/nomic-embed-text-v2-moe": "dense_nomic_embed_text_v2_moe",
  "nomic-ai/modernbert-embed-base": "dense_nomic_modernbert",
  "WhereIsAI/UAE-Large-V1": "dense_uae_large_v1",
  "Qwen/Qwen3-Embedding-0.6B": "dense_qwen_3",
}

FIELD_TO_INDEX = "object"
METADATA_FIELDS_TO_INDEX = ["page_title", "source"]

def main():
    for embedding_model_name, embedding_folder_key in EMBEDDING_MODELS_CONFIG.items():
        dense_retriever = FaissDenseRetriever(embedding_model_name)
        for file_name in os.listdir(INPUT_FOLDER):
            file_path = os.path.join(INPUT_FOLDER, file_name)
            output_folder = os.path.join(INDEXES_BASE_DIR_TEMPLATE.format(embedding_folder_key=embedding_folder_key), file_name.split('.')[0])
            start_time = time()
            dense_retriever.index(file_path, output_folder, FIELD_TO_INDEX, METADATA_FIELDS_TO_INDEX)
            elapsed_time = time() - start_time
            print(f"Indexed {file_name} with {embedding_model_name} in {elapsed_time:.2f} seconds. Output folder: {output_folder}")
if __name__ == "__main__":
    main()
    print("Finished indexing with all embeddings.")
