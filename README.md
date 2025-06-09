# ARM Paper Reproduction for Database Systems

This repository contains the project for the **Database Systems** course, part of the MSc in Data Science & Information Technologies (Class of 2025).

## Objective

The primary objective of this project is to reproduce the methodology described in the paper: **[Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](https://arxiv.org/abs/2501.18539)**.

## 📊 Datasets

While the original ARM paper evaluated its methods on the BIRD and OTT-QA benchmarks, this project extends the evaluation to a broader and more diverse set of benchmarks to thoroughly test the retrieval methodology.

We have curated a collection of benchmarks that include table-based, text-based, and mixed-modality retrieval tasks. To focus solely on the retrieval aspect, we have standardized all datasets into a unified format.

### Benchmarks Included

*   **[FEVEROUS](https://fever.ai/dataset/feverous.html)**: A fact verification dataset requiring systems to find evidence from both text and tables in Wikipedia to support or refute claims.
*   **[MT-RAIG](https://arxiv.org/abs/2502.11735)**: Designed to evaluate Retrieval-Augmented Insight Generation over multiple, unknown tables, moving beyond simple factoid QA.
*   **[MultiHop-RAG](https://arxiv.org/abs/2401.15391)**: A benchmark focused on multi-hop queries, where systems must retrieve and reason over multiple text passages to find an answer.
*   **[TableBench](https://arxiv.org/abs/2408.09174)**: A complex benchmark designed to mirror real-world industrial scenarios for table-based question answering.
*   **[BIRD](https://arxiv.org/pdf/2305.03111)**: A large-scale text-to-SQL benchmark that emphasizes challenges found in real-world databases, such as dirty data and the need for efficient SQL queries.
*   **[FeTaQA](https://arxiv.org/abs/2104.00369)**: Features questions that require generating free-form, explanatory answers by integrating facts from multiple cells within a single table.
*   **[OTT-QA](https://arxiv.org/abs/2010.10439)**: A large-scale benchmark for open-domain question answering over both tables and text, often requiring multi-hop reasoning across them.
*   **[Spider](https://arxiv.org/abs/1809.08887)**: A large, complex, and cross-domain text-to-SQL dataset that tests a model's ability to generalize to new database schemas and SQL structures.
*   **[TabFact](https://arxiv.org/abs/1909.02164)**: A large-scale dataset for fact verification where statements must be verified as entailed or refuted based on evidence from a given table.

> **Note**: the format and the pre-processing for the last 5 datasets was taken from **[TARGET](https://arxiv.org/abs/2505.11545)**.

> **Note**: for the Spider dataset we try two serialization; one per row and one per schema. For BIRD since the per-row serialization, after some experiments, was proven to not work, because even when retiieving 2048 candidates per query (the max candidates faiss supports), at recall@5, 90% of the retrievals had less than 5 retrieved objects, meaning that from one table multiple rows where rerieved.

### ⚙️ Data Serialization

To create a unified retrieval corpus, all source documents were processed into a `jsonl` format where each line is a retrievable "object".

*   **Table Serialization**: We experimented with three strategies for serializing tables: as a **whole table**, **row-by-row**, and **value-by-value**. Based on experiments on the FEVEROUS benchmark, the **row-by-row** format proved most effective. Therefore, all tables in the corpus are serialized as individual rows using the format:
    `"title if applicable [SEP] [H] colname1 : val1 , [H] colname2 : val2 ..."`

*   **Text Serialization**: Text passages are serialized as:
    `"title [SEP] text_passage"`
    *For long documents like those in MultiHop-RAG, we performed semantic chunking using [chonky](https://github.com/mirth/chonky).*

Each `jsonl` record contains the following fields:
*   `object`: The serialized string of the table row or text passage.
*   `page_title`: The identifier for the broader document (e.g., database name, Wikipedia page title).
*   `source`: The specific identifier within the document (e.g., `table_01`, `sentence_42`).

### ⚠️ A Note on Retrieval vs. Evaluation Granularity

It's important to note a key distinction in our process:
*   **Retrieval is performed at the *object* level**: The system retrieves individual table rows or text chunks.
*   **Evaluation is measured at the *document* level**: Success is determined by whether the correct unique document ID (e.g., the source table or article, identified by `page_title` + `source`) is retrieved.

This creates a challenge: retrieving the top *k* objects might result in fewer than *k* unique documents, as multiple rows from the same table could be returned. Our evaluation scripts are designed to handle this by de-duplicating the document IDs post-retrieval before calculating metrics.

### 🎯 Query Set Creation

Given the large size of the original benchmarks and the computational cost of advanced rerankers, we subsampled the queries for each dataset.

*   **Method**: We created a representative subset of **500 queries** per benchmark by performing k-means clustering (k=500) on the TF-IDF vectors of the queries.
*   **Format**: The final query sets are `json` files, where each entry contains:
    *   `query`: The natural language query.
    *   `document_ids`: A list of the ground-truth unique document IDs required to answer the query.

### 🗂️ Final Corpus Sizes

The resulting corpus files for retrieval are as follows:

| Benchmark       | Corpus Size |
| --------------- | ----------- |
| `bird.jsonl`      | 2.0G        |
| `feverous.jsonl`  | 106M        |
| `mt_raig.jsonl`   | 79M         |
| `spider.jsonl`    | 36M         |
| `multi_hop_rag.jsonl` | 9.4M        |
| `fetaqa.jsonl`    | 7.5M        |
| `tabfact.jsonl`   | 5.3M        |
| `ottqa.jsonl`     | 2.6M        |
| `table_bench.jsonl` | 2.5M        |


## 🚀 Retrievers

We implement and evaluate a suite of retrieval methods, ranging from classic sparse and dense retrievers to advanced agentic and hybrid approaches.

### 1. Sparse Retriever (BM25)

We implement a sparse retriever using the **BM25** score function. Our implementation leverages [Pyserini](https://github.com/castorini/pyserini), a highly efficient Python wrapper for the Java-based Lucene search library.

### 2. Dense Retriever

For dense retrieval, we use a two-stage process:
*   **Embedding Serving**: We use [Infinity](https://github.com/michaelfeil/infinity) to serve text embeddings via a high-throughput, low-latency REST API.
*   **Vector Indexing**: Embeddings are indexed and searched using [FAISS](https://github.com/facebookresearch/faiss). The GPU-accelerated implementation offers extremely fast exhaustive search. For CPU-only environments, this can be switched to an approximate nearest neighbor method like HNSW.

#### ✨ Embedding Model Benchmark

We benchmarked several top-performing embedding models from the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (under 2B parameters) to find the best fit for our mixed-modality corpus.

**Models Tested:**
*   [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
*   [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
*   [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
*   [Snowflake/snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)
*   [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
*   [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
*   [infly/inf-retriever-v1-1.5b](https://huggingface.co/infly/inf-retriever-v1-1.5b)
*   [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
*   [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
*   [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
*   [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)
*   [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

**Performance Results:**

| Embedding Model                      | Recall@5 |
| ------------------------------------ | -------- |
| `inf-retriever-v1-1.5b`              | **0.769**  |
| `bge-m3`                             | 0.763    |
| `UAE-Large-V1`                       | 0.761    |
| `Qwen/Qwen3-Embedding-0.6B`          | 0.753    |
| `gte-modernbert-base`                | 0.749    |
| `multilingual-e5-large-instruct`     | 0.747    |
| `gte-multilingual-base`              | 0.744    |
| `snowflake-arctic-embed-l-v2.0`      | 0.743    |
| `nomic-embed-text-v2-moe`            | 0.727    |
| `jina-embeddings-v3`                 | 0.715    |
| `modernbert-embed-base`              | 0.707    |
| `snowflake-arctic-embed-s`           | 0.664    |

| Embedding Model                      | QPS (Queries/Sec) |
| ------------------------------------ | ----------------- |
| `snowflake-arctic-embed-s`           | 79.1              |
| `modernbert-embed-base`              | 75.9              |
| `jina-embeddings-v3`                 | 75.5              |
| `gte-multilingual-base`              | 74.6              |
| `snowflake-arctic-embed-l-v2.0`      | 72.2              |
| `nomic-embed-text-v2-moe`            | 71.5              |
| `bge-m3`                             | 66.9              |
| `UAE-Large-V1`                       | 65.2              |
| `multilingual-e5-large-instruct`     | 63.5              |
| `Qwen/Qwen3-Embedding-0.6B`          | 62.1              |
| `gte-modernbert-base`                | 61.9              |
| `inf-retriever-v1-1.5b`              | 60.2              |

Based on these results, we selected **`BAAI/bge-m3`** for our dense retrieval stages. It provides high recall with a very reasonable QPS, offering a great balance between performance and speed.

### 3. Dense Retriever with Reranker

We enhance our dense retriever with a powerful cross-encoder reranker. We use [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2), one of the top-performing models on reranking leaderboards, to re-score the top candidates from the dense retriever.

### 4. Dense Retriever with Query Decomposition

This retriever uses an LLM to decompose complex queries into simpler sub-queries, which are then executed individually.
*   **LLM Serving**: We provide two implementations: one using [LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com/) for smaller workloads, and a high-performance version using [vLLM](https://github.com/vllm-project/vllm) for large-scale inference.
*   **Model**: We use few-shot prompting with [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) as the base model for decomposition.

We also implement a hybrid version that combines **Decomposition with the Reranker**.

### 5. Agentic Retriever (ReAct)

We implement a ReAct (Reasoning and Acting) agent, which iteratively reasons about the task and chooses an action (`Search` or 'Finish') to make progress. To ensure the LLM's output strictly follows the required format (`Search[keywords]` and 'Finish'), we use [guidance](https://github.com/guidance-ai/guidance) for constrained generation.

### 6. ARM Retriever (Core Implementation)

This is our implementation of the paper's core contribution. Its key components are:
*   **Base Retrievers**: Utilizes the **BM25** and **Dense** retrievers described above as its foundation.
*   **Constrained N-gram Generation**: We use the principles from **[RecLM-gen](https://arxiv.org/abs/2505.03336)**, a method for constrained generation in recommender systems for item recommendation that ensures an LLM only generates in-domain outputs. We adapt this to recommend valid n-grams from our corpus, with an underlying implementation based on the `transformers` library to support beam search.
*   **Main Retrieval Loop**: The multi-step retrieval process is powered by **vLLM**. Its automatic KV caching is highly effective, as the prompt (containing few-shot examples and the history of previous steps) is efficiently reused across the generation steps for each query.
*   **MIP Solver**: For the final Mixed Integer Programming (MIP) step that selects the optimal set of documents, we use [PuLP](https://github.com/coin-or/pulp) as the modeler. Our default solver is [Gurobi](https://www.gurobi.com/), known for its high performance.
*  **LLM models**: The main retrieval process with vLLM uses google/gemma-3-27b-it as its base model, while the constrained n-gram generation component leverages meta-llama/Meta-Llama-3.1-8B-Instruct, as the original RecLM-gen code was specifically designed to work with this model.


# 🛠️ How to Reproduce Results

This section shows the project's architecture and the core components for running retrieval and evaluation.

### 📂 Project Structure

The project has two main directories:

*   `src/`: Contains the implementation for retrievers, evaluation, and utilities. Subdirectories group related code (e.g., `src/retrieval/`).
*   `scripts/`: Contains scripts to run the pipeline steps like data processing, indexing, retrieval, and evaluation.

### 🧩 Core Components

The framework uses two base classes for a consistent structure.

#### The Retriever Interface (`src/retrieval/base.py`)

Located at `src/retrieval/base.py`, the `BaseRetriever` class acts as a blueprint for all retrievers. It ensures every retriever has two main functions:

*   `index(...)`: Takes a corpus file and builds a searchable index.
*   `retrieve(...)`: Takes queries and returns the top-k `RetrievalResult` objects, which contain the score, retrieved text, and metadata.

This common interface allows any retriever to be swapped in and out of the pipeline easily.

#### The Evaluation Framework (`src/evaluation/metrics.py`)

Located at `src/evaluation/metrics.py`, the `EvaluationMetrics` class handles all performance scoring.

*   **Metrics**: It computes **Precision**, **Recall**, **F1-Score**, and **Perfect Recall** at different `n` values (e.g., @5, @10).
*   **Granularity**: It automatically handles the object-level vs. document-level evaluation by de-duplicating results before scoring.

Usage is straightforward:
1.  Instantiate the class: `evaluator = EvaluationMetrics(n_values=[5, 10, 20])`
2.  Calculate scores: `results_df = evaluator.calculate_metrics(ground_truth, predictions)`
3.  Plot results: `evaluator.visualize_results(results_df, title="BM25 Performance")`

## 🚀 Getting Started

Follow these steps to set up the project environment and download the necessary data.

### 1. Clone the Repository

First, clone this repository and navigate into the project directory:

```bash
git clone https://github.com/apostolhskouk/ARM.git
cd ARM/
```

### 2. Set Up the Conda Environment and instal code in editable mode

This project uses Conda to manage dependencies. Assuming you have Conda installed, create and activate the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate arm
pip install -e .
```

### 3. Download the Pre-processed Data

The data preparation process is resource-intensive and time-consuming. To make it easier to get started, we have pre-processed all the datasets and uploaded them to Hugging Face.

> **Note**: The downloaded data includes the serialized corpora and benchmark query sets. It does **not** include the pre-built vector indexes for the embedding model comparison, since they are 12 different embedding indexes for the 9 different datasets, requiring ~250GB of storage. If you wish to reproduce the embedding benchmark, you will need to run the indexing scripts yourself as described below.

To download the data, run the following command in your terminal:

```bash
huggingface-cli download ApostolosK/arm_reproduction_data_processed --repo-type dataset --include "assets/*" --local-dir .
```

This will download the data into the `assets/` directory. You can explore `assets/all_data/serialized_data` and `assets/all_data/benchmarks` to familiarize yourself with the datasets and benchmarks format.

## 🧪 Running the Experiments

Once the environment is set up and the data is in place, you can run the experiments using the scripts in the `scripts/` directory.

> **Note**: All evaluation scripts use [Weights & Biases](https://wandb.ai/) for logging. You must replace `your_entity` with your personal or team's wandb entity name for the scripts to run correctly.

### 1. FEVEROUS Granularity Benchmark

This script evaluates the impact of different table serialization methods (table vs. row vs. cell) on retrieval performance using the FEVEROUS dataset. The specific data files for this experiment are located in `assets/feverous/serialized_output/`.

```bash
python scripts/run_feverous_benchmark.py --wandb_entity your_entity
```

The script will automatically use existing indexes if they were downloaded or create them if they are missing.

### 2. Embedding Model Comparison

This two-step process benchmarks the performance of all selected embedding models across the datasets.

**Step 2a: Indexing**

First, build the indexes for each embedding model.

```bash
python scripts/run_embedding_indexing.py
```

⚠️ **Warning**: This step is extremely time-consuming and resource-intensive, as it builds a separate, large index for each of the 12 embedding models.

**Step 2b: Evaluation**

Once indexing is complete, run the evaluation.

```bash
python scripts/run_embedding_evaluation.py --wandb_entity your_entity
```

### 3. Full Evaluation Across All Datasets

This is the main experiment to reproduce the final results, evaluating all retrieval methods across all benchmarks.

**Step 3a: Indexing**

First, ensure the necessary indexes for the primary retriever are built.

```bash
python scripts/run_all_indexing.py
```

If you downloaded the data from Hugging Face, this script will verify the files and finish quickly. Otherwise, it will build the required indexes.

**Step 3b: Evaluation**

Finally, run the main evaluation script. This will test all implemented retrievers (BM25, Dense, Reranker, Agentic, ARM, etc.) on all datasets.

```bash
python scripts/run_all_evaluation.py --wandb_entity your_entity
```


# 💻 Using on Your Own Corpus

You can use the retrievers in this project on your own data. The general workflow is:

1. **Prepare Data**: Create a `jsonl` file with your documents.
2. **Choose & Configure a Retriever**: Select a retriever from the list below and instantiate it with your desired parameters.
3. **Index & Retrieve**: Use the common `.index()` and `.retrieve()` methods to process your data and get results.

### 1. Prepare Your Data

First, prepare your corpus as a `jsonl` file where each line is a single JSON object. You can include as many fields as you need.

> **Note**: There is no need to provide a unique ID field in your `jsonl` file. The underlying system automatically adds a unique identifier to each object internally.

**Example `my_corpus.jsonl`:**
```json
{"doc_title": "Paper A", "content": "The first document is about sparse retrieval.", "year": 2020}
{"doc_title": "Paper B", "content": "The second document covers dense retrieval methods.", "year": 2021}
```

### 2. Choose and Configure a Retriever

#### BM25 Retriever

**Example:**
```python
from src.retrieval.bm25 import PyseriniBM25Retriever
retriever = PyseriniBM25Retriever()
```

#### Dense Retriever

Uses vector embeddings for retrieval.

**Key Parameters:**
- `model_name_or_path`: The Hugging Face embedding model to use.
- `use_vllm_indexing`: Flag to use vllm for faster embedding computation (not optimal but better than the default sentence-transformers)
- `use_infinity_indexing`: Flag to use infinity_emb for the fastest embedding computation (default is sentence-transformers).

**Example:**
```python
from src.retrieval.dense import FaissDenseRetriever
retriever = FaissDenseRetriever(
    model_name_or_path="infly/inf-retriever-v1-1.5b",
    use_infinity_indexing=True
)
```

⚠️ **Dependency Note**: For maximum efficiency, set `use_infinity_indexing=True`. However, infinity_emb requires transformers<4.48 while vllm requires transformers>=4.52. The imports are commented out in the source. To use infinity_emb, you must uncomment its import, downgrade transformers, run indexing, and then upgrade again. This is because vllm is needed for the ARM implementation and the query decomposition.

#### Dense Retriever with Reranker

Adds a cross-encoder reranking step.

**Key Parameters:**
- `embedding_model_name`: The embedding model for initial retrieval.
- `reranker_model_name`: The Hugging Face reranker model.
- `k_multiplier`: A multiplier for k to fetch enough candidates for the reranker to be effective (e.g., `k_multiplier=3` fetches 3*k items).

**Example:**
```python
from src.retrieval.dense_rerank import DenseRetrieverWithReranker
retriever = DenseRetrieverWithReranker(
    embedding_model_name="infly/inf-retriever-v1-1.5b",
    reranker_model_name="mixedbread-ai/mxbai-rerank-large-v2",
    k_multiplier=3
)
```

#### Dense Retriever with Query Decomposition

Uses an LLM to break down complex queries.


**Key Parameters:**
- `embedding_model_name`: The embedding model.
- `model_name`: The LLM for decomposition (via vllm or ollama).
- `use_vllm`: Set to True to use the high-performance vLLM backend (default). To modify its behavior (e.g., prompts), edit `src/utils/query_decomposition_vllm.py`.
- `decomposition_cache_folder`: Optional path to a folder where a `decomposition.json` file will be stored, mapping queries to their decompositions for future use.

**Example:**
```python
from src.retrieval.dense_decomp import DenseRetrieverWithDecomposition
retriever = DenseRetrieverWithDecomposition(
    embedding_model_name="infly/inf-retriever-v1-1.5b",
    model_name="gaunernst/gemma-3-27b-it-int4-awq",
    use_vllm=True
)
```

#### Dense Retriever with Decomposition and Reranker

Combines the two previous methods.

**Parameters:** A combination of the parameters from the Decomposition and Reranker retrievers.

**Example:**
```python
from src.retrieval.dense_decomp_rerank import DenseRetrieverWithDecompositionAndReranker
retriever = DenseRetrieverWithDecompositionAndReranker(
    embedding_model_name="infly/inf-retriever-v1-1.5b",
    reranker_model_name="mixedbread-ai/mxbai-rerank-large-v2",
    model_name="gaunernst/gemma-3-27b-it-int4-awq"
)
```

#### ReAct Retriever

An agentic retriever that iteratively searches.

**Key Parameters:**
- `dense_model_name_or_path`: The embedding model for the Search tool.
- `model_path`: Path to the agent's LLM. The implementation uses guidance with a llama-cpp-python backend for performance with GGUF models. To use standard Hugging Face models, the source code needs to be modified slightly.
- `max_iterations`: The maximum number of search steps (default: 5).
- `k_react_search`: Documents to retrieve per search step (default: 5).

**Example:**
```python
from src.retrieval.react import ReActRetriever
retriever = ReActRetriever(
    dense_model_name_or_path="infly/inf-retriever-v1-1.5b",
    model_path="Qwen2.5-32B-Instruct-Q4_K_M.gguf"
)
```

#### ARM Retriever

The core implementation of the paper's method.


**Key Parameters:**
- `vllm_model_path`: The main LLM for the retrieval loop (standard HF formats are best; GGUF is not optimized here).
- `ngram_llm_model_path`: The LLM for constrained n-gram generation (implemented to work best with Llama models).
- `embedding_model_name`: The dense embedding model.
- `keyword_extraction_beams`: Number of beams for vLLM to use when extracting keywords from the query.
- `corpus_ngram_min_len` / `corpus_ngram_max_len`: Min/max token length for n-grams used in alignment.
- `keyword_rephrasing_beams`: Number of beams to generate keyword alignments.
- `vllm_quantization`: Type of dynamic quantization for vLLM (e.g., fp8).
- `alignment_retrieval_k`: Documents to retrieve for each generated n-gram using BM25
- `compatibility_semantic_weight` / `compatibility_exact_weight`: Weights for semantic vs. lexical match in the MIP solver formulation.
- `expansion_steps`: Number of neighbor expansion steps per retrieved object before feeding to MIP.
- `expansion_k_compatible`: Compatible documents to fetch during each expansion step.
- `mip_k_select`: Final number of documents to select via the MIP solver, which are then passed to the final LLM call.

**Example:**
```python
from src.retrieval.arm import ARMRetriever
retriever = ARMRetriever(
    vllm_model_path="gaunernst/gemma-3-27b-it-int4-awq",
    ngram_llm_model_path="meta-llama/Meta-Llama-3-8B",
    embedding_model_name="infly/inf-retriever-v1-1.5b"
)
```

### 3. Index and Retrieve

Since all retriever classes implement the `BaseRetriever` interface from `src/retrieval/base.py`, the methods for indexing and retrieving are the same for all of them.

#### Indexing Your Data

The `.index()` method builds a searchable index from your jsonl file.

```python
# This works for any 'retriever' object created above
retriever.index(
    input_jsonl_path="path/to/my_corpus.jsonl",
    output_folder="./my_index",
    field_to_index="content",
    metadata_fields=["doc_title", "year"] # filed from jsonl to be included
)
```

For the `ARMRetriever`, `output_folder` must be a list of two paths: `["./bm25_index_path", "./faiss_index_path"]`.

#### Retrieving Results

The `.retrieve()` method queries the index and returns the top results. An error will occur if the index is not found at the specified path.

```python
# This also works for any indexed 'retriever'
results = retriever.retrieve(
    nlqs=["query about sparse retrieval", "another query"],
    output_folder="./my_index",
    k=10
)
```

The `k` parameter is ignored by the `ReActRetriever` and `ARMRetriever`, as they use their own internal logic to determine the number of results to return.

### 4. Understanding the Output

The `.retrieve()` method returns a `List[List[RetrievalResult]]`, which is a list of result lists (one list for each query). Each `RetrievalResult` object corresponds to one retrieved object and contains:

- `score` (float): The relevance score assigned by the retriever (this varies by method).
- `object` (str): The retrieved text from the `field_to_index`.
- `metadata` (dict): A dictionary containing the metadata fields and their values for the retrieved object.
