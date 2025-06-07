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

Note that the format and the pre-processing for the last 5 datasets was taken from **[TARGET](https://arxiv.org/abs/2505.11545)**.

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

Based on these results, we selected **`infly/inf-retriever-v1-1.5b`** for our dense retrieval stages. Despite being a larger model, it provides the best recall with a very reasonable QPS, offering a great balance between performance and speed.

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

