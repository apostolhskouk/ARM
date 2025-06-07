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
