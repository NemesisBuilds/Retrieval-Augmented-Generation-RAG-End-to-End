# Hybrid RAG System (FAISS + BM25 + CrossEncoder)

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline with multiple retrieval improvements including:

- Dataset Routing
- Hybrid Search (Dense + Sparse Retrieval)
- Cross-Encoder Reranking
- LLM Answer Generation

The system efficiently searches across multiple document collections, retrieves relevant chunks, and generates answers using an LLM.

---

# Architecture Overview

```
User Query
    │
    ▼
Dataset Router (FAISS centroid search)
    │
    ▼
Hybrid Retrieval
  ├─ Dense Search (FAISS)
  └─ Sparse Search (BM25)
    │
    ▼
Rank Fusion (RRF)
    │
    ▼
CrossEncoder Reranker
    │
    ▼
Top Context Chunks
    │
    ▼
LLM Generation
```

---

# Key Features

## 1. Dataset Router

Instead of searching the entire corpus, the system first determines which datasets are relevant.

- Uses **KMeans clustering** to generate centroid vectors
- Stores centroids inside a **router FAISS index**
- Query embeddings are matched against these centroids

This significantly reduces search space.

---

## 2. Hybrid Retrieval (Dense + Sparse)

The system combines two retrieval methods.

### Dense Retrieval

Uses:

- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS similarity search

Captures **semantic similarity between queries and chunks**.

### Sparse Retrieval

Uses:

- **BM25 lexical search**
- Tokenization
- Stopword removal
- Stemming

Captures **keyword matching signals**.

---

## 3. Retrieval Rank Fusion (RRF)

Results from FAISS and BM25 are combined using **Reciprocal Rank Fusion**.

```
RRF Score = 1 / (k + rank)
```

This method is robust because it combines rankings without needing score normalization.

---

## 4. Cross-Encoder Reranking

Top candidates are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike embedding models, a **CrossEncoder evaluates the query and document together**, producing more accurate relevance scores.

---

## 5. LLM Answer Generation

After reranking:

1. Top chunks are combined into a **context block**
2. A **prompt template** is created
3. The prompt is sent to an **LLM (Groq API)**

The model is instructed to **answer strictly using the retrieved context**.

---

# Project Structure

```
.
├── sample_data/
│   ├── dataset1.txt
│   ├── dataset2.txt
│
├── vectorStorageFiles/
│   ├── *_faiss_index.faiss
│   ├── *_all_chunks.json
│
├── bm25StorageFiles/
│   ├── *_bm25_index.pkl
│   ├── *_tokenized_corpus.pkl
│
├── router_indices.faiss
├── router_metadata.json
│
├── build_indices.py
├── rag_pipeline.py
│
└── README.md
```

---

# Index Building Pipeline

The retrieval infrastructure is built through the following steps:

1. Document ingestion  
2. Chunking (RecursiveCharacterTextSplitter)  
3. Embedding generation  
4. KMeans clustering  
5. Router index construction  
6. FAISS index creation  
7. BM25 index creation  

All indices are **stored on disk to avoid recomputation**.

---

# Installation

Install dependencies:

```bash
pip install faiss-cpu
pip install sentence-transformers
pip install rank-bm25
pip install nltk
pip install scikit-learn
pip install langchain-text-splitters
pip install groq
```

Download NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```

---

# Running the System

## 1. Add documents

Place `.txt` files inside:

```
sample_data/
```

---

## 2. Build indices

Run:

```bash
python build_indices.py
```

This creates:

- FAISS indices
- BM25 indices
- Router index

---

## 3. Run the RAG pipeline

```bash
python rag_pipeline.py
```

Example:

```
User: What is reinforcement learning?

Answer: ...
```

---

# Technologies Used

- **FAISS** – dense vector search  
- **Sentence Transformers** – embeddings  
- **BM25 (rank-bm25)** – lexical search  
- **CrossEncoder** – reranking  
- **KMeans (scikit-learn)** – dataset routing  
- **Groq API** – LLM inference  
- **LangChain Text Splitters** – document chunking  

---

# Future Improvements

Potential extensions:

- Query rewriting  
- Multi-query retrieval  
- Metadata filtering  
- Chunk deduplication  
- Streaming LLM responses  
- RAG evaluation benchmarks
