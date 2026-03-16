Hybrid RAG System (FAISS + BM25 + CrossEncoder)

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline with multiple retrieval improvements including:

Dataset Routing

Hybrid Search (Dense + Sparse Retrieval)

Cross-Encoder Reranking

LLM Answer Generation

The system is designed to efficiently search across multiple document collections, retrieve the most relevant text chunks, and generate answers using a language model.

Architecture Overview

The retrieval pipeline consists of four major stages:

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
Key Features
1. Dataset Router

Instead of searching every document collection, the system first determines which datasets are relevant.

Uses KMeans clustering to generate centroid vectors for each dataset

Stores them inside a router FAISS index

Query embeddings are matched against these centroids

This significantly reduces unnecessary search space.

2. Hybrid Retrieval (Dense + Sparse)

The system combines two retrieval approaches:

Dense Retrieval

Uses:

Sentence Transformers (all-MiniLM-L6-v2)

FAISS vector similarity search

Captures semantic similarity between queries and document chunks.

Sparse Retrieval

Uses:

BM25 lexical search

Tokenization + stopword removal + stemming

Captures keyword matching signals.

3. Retrieval Rank Fusion (RRF)

Results from FAISS and BM25 are combined using Reciprocal Rank Fusion:

RRF Score = 1 / (k + rank)

This approach is robust because it combines rankings without requiring score normalization.

4. Cross-Encoder Reranking

The top retrieved candidates are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

Unlike embedding models, a CrossEncoder jointly evaluates the query and document, producing more accurate relevance scores.

5. LLM Answer Generation

After reranking:

Top chunks are combined into a context block

A prompt template is created

The prompt is sent to a Groq-hosted LLM

The model is instructed to answer strictly using the retrieved context.

Project Structure
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
Index Building Pipeline

The system builds the retrieval infrastructure through the following steps:

Document Ingestion

Chunking (RecursiveCharacterTextSplitter)

Embedding Generation

KMeans Clustering

Router Index Construction

FAISS Index Creation

BM25 Index Creation

All indices are stored on disk to avoid recomputation.

Installation
pip install faiss-cpu
pip install sentence-transformers
pip install rank-bm25
pip install nltk
pip install scikit-learn
pip install langchain-text-splitters
pip install groq

Download stopwords:

import nltk
nltk.download("stopwords")
Running the System
Step 1 — Add documents

Place .txt files inside:

sample_data/
Step 2 — Build indices

Run:

python build_indices.py

This will create:

FAISS indices

BM25 indices

Router index

Step 3 — Run the RAG pipeline
python rag_pipeline.py

Example:

User: What is reinforcement learning?

Answer: ...
Technologies Used

FAISS – dense vector search

Sentence Transformers – embedding generation

BM25 (rank-bm25) – lexical search

CrossEncoder – reranking

KMeans (scikit-learn) – dataset routing

Groq API – LLM inference

LangChain Text Splitters – document chunking

Why This Architecture Matters

Traditional RAG pipelines struggle with:

searching large multi-dataset corpora

balancing semantic and lexical retrieval

ranking noisy chunks

This implementation improves retrieval quality by combining:

Routing

Hybrid Search

Rank Fusion

Cross-Encoder Reranking

Possible Improvements

Future extensions could include:

Query rewriting

Multi-query retrieval

Metadata filtering

Chunk deduplication

Streaming LLM responses

Evaluation with RAG benchmarks
