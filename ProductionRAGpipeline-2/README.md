# Hybrid Retrieval Pipeline (BM25 + FAISS + ColBERT-style + Cross-Encoder)

## Overview

This project implements a **multi-stage hybrid retrieval system** designed to maximize relevance in document search.

Instead of relying on a single retrieval method, this pipeline combines:

- Keyword-based search (BM25)
- Semantic search (FAISS + embeddings)
- Token-level similarity (ColBERT-inspired scoring)
- Deep reranking (Cross-Encoder)

This layered approach significantly improves retrieval accuracy for real-world AI applications like chatbots, RAG systems, and search engines.

---

## Libraries Used

### 1. FAISS
Used for **fast vector similarity search**.
- Stores dense embeddings
- Enables semantic retrieval using inner product similarity

### 2. Sentence Transformers
Used for:
- Generating embeddings (`BAAI/bge-m3`)
- Cross-Encoder reranking (`ms-marco-MiniLM`)

### 3. BM25 (rank-bm25)
Used for **keyword-based retrieval**
- Captures exact term matches
- Complements semantic search

### 4. LangChain Text Splitter
Used for splitting large documents into manageable chunks with overlap.

### 5. NumPy
Used for:
- Vector operations
- Normalization
- Similarity computation

### 6. Groq API
Used to connect to an LLM for response generation.

### 7. NLTK
Provides stopwords (though not heavily used here).

---

## Pipeline Architecture

### Step 1: Text Extraction
- Reads raw text from a file.

### Step 2: Chunking
- Splits text into overlapping chunks.
- Ensures context continuity.

---

### Step 3: Dual Indexing

#### FAISS Index (Semantic)
- Converts chunks into embeddings.
- Stores them for similarity search.

#### BM25 Index (Keyword)
- Tokenizes text.
- Scores based on term frequency.

---

### Step 4: Hybrid Retrieval (Core Idea)

Instead of choosing one method, we combine both.

#### Process:
1. Query → embedding → FAISS search
2. Query → tokens → BM25 scoring
3. Combine results using **Reciprocal Rank Fusion (RRF)**

#### Why this matters:
- FAISS captures meaning
- BM25 captures exact keywords
- RRF balances both

---

### Step 5: ColBERT-Inspired Token Matching

This is a **key innovation layer**.

Instead of comparing whole embeddings:
- Break query into tokens
- Break document into tokens
- Compare each query token with all document tokens

#### MaxSim Strategy:
For each query token:
- Find the most similar token in the document
- Sum all max similarities

#### Why this is powerful:
- Preserves fine-grained meaning
- Handles partial matches better
- More precise than sentence-level embeddings

---

### Step 6: Cross-Encoder Reranking

Final refinement step.

#### What it does:
- Takes (query, document) pairs
- Scores them jointly
- Produces highly accurate rankings

#### Why needed:
- Bi-encoders are fast but approximate
- Cross-encoders are slow but precise

---

## Final Pipeline Flow
User Query
↓
Hybrid Search (FAISS + BM25)
↓
ColBERT-style Token Scoring
↓
Cross-Encoder Reranking
↓
Top-K Relevant Documents
↓
(Optional) LLM Response Generation


---

## Key Concept

This system follows a **"coarse → fine" retrieval strategy**:

1. **Broad Recall**
   - Hybrid search retrieves many candidates

2. **Mid-Level Precision**
   - Token-level similarity refines results

3. **High Precision**
   - Cross-encoder selects the best

---

## Why This Architecture is Strong

- Avoids weaknesses of single-method retrieval
- Balances speed vs accuracy
- Scales well for large datasets
- Matches modern RAG system design patterns

---

## Limitations

- Token-level encoding is slow (not optimized like real ColBERT)
- Chunking is static (not adaptive)
- No caching → recomputes embeddings every run
- No batching → inefficient at scale

---

## Future Improvements

- Dynamic chunking based on document type
- ANN indexing (IVF, HNSW) instead of flat FAISS
- True ColBERT implementation
- Query expansion
- Reranker ensembles
- Caching + persistence layer

---

## How to Run

1. Install dependencies: pip install -r requirements.txt

2. Add your Groq API key

3. Place your data in `sample_data`

4. Run: python script.py


### 📌 NOTE: Sample Dataset Included

A sample `.txt` dataset has been provided along with this project to help you quickly understand and test the pipeline.

* This dataset is **small and intentionally simplified** for demonstration purposes.
* It is **not representative of real-world data**, which is typically noisy, unstructured, and diverse.
* The current system may perform well on this dataset, but that does **not guarantee production-level performance**.

⚠️ **Important:**
For realistic evaluation, you should replace this sample file with larger, noisier, and domain-diverse datasets. This will help you properly test retrieval quality, ranking robustness, and failure handling.

Use the sample dataset only for:

* Initial testing
* Debugging
* Understanding pipeline behavior

Do not use it as a benchmark for system accuracy or real-world readiness.
