"""Hybrid-Retrieval: bm25 Retrieval + Bi-Encoder followed by Reraking Using a Cross-Encoder"""
""" Uses token level vector embeddings AFTER retrieval INSPIRED by the ColBERT technique ( not pure ColBERT )"""

import faiss 
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from rank_bm25 import BM25Okapi
from pathlib import Path
import string
from nltk.corpus import stopwords

#=====================-- Initializations --======================#

#Folder/File paths using Pathlib:- 
BASE_DIR = Path(__file__).resolve().parent
file_path = str(BASE_DIR / "sample_data" / "data.txt")

#Models:-
encoding_model = SentenceTransformer('BAAI/bge-m3')
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker = CrossEncoder(model_name)

#LLM API
groq_model = Groq(api_key="YOUR_API_KEY")

#Prompt to use in the LLM :-
prompt = """
    You are an expert at answering the user's question based on the 
    context provided to you.

        Your job is to use ONLY the context provided to you 
        to answer the user's question.
        It should be completely based on the context provided.

        ONLY output the answer to the user's question and 
        don't add any information on your own in the answer.


    If the context for the user's question is not enough OR is 
    completely irrelevant, reply with these EXACT WORDS:- "No relevant information found."

    CONTEXT:-
    {CONTEXT}

    USER QUESTION:-
    {USER_QUESTION}

"""

def LLM(prompt: str, max_tokens: int = 256):
    """
    Sends a prompt to the Groq LLM API and returns the generated response.

    Args:
        prompt (str): The input text/question to send to the model.
        max_tokens (int): Maximum number of tokens the model should generate.

    Returns:
        str: The generated response from the LLM.
    """

    response = groq_model.chat.completions.create(
        messages = [{"role": "user",
                     "content": prompt}],
        model= "moonshotai/kimi-k2-instruct-0905",
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

def tokenizer(text: str):
    """
    Converts text to lowercase and splits it into words.

    Args:
        text (str): Input text.

    Returns:
        list: A list of word tokens.
    """
    text = text.lower()
    return text.split()

def extract_text(file_path: str):
    """
    Reads and returns the content of a text file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Full text content of the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def chunk_text(text: str, chunk_size=500, chunk_overlap=50):
    """
    Splits a large text into smaller overlapping chunks.

    Args:
        text (str): The full input text.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
    
def create_embeddings(chunks: list):
    """
    Converts text chunks into vector embeddings using a bi-encoder model.

    Args:
        chunks (list): List of text chunks.

    Returns:
        numpy.ndarray: Array of embeddings (float32).
    """
    embeddings = encoding_model.encode(chunks)
    embeddings = np.array(embeddings,dtype="float32")
    return embeddings

def build_faiss_index(embeddings, dim):
    """
    Builds a FAISS index using inner product similarity.

    Args:
        embeddings (numpy.ndarray): Vector embeddings.
        dim (int): Dimension of embeddings.

    Returns:
        faiss.Index: FAISS index with normalized embeddings.
    """
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def build_bm25_index(chunks):
    """
    Builds a BM25 keyword-based retrieval index.

    Args:
        chunks (list): List of text chunks.

    Returns:
        BM25Okapi: BM25 index object.
    """
    tokenized_chunks = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def hybrid_search(query, index, bm25_index, texts, top_n=30):
    """
    Performs hybrid retrieval using FAISS (semantic) and BM25 (keyword),
    and combines results using Reciprocal Rank Fusion (RRF).

    Args:
        query (str): User query.
        index (faiss.Index): FAISS vector index.
        bm25_index (BM25Okapi): BM25 index.
        texts (list): Original text chunks.
        top_n (int): Number of top results to retrieve.

    Returns:
        list: Top candidate text chunks after hybrid scoring.
    """
    query_vector = encoding_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_n)

    faiss_results = {idx: (1 / (60 + rank)) for rank, idx in enumerate(indices[0])}

    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    bm25_results = {}

    for i in range(len(bm25_scores)):
        if bm25_scores[i] > 0:
            bm25_results[i] = bm25_scores[i]

    sorted_bm25 = sorted(bm25_results.items(), key= lambda x: x[1], reverse= True)

    for rank, (idx,score) in enumerate(sorted_bm25[:top_n]):
        rrf_score = 1/(60 + rank)
        faiss_results[idx] = faiss_results.get(idx, 0) + rrf_score

    sorted_indices = sorted(faiss_results.items(), key=lambda x: x[1], reverse=True)

    candidate_indices = [idx for idx, score in sorted_indices[:top_n]]

    return [texts[i] for i in candidate_indices]

def rerank_results(query, candidates, top_k=3):
    """
    Reranks retrieved documents using a Cross-Encoder for better relevance.

    Args:
        query (str): User query.
        candidates (list): List of candidate text chunks.
        top_k (int): Number of top results to return.

    Returns:
        list: Top-k most relevant documents.
    """
    # Create pairs of [query, document]
    pairs = [[query, doc] for doc in candidates]
    
    # Get scores (CrossEncoder does the heavy lifting)
    scores = reranker.predict(pairs)
    
    # Pair scores with candidates and sort
    scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # Return top K
    return [doc for doc, score in scored_results[:top_k]]



def ColBERT_Concept(user_query: str, relevant_documents: list, top_n: int = 10):
    """
    Applies a ColBERT-inspired token-level similarity scoring. (NOTE: Not pure ColBERT)

    Instead of comparing whole embeddings, it:
    - Compares each query token to all document tokens
    - Picks the best match per token (MaxSim)
    - Sums scores to get final relevance

    Args:
        user_query (str): The user query.
        relevant_documents (list): Candidate documents.
        top_n (int): Number of top results to return.

    Returns:
        list: Ranked documents with scores.
    """
    results = []

    # Tokenize query (no stopword removal for better semantic retention)
    tokenized_query = user_query.lower().split()

    # Encode query tokens once
    Q = encoding_model.encode(tokenized_query)
    Q = np.array(Q, dtype="float32")

    # Normalize query embeddings
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    for doc in relevant_documents:
        # Tokenize document
        tokenized_doc = doc.lower().split()

        # Encode document tokens
        D = encoding_model.encode(tokenized_doc)
        D = np.array(D, dtype="float32")

        # Normalize document embeddings
        D = D / np.linalg.norm(D, axis=1, keepdims=True)

        # Similarity matrix (q_tokens x d_tokens)
        S = np.dot(Q, D.T)

        # MaxSim: best match per query token
        max_sim = np.max(S, axis=1)

        # Final score
        score = np.sum(max_sim)

        results.append({
            "score": float(score),
            "chunk": doc
        })

    # Sort by score (descending)
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_n]

        
def retrieve(query: str, index, bm25_index, texts):
    """
    Full retrieval pipeline:
    1. Hybrid search (FAISS + BM25)
    2. Token-level scoring (ColBERT-style)
    3. Cross-encoder reranking

    Args:
        query (str): User query.
        index (faiss.Index): FAISS index.
        bm25_index (BM25Okapi): BM25 index.
        texts (list): Text chunks.

    Returns:
        list: Final top-ranked documents.
    """
    candidates = hybrid_search(query=query,index=index, bm25_index=bm25_index, texts=texts)

    ColBERT_Concept_result = ColBERT_Concept(user_query=query, relevant_documents=candidates)

    selected = [t["chunk"] for t in ColBERT_Concept_result]

    results = rerank_results(candidates=selected, query=query)

    return results
def main():
    """
    Runs the full pipeline:
    - Extracts text
    - Splits into chunks
    - Builds retrieval indexes
    - Accepts user query
    - Retrieves and refines relevant documents
    - Prints final context

    This acts as the entry point of the script.
    """
    print("Extracting and Chunking...")
    raw_text = extract_text(file_path=file_path)
    chunked_text = chunk_text(text=raw_text)
    
    print("Building Indexes...")
    embeddings = create_embeddings(chunks=chunked_text)
    dim = embeddings.shape[1]
    bm25_index = build_bm25_index(chunks=chunked_text)
    index = build_faiss_index(embeddings=embeddings, dim=dim)
    
    

    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() in ["exit", "quit"]:
            break


    
        results = retrieve(
            query=user_input,
            index=index,
            bm25_index=bm25_index,
            texts=chunked_text
        )   

        context = """ """

        for i, text in enumerate(results, 1):
            context += f"\nDocument {i}:\n{text}"

        print("="*80)
        print("Relevant context collected:-")
        print(context)

        master_prompt = prompt.format(
            CONTEXT = context,
            USER_QUESTION = user_input
        )

        answer = LLM(prompt=master_prompt)
        print("="*80)
        print("LLM Generated Answer:-")
        print(answer)

main()