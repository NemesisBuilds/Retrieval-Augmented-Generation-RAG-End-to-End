import faiss 
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle as pkl
import json
from datetime import datetime
from build_indices import bm25_preprocessing, Build_Both_Indices, DATA_FOLDER_PATH
from groq import Groq



BASE_DIR = Path(__file__).resolve().parent

DATA_FOLDER_PATH = BASE_DIR / "sample_data"
VECTOR_STORAGE_FOLDER_PATH = BASE_DIR / "vectorStorageFiles"
BM25_FOLDER_STORAGE_PATH = BASE_DIR / "bm25StorageFiles"
#Initializations:-
encoding_model = SentenceTransformer("all-MiniLM-L6-v2")
stemmer = PorterStemmer()
all_files = os.listdir(DATA_FOLDER_PATH)
english_stopwords = set(stopwords.words("english"))
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker = CrossEncoder(model_name)
groq_llm = Groq(api_key="YOUR_API_KEY")
for path in [VECTOR_STORAGE_FOLDER_PATH, BM25_FOLDER_STORAGE_PATH]:
    path.mkdir(parents=True, exist_ok=True)



def LLM(prompt: str, max_tokens: int = 256):
    """
    Sends a prompt to the configured LLM and returns the generated response.

    This function acts as a lightweight wrapper around the Groq API
    to simplify LLM calls within the RAG pipeline.

    Parameters
    ----------
    prompt : str
        Input prompt containing context and the user's question.

    max_tokens : int
        Maximum number of tokens the model is allowed to generate.

    Returns
    -------
    str
        The model's generated response.
    """
    response = groq_llm.chat.completions.create(
        messages=[{"role":"user",
                   "content": prompt}],
        model="moonshotai/kimi-k2-instruct-0905",
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def RouterFunction(user_query: str, ParentIndicesDict: dict): 

    """
    Determines which datasets are most relevant to the user query.

    The query is embedded and compared against dataset centroid vectors
    stored in the router FAISS index. The top matching datasets are
    selected for further retrieval.

    This step prevents unnecessary searches across unrelated datasets,
    improving efficiency and scalability.

    Parameters
    ----------
    user_query : str
        The user's search query.

    ParentIndicesDict : dict
        Dictionary containing all FAISS and BM25 indices.

    Returns
    -------
    dict
        A structure containing only the selected indices and chunks
        relevant to the query.
    """

    print("Starting routing") 
    #Storing the Indices seperately in different variables:- 
    faiss_indices_list = ParentIndicesDict["faiss_indices"] # list of dicts having the keys as ["title","faiss_index"] 
    faiss_indices_titles = [a["title"] for a in faiss_indices_list] 
    bm25_indices_list = ParentIndicesDict["bm25_indices"] # keys["title","bm25_index"] 
    bm25_indices_titles = [b["title"] for b in bm25_indices_list] 
    faiss_indices_titles.extend(bm25_indices_titles) 
    all_titles = set(faiss_indices_titles) 
    # a set of all the titles of the indices:- which ultimately represent the whole corpus. 
    user_query_vector = encoding_model.encode([user_query]) 
    user_query_vector = np.array(user_query_vector, dtype="float32")
    faiss.normalize_L2(user_query_vector) 
    
    router_index = faiss.read_index(str(BASE_DIR / "router_indices.faiss"))

    router_index_scores, router_index_indices = router_index.search(user_query_vector, 3)

    with open(str(BASE_DIR / "router_metadata.json"), "r") as x:
        router_metadata = json.load(x)
    
    router_index_titles = []
    for rii in router_index_indices[0]:
        router_index_title = router_metadata[rii]["title"]
        router_index_titles.append(router_index_title)
    router_index_titles = set(router_index_titles)

    print("Selected datasets from the router.")
    print(router_index_titles)

    all_data = []
    all_faiss_indices = {}
    all_bm25_indices = {}
    all_chunks = {}
    # Selecting the relevant FAISS Indices
    for faiss_index in faiss_indices_list:
        if faiss_index["title"] in router_index_titles:
            all_faiss_indices[faiss_index["title"]] = faiss_index["faiss_index"]
    # Selecting the relevant BM25 Indices 
    for bm25_index in bm25_indices_list:
        if bm25_index["title"] in router_index_titles:
            all_bm25_indices[bm25_index["title"]] = bm25_index["bm25_index"]
    
    for st in router_index_titles:
        chunks_file_name = st + "_all_chunks" +".json"
        with open(str(VECTOR_STORAGE_FOLDER_PATH / chunks_file_name), "r") as chunk_file:
            all_chunks[st] = json.load(chunk_file)
    return {
        "all_faiss_indices": all_faiss_indices,
        "all_bm25_indices": all_bm25_indices,
        "all_chunks": all_chunks,
        "user_query": user_query
    }





def HybridSearchFunction(query_and_indices: dict, top_n: int = 30, rrf_k: int = 60):

    """
    Performs hybrid retrieval using both dense (FAISS) and sparse (BM25)
    search techniques.

    The function retrieves candidate chunks from both retrieval systems
    and combines their rankings using Retrieval Rank Fusion (RRF).

    RRF assigns scores based on ranking positions rather than raw
    similarity values, making it robust across different retrieval models.

    Parameters
    ----------
    query_and_indices : dict
        Output from RouterFunction containing selected indices and query.

    top_n : int
        Number of candidate results retrieved from each retriever.

    rrf_k : int
        Constant used in the RRF scoring formula.

    Returns
    -------
    dict
        Dictionary containing:
        - user_query
        - selected candidate chunks
    """

    print("Starting hybrid search")

    user_query = query_and_indices["user_query"]
    all_faiss_indices = query_and_indices["all_faiss_indices"]
    all_bm25_indices = query_and_indices["all_bm25_indices"]
    all_chunks = query_and_indices["all_chunks"]

    user_query_vector = encoding_model.encode([user_query])
    user_query_vector = np.array(user_query_vector, dtype="float32")
    faiss.normalize_L2(user_query_vector)

    fused_scores = {}

    # -------- FAISS SEARCH --------
    for title, faiss_index in all_faiss_indices.items():

        scores, indices = faiss_index.search(user_query_vector, top_n)

        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            key = (title, idx)
            rrf_score = 1 / (rrf_k + rank)

            fused_scores[key] = fused_scores.get(key, 0) + rrf_score


    # -------- BM25 SEARCH --------
    tokenized_query = bm25_preprocessing(text=user_query)

    for title, bm25_index in all_bm25_indices.items():

        bm25_scores = bm25_index.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(bm25_scores)),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        for rank, (idx, _) in enumerate(ranked):
            key = (title, idx)
            rrf_score = 1 / (rrf_k + rank)

            fused_scores[key] = fused_scores.get(key, 0) + rrf_score


    # -------- FINAL SORT --------
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    selected_chunks = []

    for (title, idx), _ in sorted_results:
        chunk_text = all_chunks[title][idx]
        selected_chunks.append(chunk_text)

    print("Hybrid search complete")

    return {
        "user_query": user_query,
        "selected": selected_chunks
    }

def RerankerFunction(query_and_candidates: dict, top_k: int = 4):

    """
    Improves retrieval quality by reranking candidate chunks using
    a CrossEncoder model.

    Unlike bi-encoder embeddings, the cross-encoder jointly processes
    the query and document, allowing for more accurate relevance scoring.

    The top scoring chunks are returned as the final retrieval results.

    Parameters
    ----------
    query_and_candidates : dict
        Output from HybridSearchFunction containing candidate chunks.

    top_k : int
        Number of top chunks to keep after reranking.

    Returns
    -------
    dict
        Dictionary containing the final ranked chunks and query.
    """

    print("Reranking...")

    user_query = query_and_candidates["user_query"]
    candidates = query_and_candidates["selected"]

    pairs = [[user_query, doc] for doc in candidates]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "retrieved_chunks": [(doc, score) for doc, score in ranked[:top_k]],
        "user_query": user_query
    }



def Load_Indices():
    """
    Loads previously built FAISS and BM25 indices from disk.

    If indices are not found, the function automatically triggers
    the full index building pipeline.

    This ensures the retrieval system is always initialized with
    the required indices.

    Returns
    -------
    dict
        Dictionary containing loaded FAISS and BM25 indices.
    """

    faiss_indices = []
    bm25_indices = []

    for file in os.listdir(VECTOR_STORAGE_FOLDER_PATH):
        if file.endswith("_faiss_index.faiss"):
            title = file.replace("_faiss_index.faiss","")
            index = faiss.read_index(str(VECTOR_STORAGE_FOLDER_PATH / file))
            faiss_indices.append({"title": title, "faiss_index": index})

    for file in os.listdir(BM25_FOLDER_STORAGE_PATH):
        if file.endswith("_bm25_index.pkl"):
            title = file.replace("_bm25_index.pkl","")
            with open(BM25_FOLDER_STORAGE_PATH / file,"rb") as f:
                bm25 = pkl.load(f)
            bm25_indices.append({"title": title, "bm25_index": bm25})

    if not faiss_indices or not bm25_indices:
        result = Build_Both_Indices(data_folder_path=DATA_FOLDER_PATH)
        return {
            "faiss_indices": result["faiss_indices"],
            "bm25_indices": result["bm25_indices"]
        }

    return {
        "faiss_indices": faiss_indices,
        "bm25_indices": bm25_indices
    }

#Loading indices once only...
INDICES = Load_Indices()

def Retrieve(user_query: str):
    """
    Executes the complete retrieval pipeline for a user query.

    The pipeline consists of:
    1. Dataset routing
    2. Hybrid retrieval (FAISS + BM25)
    3. Cross-encoder reranking

    The final output contains the most relevant chunks for answering
    the user's question.

    Parameters
    ----------
    user_query : str
        The user's search query.

    Returns
    -------
    dict
        Dictionary containing the reranked retrieval results.
    """

    router_result = RouterFunction(
        user_query=user_query,
        ParentIndicesDict=INDICES
    )

    hybrid_result = HybridSearchFunction(
        query_and_indices=router_result
    )

    reranked = RerankerFunction(
        query_and_candidates=hybrid_result
    )

    return reranked

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

def TriggerRagFunction(user_query: str):
    """
    Executes the full Retrieval-Augmented Generation (RAG) workflow.

    Steps performed:
    1. Retrieve relevant chunks using the retrieval pipeline.
    2. Construct a context block from retrieved documents.
    3. Insert the context and query into a prompt template.
    4. Send the prompt to the LLM to generate the final answer.

    The LLM is instructed to answer strictly based on the provided context.

    Parameters
    ----------
    user_query : str
        The user's question.

    Returns
    -------
    str
        The final answer generated by the LLM.
    """
    answer = Retrieve(user_query = user_query)
    retrieved_chunks = [doc for (doc, score) in answer["retrieved_chunks"] ]
    
    context = """ """
    for i, docs in enumerate(retrieved_chunks, 1):
        entry = f"{i}. \n {docs}"
        context += entry
    master_prompt = prompt.format(
        CONTEXT = context,
        USER_QUESTION = user_query
    )
    print("Context:-\n ", context)
    response = LLM(prompt=master_prompt)
    
    return response


start_time = datetime.now()
user_input = input("User:- ")

answer = TriggerRagFunction(user_query=user_input)
print(f"Answer:- {answer}")
end_time = datetime.now()
print("Total time:", end_time - start_time)
