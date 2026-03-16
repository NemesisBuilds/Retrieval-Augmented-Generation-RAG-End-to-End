import faiss 
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle as pkl
import json
from datetime import datetime
from sklearn.cluster import KMeans

import nltk
nltk.download("stopwords")

start_time = datetime.now()


BASE_DIR = Path(__file__).resolve().parent

DATA_FOLDER_PATH = BASE_DIR / "sample_data"
VECTOR_STORAGE_FOLDER_PATH = BASE_DIR / "vectorStorageFiles"
BM25_FOLDER_STORAGE_PATH = BASE_DIR / "bm25StorageFiles"
#Initializations:-
encoding_model = SentenceTransformer("all-MiniLM-L6-v2")
stemmer = PorterStemmer()
all_files = os.listdir(DATA_FOLDER_PATH)
english_stopwords = set(stopwords.words("english"))
for path in [VECTOR_STORAGE_FOLDER_PATH, BM25_FOLDER_STORAGE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

def extract_text(file_paths: list):
    """
    Reads all text files from the dataset folder and converts them into a
    structured list of dictionaries.

    Each document is loaded into memory and assigned:
    - an id
    - the raw document text
    - a title derived from the file name

    This function represents the first step of the RAG pipeline where
    raw documents are ingested before any processing (chunking, embedding, etc.).

    Parameters
    ----------
    file_paths : list
        List of file names located inside the dataset directory.

    Returns
    -------
    list[dict]
        A list where each entry represents one document:

        {
            "id": int,
            "content": str,
            "title": str
        }
    """
    print("Extracting text from all documents")
    all_text = []
    for i, file in enumerate(file_paths, 1):
        data_title = file.split(".")[0]
        file_text = None
        with open(f"{DATA_FOLDER_PATH / file}", "r", encoding="utf-8") as f:
            file_text = f.read()
        new_entry = {"id":i, "content": file_text, "title": data_title}
        all_text.append(new_entry)
    print("All data extracted from documents.")
    return all_text

def chunk_text(texts: list, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits each document into smaller overlapping chunks suitable for retrieval.

    Large documents are broken into smaller pieces because embedding models
    perform better when encoding shorter passages. Chunk overlap is used to
    preserve context between neighboring chunks.

    This function uses LangChain's RecursiveCharacterTextSplitter with
    token-based splitting.

    Parameters
    ----------
    texts : list
        List of document dictionaries produced by `extract_text`.

    chunk_size : int
        Maximum token length of each chunk.

    chunk_overlap : int
        Number of tokens shared between consecutive chunks.

    Returns
    -------
    list[dict]
        List of dictionaries containing chunked text for each document:

        {
            "id": int,
            "chunks": list[str],
            "title": str
        }
    """
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        encoding_name="cl100k_base"
    )
    all_chunks = []
    for text in texts:
        content = text["content"]
        id = text["id"]
        data_title = text["title"]
        text_chunks = splitter.split_text(content) # Output: list[str]
        new_entry = {"id":id, "chunks": text_chunks, "title": data_title}
        all_chunks.append(new_entry)
    print("Splitting done.")
    return all_chunks

def create_embeddings(chunks_dict):
    """
    Converts text chunks into dense vector embeddings using a
    SentenceTransformer model.

    Each chunk is embedded into a numerical vector representation that
    captures its semantic meaning. These embeddings will later be indexed
    in FAISS to enable fast similarity search.

    This function also saves the chunk text to disk so that retrieved
    vector indices can later be mapped back to the original text.

    Parameters
    ----------
    chunks_dict : list[dict]
        Output from `chunk_text`, containing chunked documents.

    Returns
    -------
    list[dict]
        List containing embeddings for each dataset:

        {
            "id": int,
            "embeddings": np.ndarray,
            "title": str
        }
    """
    print("Creating embeddings...")
    json_files_list = os.listdir(VECTOR_STORAGE_FOLDER_PATH)
    all_embeddings = []
    for chunk in chunks_dict:
        chunks = chunk["chunks"]
        id = chunk["id"]
        data_title = chunk["title"]
        chunk_file_name = data_title + "_all_chunks" + ".json"
        if chunk_file_name not in json_files_list:
            with open(f"{VECTOR_STORAGE_FOLDER_PATH / chunk_file_name}", "w", encoding="utf-8") as f:
                json.dump(chunks, f)
        embeddings = encoding_model.encode(chunks)
        embeddings = np.array(embeddings, dtype="float32")
        embedding_entry = {"id": id, "embeddings": embeddings, "title": data_title}
        all_embeddings.append(embedding_entry)
    print("Embeddings successfully created.")
    return all_embeddings

def create_centroids(embeddings_dict: dict, k: int = 5):
    """
    Creates cluster centroids from document embeddings using K-Means.

    These centroids act as representative vectors for each dataset and
    are later used by the router to determine which dataset is most
    relevant to a user query.

    By clustering embeddings, we approximate the semantic distribution
    of each document collection.

    Parameters
    ----------
    embeddings_dict : list[dict]
        Output from `create_embeddings`.

    k : int
        Number of clusters to generate per dataset.

    Returns
    -------
    tuple
        router_vectors : np.ndarray
            All centroid vectors combined into a single matrix.

        router_metadata : list[dict]
            Metadata linking each centroid to its dataset title.
    """
    router_vectors = []
    router_metadata = []
    for ed in embeddings_dict:
        embeddings = ed["embeddings"]
        title = ed["title"]

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)

        centroids = kmeans.cluster_centers_

        for centroid in centroids:
            centroid = centroid.astype("float32").reshape(1,-1)
            faiss.normalize_L2(centroid)
            router_vectors.append(centroid[0])
            router_metadata.append({
                "title": title
            })
    router_vectors = np.vstack(router_vectors)

    return router_vectors, router_metadata
    
def build_router_index(router_vectors: list, router_metadata: list):
    """
    Builds and stores a FAISS index used for dataset routing.

    The router index contains centroid vectors representing each dataset.
    When a user query arrives, the query embedding is compared against
    these centroids to determine which dataset(s) are most relevant.

    This enables scalable multi-dataset retrieval by avoiding searches
    across the entire corpus.

    Parameters
    ----------
    router_vectors : np.ndarray
        Matrix of centroid embeddings representing datasets.

    router_metadata : list
        Metadata linking each centroid to a dataset title.

    Returns
    -------
    None
        The FAISS router index and metadata are saved to disk.
    """
    router_index_path = BASE_DIR / "router_indices.faiss"
    router_metadata_path = BASE_DIR / "router_metadata.json"
    
    dim = router_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)

    index.add(router_vectors)

    faiss.write_index(index, str(router_index_path))

    with open (router_metadata_path, "w") as f:
        json.dump(router_metadata, f)


def build_faiss_index(embeddings_dict):

    """
    Builds FAISS vector indices for each dataset using chunk embeddings.

    FAISS enables efficient similarity search over dense embeddings.
    Each dataset receives its own FAISS index containing all chunk vectors.

    If an index already exists on disk, it is loaded instead of rebuilt.

    Parameters
    ----------
    embeddings_dict : list[dict]
        Output from `create_embeddings`.

    Returns
    -------
    list[dict]
        List containing FAISS indices:

        {
            "title": str,
            "faiss_index": faiss.Index
        }
    """



    print("Building and storing faiss indices")
    vectorStorageFilesList = os.listdir(VECTOR_STORAGE_FOLDER_PATH)
    faiss_indices_dict = []
    for embeddings in embeddings_dict:
        embedding = embeddings["embeddings"]
        data_title = embeddings["title"]
        file_name = data_title + "_faiss_index" + ".faiss"
        
        
        if file_name not in vectorStorageFilesList:
            faiss.normalize_L2(embedding)
            dim = embedding.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(embedding)
            faiss.write_index(faiss_index, f"{VECTOR_STORAGE_FOLDER_PATH / file_name}")
            new_entry = {"title": data_title, "faiss_index": faiss_index}
            faiss_indices_dict.append(new_entry)

        else:
            faiss_index = faiss.read_index(f"{VECTOR_STORAGE_FOLDER_PATH / file_name}")
            new_entry = {"title": data_title, "faiss_index": faiss_index}
            faiss_indices_dict.append(new_entry)
    print("Faiss complete")
    return faiss_indices_dict


def bm25_preprocessing(text: str):
    """
    Preprocesses text for BM25 lexical search.

    The preprocessing pipeline includes:
    - lowercasing
    - punctuation removal
    - tokenization
    - stopword removal
    - stemming

    This normalization improves BM25 retrieval by reducing vocabulary
    variations across similar words.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    list[str]
        List of processed tokens suitable for BM25 indexing.
    """
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    tokens = text.split()
    tokens = [token for token in tokens if token not in english_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens




def build_bm25_index(entries: list):

    """
    Builds BM25 lexical search indices for each dataset.

    BM25 is a sparse retrieval method that ranks documents based on
    keyword overlap and term frequency statistics.

    Each dataset's chunk corpus is tokenized and used to construct a
    BM25 index. The index and tokenized corpus are saved to disk.

    If an index already exists, it is loaded instead of rebuilt.

    Parameters
    ----------
    entries : list[dict]
        Output from `chunk_text`.

    Returns
    -------
    list[dict]
        List containing BM25 indices:

        {
            "title": str,
            "bm25_index": BM25Okapi
        }
    """

    print("Creating bm25 indices...")
    bm25_storage_files_list = os.listdir(BM25_FOLDER_STORAGE_PATH)
    bm25_index_dicts = []
    for entry in entries:
        chunk = entry["chunks"]  # This is a list of strings: representing the chunks of ONE WHOLE subject.
        id = entry["id"]
        title = entry["title"]

        file_name = title + "_bm25_index" + ".pkl"
        if file_name not in bm25_storage_files_list:

            tokenized_chunk = [bm25_preprocessing(text=ch) for ch in chunk]
            tokenized_corpus_file = title + "_tokenized_corpus" + ".pkl"
            with open (f"{BM25_FOLDER_STORAGE_PATH / tokenized_corpus_file}", "wb") as f:
                pkl.dump(tokenized_chunk, f)
            bm25_index = BM25Okapi(tokenized_chunk)
            with open(f"{BM25_FOLDER_STORAGE_PATH / file_name}", "wb") as f:
                pkl.dump(bm25_index, f)
            new_entry = {"title": title, "bm25_index": bm25_index}
            bm25_index_dicts.append(new_entry)
        else:
            with open(f"{BM25_FOLDER_STORAGE_PATH / file_name}", "rb") as f:
                bm25_index = pkl.load(f)
            new_entry = {"title": title, "bm25_index": bm25_index}
            bm25_index_dicts.append(new_entry)
    print("BM25 done.")
    return bm25_index_dicts

def Build_Both_Indices(data_folder_path: str):
    """
    Complete pipeline for building all retrieval indices.

    This function orchestrates the full preprocessing workflow:
    1. Load documents
    2. Split documents into chunks
    3. Generate embeddings
    4. Build router centroids
    5. Construct FAISS indices
    6. Construct BM25 indices

    It serves as the main initialization step for preparing the
    retrieval infrastructure used by the RAG system.

    Parameters
    ----------
    data_folder_path : str
        Path to the dataset folder containing text documents.

    Returns
    -------
    dict
        Dictionary containing built FAISS and BM25 indices.
    """
    parent_indices_dict = {}
    data_files_list = os.listdir(data_folder_path)
    all_files_data_list = extract_text(file_paths=data_files_list) # list[dict] dict = {id,title,whole_document_string}
    all_files_data_chunks = chunk_text(texts=all_files_data_list)
    all_files_data_embeddings = create_embeddings(chunks_dict=all_files_data_chunks)

    all_centroids, all_centroids_metadata = create_centroids(embeddings_dict=all_files_data_embeddings)

    build_router_index(router_vectors=all_centroids,
                       router_metadata=all_centroids_metadata) 
    """    #Building FAISS indices:- 
    all_files_faiss_indices = build_faiss_index(embeddings_dict=all_files_data_embeddings) #list[dict]
    parent_indices_dict["faiss_indices"] = all_files_faiss_indices
    #Building BM25 indices:-
    all_files_bm25_indices = build_bm25_index(entries=all_files_data_chunks)#list[dict]
    parent_indices_dict["bm25_indices"] = all_files_bm25_indices"""
    return parent_indices_dict

Build_Both_Indices(data_folder_path=DATA_FOLDER_PATH)
end_time = datetime.now()
print(f"Total time taken: {end_time - start_time}")
