"""
Chroma-backed vector store of known recalled medicine batches.
Seeded once from data/recalled_batches.json. Used as the first lookup layer
before falling back to a live web search via the LLM.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from decouple import config

CHROMA_DIR = config("CHROMA_DIR", default="./chroma_db")
DATA_FILE = Path(__file__).parent / "data" / "recalled_batches.json"
COLLECTION_NAME = "recalled_batches"
MATCH_THRESHOLD = config("VECTOR_MATCH_THRESHOLD", default=0.35, cast=float)

_client = None
_collection = None
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _doc_text(entry: dict) -> str:
    return (
        f"{entry['medicine_name']} ({', '.join(entry['aliases'])}) - "
        f"Batch: {entry['batch_number']}, Manufacturer: {entry['manufacturer']}. "
        f"Recall Reason: {entry['recall_reason']}"
    )


def init_vector_store(force_reseed: bool = False):
    """Creates the collection and seeds it from recalled_batches.json if empty."""
    global _collection
    client = _get_client()
    
    if force_reseed:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embedder
    )

    if _collection.count() == 0:
        with open(DATA_FILE, "r") as f:
            entries = json.load(f)

        _collection.add(
            ids=[str(i) for i in range(len(entries))],
            documents=[_doc_text(e) for e in entries],
            metadatas=[
                {
                    "medicine_name": e["medicine_name"],
                    "batch_number": e["batch_number"],
                    "aliases": json.dumps(e["aliases"]),
                    "manufacturer": e["manufacturer"],
                    "recall_date": e["recall_date"],
                    "recall_reason": e["recall_reason"],
                    "recalling_agency": e["recalling_agency"],
                    "recall_class": e["recall_class"],
                    "recommendation": e["recommendation"],
                }
                for e in entries
            ],
        )
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        init_vector_store()
    return _collection


def search_recalled_db(batch_number: str, medicine_name: str = "", top_k: int = 3) -> list[dict]:
    """
    Searches the recalled batches database.
    First tries direct exact match (case-insensitive, whitespace stripped) on batch_number.
    If no exact match, falls back to semantic vector search.
    """
    collection = get_collection()
    
    # 1. Try exact batch matching first (if batch_number is provided)
    clean_batch = (batch_number or "").strip().upper()
    if clean_batch:
        try:
            get_results = collection.get(where={"batch_number": clean_batch})
            if get_results and get_results["ids"]:
                matches = []
                for i in range(len(get_results["ids"])):
                    meta = get_results["metadatas"][i]
                    matches.append({
                        "medicine_name": meta["medicine_name"],
                        "batch_number": meta["batch_number"],
                        "aliases": json.loads(meta["aliases"]),
                        "manufacturer": meta["manufacturer"],
                        "recall_date": meta["recall_date"],
                        "recall_reason": meta["recall_reason"],
                        "recalling_agency": meta["recalling_agency"],
                        "recall_class": meta["recall_class"],
                        "recommendation": meta["recommendation"],
                        "distance": 0.0  # Exact match
                    })
                return matches
        except Exception as e:
            print(f"[vector_store] Error in exact batch match: {e}")

    # 2. Semantic query fallback
    query_parts = []
    if medicine_name:
        query_parts.append(medicine_name)
    if batch_number:
        query_parts.append(f"batch {batch_number}")
    query_text = " ".join(query_parts).strip()
    
    if not query_text:
        return []

    results = collection.query(query_texts=[query_text], n_results=top_k)
    matches = []
    if not results["ids"] or not results["ids"][0]:
        return matches

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]  # cosine distance, lower = closer
        if distance <= MATCH_THRESHOLD:
            meta = results["metadatas"][0][i]
            matches.append({
                "medicine_name": meta["medicine_name"],
                "batch_number": meta["batch_number"],
                "aliases": json.loads(meta["aliases"]),
                "manufacturer": meta["manufacturer"],
                "recall_date": meta["recall_date"],
                "recall_reason": meta["recall_reason"],
                "recalling_agency": meta["recalling_agency"],
                "recall_class": meta["recall_class"],
                "recommendation": meta["recommendation"],
                "distance": distance,
            })
    return matches
