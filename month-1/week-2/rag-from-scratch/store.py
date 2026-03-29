import chromadb
from chromadb.config import Settings


def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.ClientAPI:
    """
    Create a persistent Chroma client.
    Data survives between runs — stored in persist_dir on disk.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def get_or_create_collection(client: chromadb.ClientAPI, collection_name: str):
    """
    Get existing collection or create a new one.
    Collection = a named group of vectors (like a table in a database).
    We use cosine similarity — standard for text embeddings.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def store_chunks(collection, chunks: list[dict]) -> None:
    """
    Insert embedded chunks into Chroma.
    Chroma needs four things per chunk:
      - ids        : unique string identifier
      - embeddings : the vector
      - documents  : the raw text (so we can return it at query time)
      - metadatas  : page number, strategy, etc.
    """
    if not chunks:
        print("No chunks to store.")
        return

    # Check all chunks have embeddings
    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        raise ValueError(f"Chunks missing embeddings: {missing}")

    ids = [str(c["chunk_id"]) for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "page_number": c["page_number"],
            "strategy": c["strategy"],
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]

    # Upsert = insert or update if id already exists
    # This means re-running won't create duplicates
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in collection '{collection.name}'")
    print(f"Total chunks in collection: {collection.count()}")


def retrieve_chunks(collection, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the top_k most similar chunks to the query embedding.
    Returns a clean list of results with text, metadata and similarity score.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Chroma returns nested lists (one per query) — unwrap the first
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page_number": results["metadatas"][0][i]["page_number"],
            "strategy": results["metadatas"][0][i]["strategy"],
            # Chroma returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score (1=identical, -1=opposite)
            "similarity_score": 1 - results["distances"][0][i],
        })

    return retrieved

def collection_exists(client: chromadb.ClientAPI, collection_name: str) -> bool:
    """Check if a collection already has data in it."""
    try:
        collection = client.get_collection(collection_name)
        return collection.count() > 0
    except Exception:
        return False


def delete_collection(client: chromadb.ClientAPI, collection_name: str) -> None:
    """Wipe a collection — useful when re-indexing a document."""
    try:
        client.delete_collection(collection_name)
        print(f"Deleted collection '{collection_name}'")
    except Exception as e:
        print(f"Could not delete collection: {e}")