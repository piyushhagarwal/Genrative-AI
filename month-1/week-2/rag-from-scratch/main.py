import os
import sys
import argparse
from dotenv import load_dotenv
from loader import load_and_validate
from chunker import chunk_document
from embedder import embed_chunks, embed_query, print_cost_summary
from store import (
    get_chroma_client,
    get_or_create_collection,
    store_chunks,
    retrieve_chunks,
    collection_exists,
    delete_collection,
)
from qa import generate_answer, print_answer

load_dotenv()


def build_collection_name(pdf_path: str) -> str:
    """
    Generate a stable collection name from the PDF filename.
    Each PDF gets its own collection — so different documents
    don't overwrite each other in Chroma.
    """
    from pathlib import Path
    # Chroma collection names must be alphanumeric + underscores
    stem = Path(pdf_path).stem
    safe = "".join(c if c.isalnum() else "_" for c in stem).lower()
    return f"pdf_{safe}"


def ingest(pdf_path: str, strategy: str, chunk_size: int, force: bool) -> tuple:
    """
    Load, chunk, embed and store a PDF.
    Returns (collection, client) ready for querying.
    
    force=True will delete the existing collection and re-embed.
    Useful when you change chunking strategy and want a fresh index.
    """
    pages = load_and_validate(pdf_path)
    client = get_chroma_client()
    collection_name = build_collection_name(pdf_path)

    if force:
        print(f"Force re-index requested — deleting existing collection '{collection_name}'")
        delete_collection(client, collection_name)

    if collection_exists(client, collection_name):
        print(f"Collection '{collection_name}' already exists — skipping embedding.")
        print("Tip: use --force to re-index with a different chunking strategy.")
        collection = get_or_create_collection(client, collection_name)
    else:
        print(f"\nChunking strategy : {strategy}")
        print(f"Chunk size        : {chunk_size} chars")
        chunks = chunk_document(pages, strategy=strategy, max_chunk_size=chunk_size)
        print(f"Total chunks      : {len(chunks)}")

        embedded_chunks = embed_chunks(chunks)
        collection = get_or_create_collection(client, collection_name)
        store_chunks(collection, embedded_chunks)

    return collection


def interactive_loop(collection, top_k: int) -> None:
    """
    Keep asking questions until the user types 'exit'.
    This is the core user experience — no restarting for each question.
    """
    print("\n" + "="*50)
    print("PDF QA System ready. Type your question.")
    print("Commands: 'exit' to quit, 'cost' to see cost summary")
    print("="*50)

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C gracefully
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() == "exit":
            break

        if query.lower() == "cost":
            print_cost_summary()
            continue

        # Embed query + retrieve + generate
        query_embedding = embed_query(query)
        retrieved = retrieve_chunks(collection, query_embedding, top_k=top_k)
        result = generate_answer(query, retrieved)
        print_answer(result)


def main():
    parser = argparse.ArgumentParser(
        description="PDF Question Answering System — no frameworks, pure RAG"
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "recursive"],
        default="recursive",
        help="Chunking strategy (default: recursive)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Max chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if collection already exists"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Ask a single question and exit (non-interactive mode)"
    )

    args = parser.parse_args()

    # Ingest the PDF
    collection = ingest(
        pdf_path=args.pdf_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        force=args.force,
    )

    # Either answer one question and exit, or enter interactive loop
    if args.question:
        query_embedding = embed_query(args.question)
        retrieved = retrieve_chunks(collection, query_embedding, top_k=args.top_k)
        result = generate_answer(query, retrieved)
        print_answer(result)
        print_cost_summary()
    else:
        interactive_loop(collection, top_k=args.top_k)
        print_cost_summary()


if __name__ == "__main__":
    main()