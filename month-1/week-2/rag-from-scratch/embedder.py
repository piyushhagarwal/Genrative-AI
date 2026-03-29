import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pricing as of 2024 — text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_COST_PER_MILLION_TOKENS = 0.020  # $0.02 per 1M tokens

# For answer generation
GENERATION_MODEL = "gpt-4o-mini"
INPUT_COST_PER_MILLION_TOKENS = 0.150   # $0.15 per 1M input tokens
OUTPUT_COST_PER_MILLION_TOKENS = 0.600  # $0.60 per 1M output tokens

# Global cost tracker — accumulates across all calls in a session
_cost_tracker = {
    "embedding_tokens": 0,
    "generation_input_tokens": 0,
    "generation_output_tokens": 0,
    "total_cost_usd": 0.0,
}


def count_tokens(text: str, model: str = EMBEDDING_MODEL) -> int:
    """Count tokens without making an API call."""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks and attach the embedding vector to each chunk dict.
    Processes in batches to respect API limits.
    Returns the same chunks with 'embedding' key added.
    """
    BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request

    print(f"\nEmbedding {len(chunks)} chunks using {EMBEDDING_MODEL}...")

    # Count total tokens before sending — so you see the cost upfront
    total_tokens = sum(count_tokens(c["text"]) for c in chunks)
    estimated_cost = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_MILLION_TOKENS
    print(f"Tokens to embed: {total_tokens:,}")
    print(f"Estimated embedding cost: ${estimated_cost:.6f}")

    all_embeddings = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Track actual tokens used (OpenAI reports this in the response)
        actual_tokens = response.usage.total_tokens
        actual_cost = (actual_tokens / 1_000_000) * EMBEDDING_COST_PER_MILLION_TOKENS
        _cost_tracker["embedding_tokens"] += actual_tokens
        _cost_tracker["total_cost_usd"] += actual_cost

        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} chunks embedded")

    # Attach embeddings back to chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Uses the same model as chunks — this is critical.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )

    actual_tokens = response.usage.total_tokens
    actual_cost = (actual_tokens / 1_000_000) * EMBEDDING_COST_PER_MILLION_TOKENS
    _cost_tracker["embedding_tokens"] += actual_tokens
    _cost_tracker["total_cost_usd"] += actual_cost

    return response.data[0].embedding


def track_generation_cost(input_tokens: int, output_tokens: int):
    """Call this after every LLM generation to track cost."""
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION_TOKENS
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION_TOKENS
    _cost_tracker["generation_input_tokens"] += input_tokens
    _cost_tracker["generation_output_tokens"] += output_tokens
    _cost_tracker["total_cost_usd"] += input_cost + output_cost


def print_cost_summary():
    """Print a full cost breakdown at the end of a session."""
    print("\n========== COST SUMMARY ==========")
    print(f"Embedding tokens used : {_cost_tracker['embedding_tokens']:,}")
    print(f"Generation input tokens : {_cost_tracker['generation_input_tokens']:,}")
    print(f"Generation output tokens: {_cost_tracker['generation_output_tokens']:,}")
    print(f"Total cost this session : ${_cost_tracker['total_cost_usd']:.6f}")
    print("==================================")