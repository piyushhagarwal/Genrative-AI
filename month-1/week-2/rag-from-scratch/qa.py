import os
from openai import OpenAI
from embedder import embed_query, track_generation_cost, GENERATION_MODEL

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the prompt that goes to the LLM.
    
    The structure matters:
    - Tell the model exactly what its job is
    - Give it the context (retrieved chunks) with source labels
    - Ask the question
    - Tell it what to do if the answer isn't in the chunks
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        block = (
            f"[Source {i+1} | Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a precise question-answering assistant.
You will be given excerpts from a document and a question.
Your job is to answer the question using ONLY the provided excerpts.
If the answer is not contained in the excerpts, say exactly:
"I could not find the answer in the provided document."
Do not use any outside knowledge. Do not speculate.

After your answer, always list which sources you used (e.g. Source 1, Source 3).

DOCUMENT EXCERPTS:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Send the prompt to the LLM and return the answer with sources.
    Returns a dict with answer text and source references.
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant chunks were retrieved. Cannot answer.",
            "sources": [],
            "query": query,
        }

    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # We want deterministic answers, not creative ones
    )

    answer_text = response.choices[0].message.content.strip()

    # Track cost
    track_generation_cost(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )

    # Build clean source references
    sources = [
        {
            "source_number": i + 1,
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "similarity_score": chunk["similarity_score"],
            "text_preview": chunk["text"][:150],
        }
        for i, chunk in enumerate(retrieved_chunks)
    ]

    return {
        "query": query,
        "answer": answer_text,
        "sources": sources,
    }


def print_answer(result: dict) -> None:
    """Pretty print the answer and its sources."""
    print("\n" + "="*50)
    print(f"QUESTION: {result['query']}")
    print("="*50)
    print(f"\nANSWER:\n{result['answer']}")
    print("\n--- SOURCES USED ---")
    for s in result["sources"]:
        print(f"\nSource {s['source_number']} | Page {s['page_number']} | Score: {s['similarity_score']:.4f}")
        print(f"Preview: {s['text_preview']}...")