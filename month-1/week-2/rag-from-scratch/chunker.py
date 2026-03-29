def chunk_fixed_size(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Strategy 1: Fixed-size chunking with overlap.
    
    Splits text by character count, with overlap between consecutive chunks
    so that sentences cut at a boundary aren't lost entirely.
    
    Returns a list of chunks, each with metadata.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "page_number": page["page_number"],
                    "strategy": "fixed",
                    "char_start": start,
                })
                chunk_id += 1

            start += chunk_size - overlap

    return chunks


def chunk_sentence(pages: list[dict], max_chunk_size: int = 500, overlap_sentences: int = 1) -> list[dict]:
    """
    Strategy 2: Sentence-boundary chunking.
    
    Respects sentence endings so chunks don't cut mid-thought.
    Groups sentences until max_chunk_size is reached, then starts
    a new chunk — with the last N sentences carried over as overlap.
    """
    import re

    chunks = []
    chunk_id = 0

    sentence_endings = re.compile(r'(?<=[.!?])\s+')

    for page in pages:
        text = page["text"]
        if not text:
            continue

        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk_sentences = []
        current_length = 0

        for sentence in sentences:
            current_chunk_sentences.append(sentence)
            current_length += len(sentence)

            if current_length >= max_chunk_size:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "page_number": page["page_number"],
                    "strategy": "sentence",
                    "sentence_count": len(current_chunk_sentences),
                })
                chunk_id += 1

                # Carry over last N sentences as overlap
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk_sentences)

        # Don't forget the last partial chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "page_number": page["page_number"],
                "strategy": "sentence",
                "sentence_count": len(current_chunk_sentences),
            })
            chunk_id += 1

    return chunks

def chunk_recursive(pages: list[dict], max_chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Strategy 3: Recursive chunking.
    
    Tries to split on natural boundaries from largest to smallest.
    Only falls back to a smaller separator if the chunk is still too big.
    This preserves semantic meaning better than fixed-size alone.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    chunk_id = 0

    def split_text(text: str, sep_index: int = 0) -> list[str]:
        """Recursively split text using fallback separators."""
        if len(text) <= max_chunk_size:
            return [text]

        if sep_index >= len(separators):
            # Last resort: hard character split
            return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size - overlap)]

        separator = separators[sep_index]
        splits = text.split(separator) if separator else list(text)

        result = []
        current = ""

        for split in splits:
            candidate = current + separator + split if current else split

            if len(candidate) <= max_chunk_size:
                current = candidate
            else:
                if current:
                    # current fits — but maybe it needs further splitting
                    result.extend(split_text(current, sep_index + 1))
                current = split

        if current:
            result.extend(split_text(current, sep_index + 1))

        return result

    for page in pages:
        text = page["text"]
        if not text:
            continue

        raw_chunks = split_text(text)

        for raw in raw_chunks:
            if raw.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": raw.strip(),
                    "page_number": page["page_number"],
                    "strategy": "recursive",
                })
                chunk_id += 1

    return chunks

def chunk_document(pages: list[dict], strategy: str = "fixed", **kwargs) -> list[dict]:
    if strategy == "fixed":
        return chunk_fixed_size(pages, **kwargs)
    elif strategy == "sentence":
        return chunk_sentence(pages, **kwargs)
    elif strategy == "recursive":
        return chunk_recursive(pages, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'fixed', 'sentence', or 'recursive'.")