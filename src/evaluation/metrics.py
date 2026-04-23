import anthropic


def precision_at_k(retrieved_chunks: list[dict], relevant_chunk_texts: list[str]) -> float:
    if not retrieved_chunks:
        return 0.0
    hits = sum(
        1
        for chunk in retrieved_chunks
        if any(rel.lower() in chunk["text"].lower() for rel in relevant_chunk_texts)
    )
    return hits / len(retrieved_chunks)


def recall_at_k(retrieved_chunks: list[dict], relevant_chunk_texts: list[str]) -> float:
    if not relevant_chunk_texts:
        return 0.0
    hits = sum(
        1
        for rel in relevant_chunk_texts
        if any(rel.lower() in chunk["text"].lower() for chunk in retrieved_chunks)
    )
    return hits / len(relevant_chunk_texts)


def faithfulness_score(
    answer: str,
    context_chunks: list[dict],
    client: anthropic.Anthropic,
    model: str,
) -> float:
    context_text = "\n".join(c["text"] for c in context_chunks)
    prompt = (
        "You are an evaluation judge. Rate how faithfully the following ANSWER is grounded "
        "in the CONTEXT. Score 1-5: 1=completely unsupported, 3=partially supported, "
        "5=fully grounded in context with no hallucination. Return only the integer score.\n\n"
        f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}\n\nScore:"
    )
    response = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 0.0


def answer_quality_score(
    answer: str,
    ground_truth: str,
    client: anthropic.Anthropic,
    model: str,
) -> float:
    prompt = (
        "You are an evaluation judge. Rate the ANSWER compared to the GROUND TRUTH. "
        "Score 1-5: 1=completely wrong/irrelevant, 3=partially correct, 5=correct and complete. "
        "Return only the integer score.\n\n"
        f"GROUND TRUTH:\n{ground_truth}\n\nANSWER:\n{answer}\n\nScore:"
    )
    response = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 0.0
