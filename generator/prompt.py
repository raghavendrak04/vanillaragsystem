"""
Prompt Templates — RAG prompts that enforce grounded answers with citations.

The prompt design is critical for ensuring:
1. Answers are ONLY based on provided context
2. Citations are included in the response
3. Unanswerable questions are detected and flagged
"""


SYSTEM_PROMPT = """You are an expert assistant specialized in Indian Defence procurement policy, \
financial delegations, and naval regulations. You answer questions ONLY based on the provided \
context documents.

CRITICAL RULES:
1. Answer ONLY using information from the provided context. Never use external knowledge.
2. ALWAYS cite your sources using the format: [Source: DocumentName, Page X]
3. If the context does NOT contain enough information to answer the question, respond with:
   "I cannot answer this question based on the available documents. The provided context does not contain sufficient information about [topic]."
4. If the answer requires combining information from multiple chunks, cite all relevant sources.
5. Be precise and specific. Quote relevant regulation numbers, section numbers, and exact text where helpful.
6. Structure your answer clearly with bullet points or numbered lists when listing multiple items.
7. Do NOT make assumptions or inferences beyond what the context explicitly states.
"""


QUERY_PROMPT_TEMPLATE = """Based on the following context documents, answer the user's question.

{context}

---

Question: {question}

Provide a clear, accurate answer grounded in the context above. Include citations for every claim.
If the context doesn't contain the answer, say so explicitly — do not guess.

Answer:"""


def build_rag_prompt(question: str, context: str) -> list[dict]:
    """
    Build the complete prompt messages for the RAG pipeline.

    Args:
        question: User's natural language question.
        context: Formatted context string from retrieved chunks.

    Returns:
        List of message dicts for the LLM API (system + user messages).
    """
    user_message = QUERY_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


EVALUATION_PROMPT_TEMPLATE = """You are evaluating the quality of a RAG system's response.

Question: {question}
Expected Answer: {expected_answer}
System's Answer: {system_answer}

Evaluate the system's answer on these criteria (score each 0-5):
1. **Correctness**: Does the answer match the expected answer in meaning?
2. **Completeness**: Does the answer cover all aspects of the expected answer?
3. **Grounding**: Is the answer properly grounded with citations?
4. **Conciseness**: Is the answer appropriately concise without unnecessary information?

Respond in this exact JSON format:
{{
    "correctness": <score>,
    "completeness": <score>,
    "grounding": <score>,
    "conciseness": <score>,
    "reasoning": "<brief explanation>"
}}
"""
