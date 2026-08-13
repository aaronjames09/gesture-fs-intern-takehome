"""
Document Q&A Pipeline — completed implementation.

The knowledge base (loading, chunking, vector store) is already built
in knowledge_base.py. This module is the response layer:
  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
from typing import Callable, Dict, List, Union

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], List[Dict[str, str]]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> List[Dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1 (done): ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(
    vector_store,
    llm: Callable[[str], List[Dict[str, str]]],
    question: str,
) -> Dict[str, Union[str, List[str]]]:
    """Retrieve relevant chunks and generate an answer.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved

    Raises:
        ValueError: if question is empty/whitespace-only.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    # 1. Retrieve the top 3 most relevant chunks.
    docs = vector_store.similarity_search(question, k=3)

    # Guard: nothing in the knowledge base matched (empty index, etc).
    if not docs:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    # 2. Pull out the raw text of each retrieved chunk.
    sources: List[str] = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)

    # 3. Plug context + question into the provided template.
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 4. Call the local LLM and extract the generated text.
    result = llm(prompt)
    answer = result[0]["generated_text"].strip()

    return {"answer": answer, "sources": sources}


# ──────────────────────────────────────────────
# Helper: consistent CLI output formatting
# (kept separate so main() stays focused on flow control — SRP)
# ──────────────────────────────────────────────
def _print_result(result: Dict[str, Union[str, List[str]]]) -> None:
    print("\n📄 Sources:")
    for i, src in enumerate(result["sources"], start=1):
        preview = src if len(src) <= 150 else src[:150].rstrip() + "..."
        print(f"  {i}. {preview}")
    print(f"\n💬 Answer: {result['answer']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2 (done): interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop, plus a `--query` single-shot mode (bonus)."""
    parser = argparse.ArgumentParser(
        description="Ask questions about the marketing agency's services, pricing, and process."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # 1. Build the knowledge base and load the LLM.
    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    # Single-question mode (bonus: --query CLI flag)
    if args.query is not None:
        try:
            result = ask_question(vector_store, llm, args.query)
            _print_result(result)
        except ValueError as e:
            print(f"Error: {e}")
        return

    # 2/3. Interactive loop.
    print("Ask a question about our services, pricing, or process. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question, or type 'quit' to exit.\n")
            continue

        try:
            result = ask_question(vector_store, llm, question)
            _print_result(result)
        except Exception as e:  # keep the REPL alive on unexpected errors
            print(f"Sorry, something went wrong answering that: {e}\n")


if __name__ == "__main__":
    main()
