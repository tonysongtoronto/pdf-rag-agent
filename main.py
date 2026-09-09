"""
Interactive CLI for the PDF RAG + Agent backend.

Usage:
    python main.py
"""
from __future__ import annotations

from src.agent.graph import Agent


def main() -> None:
    print("PDF RAG Agent\n")

    try:
        agent = Agent()
    except Exception as exc:  # noqa: BLE001 - e.g. missing FAISS index
        print(f"Error: {exc}")
        return

    while True:
        try:
            question = input("Ask a question:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            response = agent.ask(question)
        except Exception as exc:  # noqa: BLE001 - surface any error to the CLI user
            print(f"\nError: {exc}\n")
            continue

        print(f"\nAnswer:\n{response['answer']}\n")
        if response["sources"]:
            print("Sources:")
            for source in response["sources"]:
                print(f"- {source['source']}, page {source['page']}")
            print()


if __name__ == "__main__":
    main()
