"""Entry point for the local RAG assistant.

This script ties everything together:
  1. Makes sure the document database (rag.db) exists, ingesting docs if needed.
  2. Loads the embedding model (to search) and the chat model (to answer).
  3. Runs a simple question/answer loop in the terminal.

Everything runs locally through Foundry Local -- no cloud API keys needed.
"""

import os
import time

from rag import (
    DB_PATH,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    get_manager,
    load_model,
    get_top_chunks,
    stream_answer,
)
import ingest


BANNER = """
==============================================================
  Local RAG Assistant
==============================================================
This assistant answers questions using ONLY the documents in
the 'docs' folder. It runs fully on your own machine with
Foundry Local -- nothing is sent to the cloud.

Note: on the very first run the models are downloaded, which
can take a few minutes. After that, startup is fast.

Try asking things like:
  - How does DHCP assign an IP address to a new host?
  - What is the difference between UDP and TCP?
  - How does CSMA/CD handle collisions in Ethernet?
  - How does DNS resolve a hostname to an IP address?
  - What is CIDR and how does longest prefix match work?

Type 'quit' (or just press Enter) to exit.
==============================================================
"""


def print_sources(chunks):
    """Print the unique source document names, in retrieval order."""
    # We keep the first occurrence of each source so the order matches
    # how relevant the documents were (most relevant first).
    seen = []
    for chunk in chunks:
        if chunk["source"] not in seen:
            seen.append(chunk["source"])
    if not seen:
        print("Sources: (none - the database returned no chunks)")
        return
    print("Sources: " + ", ".join(seen))


def main():
    print(BANNER)

    # Start (or reuse) the Foundry Local manager singleton.
    get_manager()

    # First run? Build the database automatically so the user
    # does not have to remember to run ingest.py by hand.
    if not os.path.exists(DB_PATH):
        print("No database found (rag.db). Ingesting documents first...")
        ingest.ingest()
        print()

    # Load both models. load_model() downloads them if they are not
    # cached yet, then loads them into memory.
    print("Loading embedding model...")
    embedding_model = load_model(EMBEDDING_MODEL)
    embedding_client = embedding_model.get_embedding_client()

    print("Loading chat model...")
    chat_model = load_model(CHAT_MODEL)
    chat_client = chat_model.get_chat_client()
    # Low temperature = more factual, less creative answers.
    chat_client.settings.temperature = 0.2
    chat_client.settings.max_tokens = 500

    print("\nReady! Ask a question about the documents.\n")

    try:
        while True:
            question = input("Question: ").strip()

            # Empty input or a quit word ends the session.
            if question == "" or question.lower() in ("quit", "exit"):
                break

            # Step 1: find the 3 most relevant chunks in the database.
            chunks = get_top_chunks(embedding_client, question, k=3)

            # If nothing scored above the similarity threshold, the documents
            # simply don't cover this question. Say so directly instead of
            # asking the model (which might improvise an ungrounded answer).
            if not chunks:
                print("\nI don't have that information. "
                      "(No relevant passages found in the documents.)\n")
                continue

            # Step 2: stream the answer token by token, so the user
            # sees text appear as the model generates it. We also time it,
            # because the course plan asks us to check response times.
            print()
            start = time.perf_counter()
            answer_parts = []
            try:
                for token in stream_answer(chat_client, question, chunks):
                    print(token, end="", flush=True)
                    answer_parts.append(token)
            except Exception as error:
                # A rare runtime hiccup in the model should not end the session.
                print(f"\n(The model could not finish this answer: {error}. "
                      "Please ask again.)\n")
                continue
            elapsed = time.perf_counter() - start
            print("\n")

            # Step 3: show which documents the answer came from - unless the
            # model refused, in which case listing sources would be misleading.
            answer = "".join(answer_parts).replace("’", "'")
            if "don't have that information" in answer.lower():
                print(f"(no sources cited - answered in {elapsed:.1f} s)")
            else:
                print_sources(chunks)
                print(f"(answered in {elapsed:.1f} s)")
            print()
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C (or the end of piped input) should exit politely,
        # not with a scary traceback.
        print("\nInterrupted.")
    finally:
        # Free the models from memory before exiting.
        print("Unloading models, goodbye!")
        embedding_model.unload()
        chat_model.unload()


if __name__ == "__main__":
    main()
