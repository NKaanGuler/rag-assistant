"""Non-interactive evaluation script for the RAG assistant (Phase 3 evidence).

This script asks the assistant a fixed list of questions and prints a
readable transcript:

  * ANSWERABLE questions  -> the answer should come from our documents.
  * UNANSWERABLE questions -> the assistant should say it does not have
    that information (because the docs do not cover these topics).

There are no pass/fail checks on purpose. A human reads the transcript
and judges whether the assistant behaved correctly. Save the output as
your Phase-3 test evidence, for example:

    python test_questions.py > test_transcript.txt
"""

import os
import time

import ingest
import rag

# Questions the knowledge base SHOULD be able to answer.
ANSWERABLE_QUESTIONS = [
    "How does DHCP assign an IP address to a new host?",
    "What is the difference between UDP and TCP?",
    "What is CIDR and how does longest prefix match work?",
    "How does DNS resolve a hostname to an IP address?",
    "How does CSMA/CD handle collisions in Ethernet?",
]

# Questions clearly OUTSIDE the knowledge base. The assistant should
# admit it does not have this information instead of making things up.
UNANSWERABLE_QUESTIONS = [
    "What is the capital of Australia?",
    "Who won the 2026 FIFA World Cup?",
    "Can you give me a recipe for chocolate cake?",
]

# Edge cases: unusual inputs a real user might type. Each entry is a
# (label, question) pair so the transcript explains what is being tested.
# Empty input gets special handling below, because embedding models reject
# empty strings (and the interactive app treats empty input as "exit").
EDGE_CASE_QUESTIONS = [
    ("Empty input", ""),
    ("Whitespace-only input", "   "),
    (
        "Very long, rambling question",
        "I was reading my networking notes last night and got confused "
        "because there are so many layers and protocols involved, so what I "
        "really want to understand, step by step and in simple terms, is "
        "everything that happens in the network from the moment I type a web "
        "address into my browser until the page appears on my screen?",
    ),
    ("Question in Turkish", "TCP ile UDP arasındaki fark nedir?"),
    (
        "Adjacent but uncovered topic (routing algorithm)",
        "How does Dijkstra's algorithm compute shortest paths in link-state routing?",
    ),
    (
        "Adjacent but uncovered topic (traceroute)",
        "How does traceroute discover the routers along a path?",
    ),
]


def ask_question(embedding_client, chat_client, question):
    """Retrieve chunks for one question and print sources plus the answer.

    We use the exact same retrieval (get_top_chunks) and prompt building
    (build_messages) as the interactive app, but with a NON-streaming
    chat call so the transcript is produced in one piece.
    """
    print("Question: " + question)

    # Step 1: retrieve the 3 most relevant chunks from the database.
    chunks = rag.get_top_chunks(embedding_client, question, k=3)

    # No chunk above the similarity threshold -> the documents don't cover
    # this question. Answer deterministically, without calling the model
    # (this mirrors what main.py does).
    if not chunks:
        print("Retrieved sources: (none above the similarity threshold)")
        print("Answer:")
        print("I don't have that information. "
              "(No relevant passages found in the documents.)")
        print("-" * 60)
        return

    print("Retrieved sources:")
    for chunk in chunks:
        # Two decimals is enough to compare relevance at a glance.
        print("  - {} (score: {:.2f})".format(chunk["source"], chunk["score"]))

    # Step 2: build the same messages the interactive app would use,
    # but call complete_chat (non-streaming) to get the full answer at once.
    messages = rag.build_messages(question, chunks)
    start = time.perf_counter()
    try:
        result = chat_client.complete_chat(messages)
    except Exception as error:
        # A rare runtime hiccup (e.g. "Operation was cancelled") should not
        # abort a 15-minute test run: retry once, then report it honestly.
        print(f"(model call failed once: {error} - retrying)")
        try:
            result = chat_client.complete_chat(messages)
        except Exception as error_again:
            print("Answer:")
            print(f"ERROR: model call failed twice: {error_again}")
            print("-" * 60)
            return
    elapsed = time.perf_counter() - start
    answer = result.choices[0].message.content

    print("Answer:")
    print(answer)
    # The course plan asks us to check response times, so record each one.
    print("Time: {:.1f} s".format(elapsed))
    print("-" * 60)


def main():
    """Run the whole evaluation: setup, questions, cleanup."""
    # Make sure the knowledge base exists before we try to query it.
    if not os.path.exists(rag.DB_PATH):
        print("Database not found. Running ingestion first...")
        ingest.ingest()

    # Set up the Foundry Local manager and load both models.
    rag.get_manager()
    embedding_model = rag.load_model(rag.EMBEDDING_MODEL)
    chat_model = rag.load_model(rag.CHAT_MODEL)

    embedding_client = embedding_model.get_embedding_client()
    chat_client = chat_model.get_chat_client()

    # Temperature 0.0 makes answers as deterministic as possible,
    # which is what we want for a repeatable test transcript.
    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 500

    print("=" * 60)
    print("SECTION 1: ANSWERABLE QUESTIONS")
    print("(The assistant should answer these from the documents.)")
    print("=" * 60)
    for question in ANSWERABLE_QUESTIONS:
        ask_question(embedding_client, chat_client, question)

    print("=" * 60)
    print("SECTION 2: UNANSWERABLE QUESTIONS")
    print("(The assistant should say it does not have this information.)")
    print("=" * 60)
    for question in UNANSWERABLE_QUESTIONS:
        ask_question(embedding_client, chat_client, question)

    print("=" * 60)
    print("SECTION 3: EDGE CASES")
    print("(Unusual inputs - the system should stay graceful, not crash.)")
    print("=" * 60)
    for label, question in EDGE_CASE_QUESTIONS:
        print(f"[{label}]")
        if not question.strip():
            # Do not send empty text to the models: the embedding client
            # rejects it, and the interactive app treats it as "exit".
            print("Question: " + repr(question))
            print(
                "Behavior: skipped without calling the models. The interactive "
                "app treats empty input as 'exit', so the pipeline never runs."
            )
            print("-" * 60)
            continue
        ask_question(embedding_client, chat_client, question)

    # Free the memory used by the models now that we are done.
    embedding_model.unload()
    chat_model.unload()
    print("Evaluation finished. Both models unloaded.")


if __name__ == "__main__":
    main()
