"""Core RAG helpers: Foundry Local setup, similarity search, and streaming answers.

This module is imported by both ingest.py (to embed documents) and main.py
(to answer questions), so it keeps all the shared pieces in one place.
"""

import json
import math
import os
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "rag.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-3.5-mini"

# Chunks scoring below this cosine-similarity value are treated as
# "nothing relevant found". In our tests, questions the documents can
# answer score 0.5+ (even a Turkish question scored 0.51), while
# unrelated questions score 0.24 or less.
MIN_SCORE = 0.30


def get_manager():
    """Initialize Foundry Local (if needed) and return the manager singleton.

    The SDK raises an error if initialize() is called twice, so we only
    initialize when the singleton does not exist yet. That makes this
    function safe to call as many times as you like.
    """
    if FoundryLocalManager.instance is None:
        config = Configuration(app_name="rag-assistant")
        FoundryLocalManager.initialize(config)
    return FoundryLocalManager.instance


def load_model(alias):
    """Get a model from the catalog, download it if needed, load it, return it."""
    manager = get_manager()
    model = manager.catalog.get_model(alias)
    if model is None:
        # get_model returns None for unknown aliases - stop with a clear
        # message instead of crashing later with a confusing AttributeError.
        raise SystemExit(
            f"Model '{alias}' was not found in the Foundry Local catalog. "
            "Check the model alias constants at the top of rag.py."
        )

    if not model.is_cached:
        # The model has to be downloaded once; after that it stays on disk.
        def show_progress(percent):
            # "\r" moves the cursor back to the start of the line, so the
            # percentage updates in place instead of printing many lines.
            print(f"\rDownloading {alias}: {percent:.0f}%", end="", flush=True)

        model.download(show_progress)
        print()  # finish the progress line with a newline

    model.load()
    return model


def cosine_similarity(a, b):
    """Cosine similarity between two vectors, in pure Python.

    Returns a value between -1.0 and 1.0 (higher = more similar).
    If either vector has zero length, we return 0.0 to avoid dividing by zero.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def get_top_chunks(embedding_client, query, k=3, db_path=DB_PATH, min_score=MIN_SCORE):
    """Find the k stored chunks most similar to the query.

    Embeds the query, compares it against every chunk in the database, and
    returns a list of dicts sorted best-first:
        [{"source": str, "content": str, "score": float}, ...]

    Chunks scoring below min_score are dropped, so the list can be EMPTY —
    callers should treat that as "the documents don't cover this question"
    and answer "I don't have that information" without asking the model.
    """
    # Fail with a clear message instead of letting sqlite silently create
    # an empty database file (which would hide the real problem).
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database '{db_path}' not found - run 'python ingest.py' first."
        )

    response = embedding_client.generate_embedding(query)
    query_vector = response.data[0].embedding

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT source, content, embedding FROM chunks"
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise SystemExit(
            f"Database '{db_path}' is unreadable or has no 'chunks' table "
            f"({error}). Delete it and run 'python ingest.py' to rebuild."
        )
    finally:
        connection.close()

    scored = []
    for source, content, embedding_json in rows:
        # Embeddings are stored as JSON text, so turn them back into lists.
        vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, vector)
        scored.append({"source": source, "content": content, "score": score})

    # Drop chunks that are not really similar to the question — sending
    # barely-related text to the model invites it to make things up.
    scored = [item for item in scored if item["score"] >= min_score]

    # Sort by similarity, highest first, and keep only the best k.
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


def build_messages(question, chunks):
    """Build the chat message list: a system prompt with context + the question."""
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[{chunk['source']}]\n{chunk['content']}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a strict document-based assistant. Rules:\n"
        "1. Answer ONLY with information found in the context below.\n"
        "2. NEVER use your own knowledge, even if you know the answer. Do not "
        "add facts, examples, or recipes that are not in the context.\n"
        "3. If the context does not contain the answer, reply with exactly "
        "this sentence and NOTHING more: I don't have that information.\n"
        "4. When you do answer, mention the name(s) of the source "
        "document(s) you used.\n"
        "5. Always answer in clear English, even if the question is in "
        "another language.\n\n"
        "Context:\n"
        f"{context}"
    )

    # The language rule is repeated right after the question: small models
    # follow an instruction in the user turn more reliably than one buried
    # in a long system prompt (a Turkish question once got a garbled Turkish
    # answer despite rule 5 above).
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question + "\n\n(Answer in English.)"},
    ]


def stream_answer(chat_client, question, chunks):
    """Yield the answer text piece by piece as the model generates it.

    This is a generator, so the caller can print each piece immediately and
    the answer appears word by word instead of all at once at the end.
    """
    messages = build_messages(question, chunks)
    for chunk in chat_client.complete_streaming_chat(messages):
        # The final stream chunk can arrive with an empty choices list
        # (it only carries end-of-stream bookkeeping), so check first.
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        # Some stream chunks carry no text (for example role markers) — skip them.
        if content:
            yield content
