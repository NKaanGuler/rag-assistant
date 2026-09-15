# Local RAG AI Assistant with Microsoft Foundry Local

A fully offline document question-answering assistant built with [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local). It answers questions about a small set of course documents using Retrieval-Augmented Generation (RAG) — no cloud APIs, no API keys, everything runs on your own machine.

Built by Nevzat Kaan Güler as the final project of the Microsoft AI Innovators Internship Program (Summer 2026).

## What it does

You ask a question in the terminal. The assistant finds the most relevant passages from the documents in `docs/`, feeds them to a local language model as context, and streams back an answer with source citations. If the answer isn't in the documents, it says so instead of making something up.

The knowledge base used for the demo and the evaluation is the lecture slides of **CMPE 344 – Computer Networks** (seven PDF chapters following Peterson & Davie, *Computer Networks: A Systems Approach*): foundations, getting connected (Ethernet, Wi-Fi, cellular), internetworking (IP, ARP, DHCP, NAT, IPv6, CIDR, routing), end-to-end protocols (UDP, TCP, congestion control) and applications (HTTP, DNS, SMTP, multimedia). Those slides are course material and are **not included in this repository**. To run the assistant, put your own documents (`.pdf`, `.pptx`, `.docx`, `.txt` or `.md`) into `docs/` — the pipeline is the same for any document set.

## Architecture

```
 docs/  (txt, md, pdf, docx, pptx)
      |
      v
 +-----------+     +--------------------+
 | ingest.py | --> | SQLite (rag.db)    |
 | chunk +   |     | chunks + JSON      |
 | embed     |     | embedding vectors  |
 +-----------+     +--------------------+
                            |
      question              v
      ----------> +--------------------+
                  | retrieval (rag.py) |
                  | cosine similarity, |
                  | top-k chunks       |
                  +--------------------+
                            |
                            v
                  +--------------------+     +------------------+
                  | augmented prompt   | --> | local LLM        | --> streamed
                  | context + grounding|     | (phi-3.5-mini)   |     answer
                  | instructions       |     | via Foundry Local|     + sources
                  +--------------------+     +------------------+
```

## Setup (Windows)

1. Install [Python 3.12](https://www.python.org/downloads/).
2. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   This installs `foundry-local-sdk` (1.x) and `openai` (pinned explicitly in `requirements.txt`; the SDK's client classes also depend on it).

## Usage

```
python main.py
```

That's all you need — on first run, `main.py` automatically ingests the documents in `docs/` (put at least one there first) if `rag.db` doesn't exist yet, then drops you into an interactive chat. Sources are printed after each answer.

Optional commands:

- `python ingest.py` — (re)build the database manually, e.g. after editing files in `docs/`.
- `python test_questions.py` — run a scripted transcript of answerable and deliberately unanswerable questions to check grounding behavior.

Each answer is followed by the list of source documents it was drawn from and the elapsed time in seconds, so you can see both where the answer came from and how long it took.

> **Note:** the first run downloads the two models (~3 GB total: `phi-3.5-mini` for chat and `qwen3-embedding-0.6b` for embeddings) through Foundry Local. Every run after that is fully offline.

## How RAG works here

1. **Ingest** — `ingest.py` splits each document in `docs/` into ~1200-character chunks, gets an embedding vector for each chunk from `qwen3-embedding-0.6b`, and stores the chunk text plus its vector (as JSON) in SQLite.
2. **Retrieve** — when you ask a question, `rag.py` embeds the question the same way and computes cosine similarity against every stored vector, keeping the top 3 chunks (`get_top_chunks`).
3. **Augment** — those chunks are pasted into a prompt with instructions telling the model to answer *only* from the provided context and to cite its sources.
4. **Generate** — `phi-3.5-mini` streams the answer token by token, and the CLI prints which documents the chunks came from.

The point of RAG: the model itself never "learns" the documents. It just gets shown the relevant parts at question time, which keeps answers grounded and lets you swap documents without retraining anything.

## Evaluation

`test_questions.py` runs 14 questions in three categories: 5 answerable (the answer is in the lecture notes), 3 unanswerable (clearly outside the documents) and 6 edge cases (empty input, a very long question, a Turkish question, and two topics adjacent to the notes but not covered by them). 13 of the 14 behaved as required: answerable questions were answered from the right chapter with sources, unanswerable and adjacent-but-uncovered questions were refused, and the edge cases were handled without crashing or improvising. The one failure — a Turkish question answered in garbled Turkish instead of English — is documented in `EVALUATION.md` together with the fix. The captured output is in `test_transcript.txt`; the full write-up with per-question verdicts and timings is in `EVALUATION.md`.

Honest numbers: on a CPU-only laptop (16 GB RAM, no GPU) a normal answer takes ~30–50 s (median ~38 s), with roughly 10 s before the first token appears; refusals that go through the model take ~22 s, and one long, multi-part question took 225 s. Retrieval itself (embedding the question and comparing it against every stored chunk) takes about 1 ms — practically all of the time is `phi-3.5-mini` generating text. The course guideline of ~1–3 s per question assumes a much smaller chat model; see Limitations for the trade-off.

## Design decisions

| Decision | Why | At larger scale you'd… |
|---|---|---|
| Chunk size ~1200 chars, paragraph-aware; over-long PDF paragraphs split at sentence ends | Big enough to keep an idea intact, small enough to stay focused and fit several chunks in the prompt; slide PDFs have no blank lines, so a page would otherwise become one oversized chunk | Tune per corpus; add overlap or semantic/heading-aware splitting |
| Top-k = 3 | With a small document set, 3 chunks almost always cover the answer without diluting the context | Raise k and add a reranker to filter noise |
| JSON vectors in SQLite | Zero extra infrastructure, human-inspectable, totally fine for a few hundred chunks | Use a vector store or SQLite extension (e.g. sqlite-vec) with binary vectors |
| Brute-force cosine similarity | Comparing a query against every chunk takes milliseconds at this size | Use an ANN index (HNSW, IVF) for sub-linear search over millions of vectors |
| Similarity threshold (0.30) | If no chunk scores above it, the app answers "I don't have that information" without calling the model — small models otherwise leak their own knowledge | Calibrate on a labeled eval set; consider a reranker |

## Limitations

- **Small model, small context** — `phi-3.5-mini` can misread nuance and occasionally hallucinates despite the grounding instructions; the test transcript exists to catch this.
- **Retrieval is only as good as the embeddings** — questions phrased very differently from the documents may miss the right chunk.
- **No conversation memory in retrieval** — each question is retrieved independently, so follow-ups like "what about the second one?" don't work well.
- **Your documents only** — quality depends on what's in `docs/`; the assistant knows nothing else. Questions in other languages are retrieved fine (the embedding model is multilingual), but answers are always in English — the small chat model produces unreliable text in other languages, so the prompt pins the output language.
- **No incremental ingest** — editing a document means re-running `ingest.py` over everything (fast at this scale, but still).
- **Slow on CPU** — `phi-3.5-mini` (3.8B parameters) was chosen for answer quality, but on a CPU-only laptop a normal answer takes ~30–50 s, and a long multi-part question can take minutes. A smaller chat model such as `qwen2.5-0.5b` would be several times faster (at the cost of weaker answers); lowering `max_tokens` (currently 500), retrieving 2 chunks instead of 3, or running on a GPU/NPU would also help. Embeddings are already cached in SQLite and never recomputed.

## Project structure

- `ingest.py` — splits the documents in `docs/` into chunks, embeds them and stores everything in `rag.db`.
- `rag.py` — Foundry Local setup, cosine-similarity retrieval with the 0.30 threshold, the grounding prompt and streaming answers.
- `main.py` — the interactive terminal chat (auto-ingests on first run).
- `test_questions.py` — scripted evaluation: 13 questions in three categories.
- `test_transcript.txt` — the captured output of `test_questions.py`.
- `EVALUATION.md` — test results with verdicts, timings and shortcomings.
- `docs/` — put the documents the assistant should answer from here (`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`). Shipped empty; the CMPE 344 slides used in the demo are not redistributed.
- `requirements.txt` — Python dependencies (`foundry-local-sdk`, `openai`, and the `pypdf` / `python-docx` / `python-pptx` readers).

## Credits

- [Tutorial: Build a RAG app with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app) — Microsoft Learn
- [What is Foundry Local?](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local) — Microsoft Learn
- [Building your first local RAG application with Foundry Local](https://azurefeeds.com/2026/03/30/building-your-first-local-rag-application-with-foundry-local/) — AzureFeeds
- [SQLite](https://sqlite.org/index.html) — the database that needs no server

Thanks to the Microsoft AI Innovators Internship Program (Microsoft Turkey, Summer 2026) for the course material and the mentoring that made this project possible.
