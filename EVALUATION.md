# Evaluation: Test Results

Week 5 deliverable: a list of the queries attempted, whether the responses were correct/appropriate, the shortcomings found, and the adjustments made.

## 1. Purpose

`test_questions.py` asks the assistant 14 fixed questions in three categories (5 answerable, 3 unanswerable, 6 edge cases) and writes a transcript to `test_transcript.txt`. It uses exactly the same retrieval (`get_top_chunks`, top-3 chunks, 0.30 similarity floor) and the same prompt (`build_messages`) as the interactive app, but with a non-streaming call and `temperature = 0.0` so the transcript is repeatable. There are no automatic pass/fail asserts on purpose; the verdicts below are a human reading of the transcript against the expected behavior.

## 2. Results

Scores are cosine similarities between the question and the retrieved chunks (1.0 = identical meaning). Times are the model's answer time for that question, from the `Time:` lines in `test_transcript.txt`.

| # | Category | Question (shortened) | Retrieved sources (top score) | Expected behavior | Observed behavior | Time | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | answerable | How does DHCP assign an IP address to a new host? | Chapter4.pdf x3 (0.60) | DHCPDISCOVER broadcast, relay agent, address from a pool, with source | Exactly that: broadcast to 255.255.255.255, relay agent when the server is on another network, dynamic assignment from a pool; cites Chapter4.pdf | 28 s | PASS |
| 2 | answerable | What is the difference between UDP and TCP? | Chapter5.pdf x3 (0.58) | UDP = simple demultiplexing, no delivery guarantee; TCP = reliable connection-oriented byte stream with flow and congestion control | Correct on all four points, concise; cites Chapter5.pdf | 28 s | PASS |
| 3 | answerable | What is CIDR and how does longest prefix match work? | Chapter4.pdf (0.75), Chapter4_add.pdf (0.74) | a.b.c.d/x notation, choose the longest matching prefix, with source | Correct definition and the slides' own example (171.69/16 vs 171.69.10/24 -> the /24 wins); cites both files | 50 s | PASS |
| 4 | answerable | How does DNS resolve a hostname to an IP address? | Chapter9.pdf x3 (0.64) | Hierarchy of name servers: local -> root -> second level ..., NS/A/CNAME/MX records | Correct 5-step walk-through with the slides' cit.princeton.edu example; cites Chapter9.pdf | 38 s | PASS |
| 5 | answerable | How does CSMA/CD handle collisions in Ethernet? | Chapter2.pdf x3 (0.70) | Collision detected -> 32-bit jam sequence -> stop -> back off | Collision detection, 32-bit jamming sequence, stop, and the 96-bit minimum (64-bit preamble + 32-bit jam); cites Chapter2.pdf | 46 s | PASS (note: does not mention the exponential backoff that follows, which is also in the slides) |
| 6 | unanswerable | Capital of Australia? | none above 0.30 (best 0.24) | "I don't have that information", model not called | Deterministic refusal from the threshold; model never called | - | PASS |
| 7 | unanswerable | Who won the 2026 FIFA World Cup? | none above 0.30 (best 0.16) | Same as #6 | Deterministic refusal; model never called | - | PASS |
| 8 | unanswerable | Recipe for chocolate cake? | none above 0.30 (best 0.21) | Same as #6 | Deterministic refusal; model never called | - | PASS |
| 9 | edge case | Empty input `''` | not run | No crash, no model call | Skipped; the interactive app treats empty input as exit | - | PASS |
| 10 | edge case | Whitespace-only `'   '` | not run | No crash, no model call | Same as #9 (input is stripped first) | - | PASS |
| 11 | edge case | Very long, rambling "what happens from typing a URL to seeing the page?" | Chapter9.pdf (0.52), Chapter1.pdf x2 | A grounded, layered overview; the slides have no such walk-through, so the model should not invent one | Gives a loose layered overview from the retrieved slides and explicitly notes that the context "does not give a detailed step-by-step explanation"; but it also drags email protocols (SMTP, MIME) into a web-browsing question and calls TCP/UDP "network layer" | 225 s | PASS with note (grounded and honest about the gap, but the weakest answer: the retrieved slides are only loosely related and the model padded the answer with them) |
| 12 | edge case | Turkish: "TCP ile UDP arasındaki fark nedir?" | Chapter5.pdf x3 (0.50) | Correct answer in English (prompt rule 5), with source | Retrieval was right (all three chunks from Chapter5.pdf) and the content is roughly right, but the model ignored the English-only rule: it answered in Turkish, the text degenerated into repetitive, partly meaningless sentences and was cut off at the 500-token limit | 96 s | FAIL (language rule not followed; see section 5 for the fix applied afterwards) |
| 13 | edge case | Dijkstra / link-state routing | Chapter4.pdf (0.59), Chapter3.pdf | Refusal: routing slides are retrieved, but Dijkstra and link-state routing are not in them | Clean refusal ending with the exact sentence | 22 s | PASS |
| 14 | edge case | How does traceroute discover the routers along a path? | Chapter4.pdf (0.47), Chapter1.pdf | Refusal: ICMP and routing are in the slides, traceroute is not | Refuses, and correctly lists what the context does cover (DHCP, ICMP, addressing, routing) | 23 s | PASS |

## 3. Summary

**13/14 pass** (11 clean passes, 2 passes with a note, 1 fail). Against the three functional-testing requirements of the course plan:

1. **Answer when the information is in the documents** (#1-5): every answerable question retrieved the right chapter as its top chunk (scores 0.58-0.75), answered correctly with the slides' own examples, and named its source file.
2. **Appropriate fallback when the information is missing** (#6-8, #13-14): the three off-topic questions scored at most 0.24 and were refused by the similarity floor without calling the model at all. The two adjacent-topic questions (#13-14) are the strongest evidence of grounding: routing and ICMP slides were retrieved at 0.47-0.59 similarity, yet the model did not invent Dijkstra or traceroute explanations.
3. **Edge cases** (#9-14): empty and whitespace input never reach the models, a rambling question is handled (with the caveat noted in #11), the Turkish question is the one failure (retrieval worked, 0.50 from the right chapter, but the answer came back in garbled Turkish instead of English), and nothing crashed.

Answer quality (accurate, concise, sources cited) is good for the five answerable questions. The notes are about completeness (#5) and about a loosely related long question (#11), both discussed in section 5.

## 4. Performance

`main.py` prints `(answered in X s)` after every answer and `test_questions.py` prints `Time: X s` after every model call, so response times are visible on every run. The numbers below come from the run that produced `test_transcript.txt` (2026-09-15, development laptop, CPU only, 16 GB RAM, no GPU); every model call in the transcript has its own `Time:` line.

| Stage | Measured |
|---|---|
| Retrieval: brute-force cosine over the 117 stored chunks (excluding embedding the question) | a few milliseconds |
| Generation: time to first token (`phi-3.5-mini`, CPU, streaming in `main.py`) | ~10 s |
| Generation: full answer from the slides (#1-5; up to 500 tokens) | 28-50 s, median 38 s |
| Generation: long, multi-part question (#11) | 225 s |
| Generation: prompt-level refusals (#13-14) | 22-23 s |
| Refusals by the similarity floor (#6-8) | instant, no model call |

The course plan's guideline is ~1-3 s per question on a typical laptop, which assumes a small chat model. This project is roughly 10-30x slower than that, for two reasons. First, a deliberate choice: `phi-3.5-mini` has 3.8B parameters and was picked for answer quality, because the smaller catalog models were much worse at following the "answer only from the context" rules. Second, the three retrieved slide chunks (~900 characters each) plus the system prompt make a long prompt that the CPU must process before the first token appears. Retrieval and the database are not a factor at this size.

Optimizations considered, in the order I would try them:

- **Smaller chat model** (for example `qwen2.5-0.5b`): several times faster, at the cost of weaker grounding and writing quality. A one-line change of `CHAT_MODEL` in `rag.py`.
- **Lower `max_tokens`** (currently 500): answers in the transcript are roughly 80-250 words, so 300 would cut the worst-case time without truncating typical answers.
- **Retrieve 2 chunks instead of 3**: a shorter prompt means less to process before the first token. With 6 small documents the top chunk alone usually holds the answer.
- **GPU/NPU execution provider**: Foundry Local can run the same model on accelerated hardware; the laptop used here has none.
- **Not recomputing embeddings**: already done. Every chunk vector is computed once by `ingest.py` and stored in SQLite; at question time only the question itself is embedded.

## 5. Shortcomings found during development and adjustments made

| Shortcoming (early runs) | Adjustment |
|---|---|
| The model leaked its own knowledge: it answered "Canberra" for the capital of Australia, and for the cake question it first said it had no information and then wrote a full chocolate-cake recipe anyway. | Added the 0.30 similarity floor in `get_top_chunks`: when no chunk clears it, `main.py` answers "I don't have that information" without calling the model. Made the system prompt stricter (never use own knowledge, do not add facts/examples/recipes, reply with exactly one sentence when the context lacks the answer). Both cases now pass (#6, #8). |
| A Turkish question produced degenerate, partly nonsensical Turkish text from the small chat model. | Prompt rule 5 pins the answer language to English regardless of the question language. Retrieval still works across languages because the embedding model is multilingual (#12). |
| The last chunk of a streamed answer arrives with an empty `choices` list, which crashed the CLI at the end of every answer. | `stream_answer` in `rag.py` skips chunks with no choices (and chunks with no text). |
| Sources were printed under every answer, including refusals, which made it look as if the refusal was "based on" those documents. | `main.py` now collects the streamed answer and, if it contains "don't have that information", prints "(no sources cited)" instead of the source list. |
| The PDF lecture slides have no blank lines between paragraphs, so a whole page arrived as one 1,000+ character "paragraph" and became a single oversized chunk. | `ingest.py` now splits over-long paragraphs at sentence ends (`split_long_paragraph`) before packing them, so the 117 chunks average ~930 characters and none exceeds 1,200. |
| During a full evaluation run the native runtime once aborted a chat call with `Operation was cancelled` (a transient error inside Foundry Local, not reproducible on the retry), which crashed the whole 15-minute run at question 11. | `test_questions.py` now retries a failed model call once and otherwise records the error and moves on; `main.py` catches the same failure mid-answer, tells the user to ask again and keeps the session alive. |
| In this run the Turkish question (#12) was answered in Turkish despite prompt rule 5, and the small model's Turkish output degenerated into repetition. | The English-only instruction is now also appended to the user message itself in `build_messages` ("(Answer in English.)"), where small models follow instructions more reliably. To be confirmed by re-running the evaluation; the transcript above is the run before this change. |

Still visible in the transcript: the refusal wording is not always the exact sentence requested by the prompt (#8 and #13 add "I'm sorry" or one explanatory sentence). The app detects refusals by the phrase "don't have that information" rather than by exact match, so this does not affect behavior, but it shows that a 3.8B model follows "reply with exactly this sentence" only approximately. The typo in #11 ("embedds") is the model's, not the documents'.

## 6. Remaining limitations and future work

- **The threshold is calibrated on 13 questions.** The weakest on-topic score (0.42, Turkish) and the strongest off-topic score (0.36, cake) are only 0.06 apart. A larger labeled question set would show whether 0.30 is too generous and whether a reranker is worth adding.
- **Refusal detection is a string match.** A paraphrased refusal would still print sources. A structured signal (for example asking the model for a fixed token) would be more robust.
- **No automatic asserts.** The script relies on a human reading the transcript. Simple checks (refusal phrase present/absent, expected source in the retrieved list) would turn it into a regression test.
- **Latency.** The trade-off in section 4 should be measured, not assumed: run the same 13 questions with a smaller chat model and compare quality and time side by side.
- **Single-turn only.** Each question is retrieved independently, so follow-ups such as "what about the second one?" do not work.
