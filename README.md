# Indian Legal Documents RAG

Ask questions about five Indian laws and get answers with page-level citations. The app refuses questions the documents don't cover.

**Documents:** RTI Act 2005, Consumer Protection Act 2019, Motor Vehicles Act 1988, Digital Personal Data Protection Act 2023, IT Act 2000.

![Chat answer with sources](docs/screenshots/chat-answer.png)
![Refusal for an out-of-scope question](docs/screenshots/chat-refusal.png)

## How it works

```
PDFs -> clean + chunk (page-aware) -> embeddings (bge-small) -> ChromaDB
Question -> dense search + BM25 keyword search -> rank fusion -> cross-encoder reranker
         -> top passages (+ following chunks) -> LLM via OpenRouter or Groq -> answer + citations
```

- **Backend:** FastAPI (`/ask`, `/health`)
- **Frontend:** Streamlit chat with example questions, a sources panel, and a model switcher (any model id of the configured provider)
- **Everything runs on free tools:** local embeddings and reranker, free-tier OpenRouter or Groq models

## Results

I wrote my own test questions with the correct page and answer for each, in two sets. The first set (11 questions) was used while choosing the retrieval methods. The second set (13 questions) was written after that choice, but I later corrected four of its page labels and adjusted the answer-stage settings after two questions failed, so neither set is a fully untouched test.

### Retrieval: is the right page in the top 5?

| Method | Set 1 (11 q) | Set 2 (13 q), original labels | Set 2, corrected labels | All 24 (Set 1 + corrected Set 2) |
|---|---|---|---|---|
| Dense (embeddings only) | 7/11 (64%) | 7/13 (54%) | 11/13 (85%) | 18/24 (75%) |
| Hybrid (dense + BM25) | 8/11 (73%) | 7/13 (54%) | 11/13 (85%) | 19/24 (79%) |
| Hybrid + reranker | **10/11 (91%)** | **9/13 (69%)** | **13/13 (100%)** | **23/24 (96%)** |

Is the right page the first result? (a stricter measure)

| Method | Set 1 | Set 2, original labels | Set 2, corrected labels | All 24 |
|---|---|---|---|---|
| Dense | 6/11 | 1/13 | 5/13 | 11/24 (46%) |
| Hybrid | 6/11 | 3/13 | 7/13 | 13/24 (54%) |
| Hybrid + reranker | 8/11 | 2/13 | 6/13 | 14/24 (58%) |

Notes:
- On review, 4 of my own page labels in Set 2 were one page too high (the section started on the previous page). I corrected them only where the PDF text showed the answer on that page, and I show both versions above.
- The first-result measure is harsh: legal sections often span two pages, and the LLM receives several passages, not just the first one.
- With 24 questions, one question is worth about 4 percentage points. The ordering (reranker best, dense weakest) is the reliable finding, not the exact percentages.
- Hybrid search was mixed: it fixed three questions (RTI penalty, IT Act electronic records, definition of a child) and broke two that dense search had right (consumer complaint time limit, digital signature). The reranker recovered both.

<details>
<summary>Which questions each method missed (top 5, corrected labels)</summary>

| Question | Dense | Hybrid | Reranker |
|---|---|---|---|
| Q1 RTI reply deadline | miss | miss | miss |
| Q3 RTI penalty for delay | miss | hit | hit |
| Q4 Consumer complaint time limit | hit | miss | hit |
| Q5 District Commission orders | miss | miss | hit |
| Q11 IT Act electronic records | miss | hit | hit |
| Q19 Driving without a licence | miss | miss | hit |
| Q21 Who is a child (DPDP Act) | miss | hit | hit |
| Q24 Digital signature definition | hit | miss | hit |

All other questions were hits for all three methods. The reranker's only miss is Q1, where it returned the neighbouring Section 6 (page 10) instead of Section 7 (page 11).

</details>

### Answer quality

An LLM judge compares each answer with my reference answer (Correct / Partial / Wrong). The judge and answer model changed between runs, so the rows are not directly comparable.

| Run | Answer model | Judge | Questions | Correct | Partial | Wrong |
|---|---|---|---|---|---|---|
| 1. 5 passages sent, similarity gate 0.60 | free OpenRouter model | free OpenRouter model | 24 | 19 | 3 | 2 |
| 2. 8 passages sent, similarity gate 0.55 | free OpenRouter model | free OpenRouter model | 24 | **22** | 2 | 0 |
| 3. Model comparison | gpt-oss-120b (Groq) | gpt-oss-20b | 13 | 13 | 0 | 0 |
| 4. Model comparison | gpt-oss-120b (Groq) | qwen3.8-27b | 13 | 12 | 1 | 0 |
| 5. Model comparison | gpt-oss-20b (Groq) | qwen3.8-27b | 13 | 12 | 1 | 0 |

Notes:
- The two WRONG answers in run 1 were refusals ("I could not find this"), not invented facts. Sending 8 passages instead of 5, and lowering the similarity gate, fixed both.
- Runs 4 and 5 use a judge from a different model family than the answerers. Run 3 used a judge from the same family and scored the same answers higher (13/13 against 12/13), so the choice of judge matters.
- Both Groq models missed the same detail on one question (an "as soon as practicable" qualifier on the RTI transfer rule). With 13 questions and one run each, the two are effectively tied.
- A run with NVIDIA Nemotron 3 Ultra (free) was graded correct on all 12 questions it completed before a daily rate limit stopped it, so I don't report it as a full result.

### Refusals

3 out-of-scope questions (jaywalking fines, pollution fines, a state minimum wage): 3/3 correctly refused, both with a free OpenRouter model and with gpt-oss-120b.

### Limitations

- Small test sets (24 questions). The numbers show the approach works, not that it is 100% reliable.
- The final configuration was tuned on questions I had already seen, so treat the scores as optimistic.
- The judge is itself a free LLM, and free models vary from run to run and have daily rate limits.
- Refusal behaviour was tested on only 3 out-of-scope questions.
- English only for now. Hindi/Hinglish and voice input are planned.
- The RTI PDF is a scan, so its text has small OCR errors.
- Each question is answered independently, with no conversation memory.
- Not legal advice.

## Choose an LLM provider

The answer model is called through an OpenAI-compatible API. Set these in `.env`:

```
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
```

Use `LLM_PROVIDER=openrouter` with an OpenRouter model id to switch. Free tiers have rate limits, and model ids change over time.

## Run it yourself

1. Clone and install (Python 3.11):
   ```
   git clone https://github.com/anasilahi10/indian-legal-rag.git
   cd indian-legal-rag
   uv sync
   ```
   Or with pip: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

   Dependency versions are pinned to ones that work on an Intel Mac (for example `torch==2.2.2` and `numpy<2`).
2. Create `.env` from `.env.example` and add your Groq or OpenRouter API key (see "Choose an LLM provider" above).
3. Download the PDFs into `data/raw/` using these names:

   | File name | Source |
   |---|---|
   | `rti_act_en.pdf` | https://cic.gov.in/sites/default/files/RTI-Act_English.pdf |
   | `consumer_protection_act_en.pdf` | https://consumeraffairs.gov.in/public/upload/files/CP%20Act%202019_1732700731.pdf |
   | `motor_vehicles_act_en.pdf` | https://chdtransport.gov.in/sites/default/files/documents/MVA1988.pdf |
   | `dpdp_act_en.pdf` | https://www.indiacode.nic.in/bitstream/123456789/22037/2/a2023-22.pdf |
   | `it_act_en.pdf` | https://www.cca.gov.in/sites/files/pdf/ACT/ACT2000.pdf |

4. Build the index:
   ```
   uv run python -m backend.ingest
   uv run python -m backend.index
   ```
5. Start the backend and the chat screen in two terminals:
   ```
   uv run uvicorn backend.main:app --port 8000
   uv run streamlit run frontend/app.py
   ```

## Evaluate

```
uv run python -m scripts.evaluate_retrieval                                   # set 1
uv run python -m scripts.evaluate_retrieval data/eval/questions_holdout.csv   # set 2
uv run python -m scripts.evaluate_answers                                     # answer quality (both sets)
uv run python -m scripts.evaluate_refusals                                    # out-of-scope questions
```

`evaluate_answers` reads these optional environment variables: `LLM_MODEL` (answer model), `JUDGE_MODEL` (grader), `RUN_NAME` (results file label), `PAUSE` (seconds between questions), and `RESUME=1` (continue a run that stopped on a rate limit). It saves results to `data/eval/answer_results_<RUN_NAME>.csv`.

## Project structure

```
backend/        ingest, index, retriever, rag (LLM call), FastAPI app
frontend/       Streamlit chat screen
scripts/        evaluation scripts
data/eval/      test questions with expected pages and answers
docs/           screenshots
```

## Roadmap

- Hindi / Hinglish questions with a multilingual embedding model
- Voice input (Whisper)
- Section-aware chunking, and OCR cleanup for scanned PDFs
- A third, untouched test set
- Docker image and a hosted demo