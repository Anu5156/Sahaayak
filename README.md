---
title: Sahaayak
emoji: 🏛️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---
# Sahaayak — Government Scheme Eligibility Assistant

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://sahaayak-1-ore5.onrender.com/)

- **Live Deployment:** [https://sahaayak-1-ore5.onrender.com/](https://sahaayak-1-ore5.onrender.com/)
- **Live API Docs:** [https://sahaayak-1-ore5.onrender.com/docs](https://sahaayak-1-ore5.onrender.com/docs)

Sahaayak helps citizens find Indian government schemes they can apply for, and explains **why** they qualify or what information is still missing. It also answers questions about schemes in plain English or Hinglish, with every answer grounded in official scheme information and shown with its sources.

Built for Smart India Hackathon (SIH) 2026.

## Features

- **Eligibility checker** — enter a few details (age, occupation, income, household facts) and get schemes grouped as *likely eligible*, *need more info*, or *not a match*, each with the exact rules that passed or are missing.
- **Deterministic rules engine** — eligibility is decided by explicit, testable rules written as data, not by an AI guess. Missing details never produce a wrong "yes"; they produce "need more info".
- **Hybrid retrieval (RAG)** — questions are matched against scheme sections using TF-IDF (word + character n-grams, so typos still match) fused with semantic similarity (LSA by default, or sentence-transformers embeddings).
- **Hinglish support** — common Hindi words are expanded before search (*kisan → farmer*, *ghar → house*, *beti → girl child*, …).
- **Grounded answers** — extractive answers with citations by default; optionally a local LLM (Ollama) writes a short answer using only the retrieved sources, falling back safely if it is unavailable.
- **Eligibility-aware answers** — share your profile with a question and each source is tagged with your eligibility for that scheme.
- **Citizen-facing web UI** — mobile-friendly, accessible single page with a sample profile and example questions.
- **REST API** with automatic OpenAPI docs at `/docs`.

Covers 16 central schemes across agriculture, health, housing, pensions, insurance, education, business loans, women & child welfare and skills (PM-KISAN, AB PM-JAY, PM Ujjwala, PMAY-G, PMAY-U 2.0, Sukanya Samriddhi, Atal Pension, PMJJBY, PMSBY, MUDRA, PM SVANidhi, Stand-Up India, Post Matric Scholarship for SC, PMMVY, PM Vishwakarma, IGNOAPS).

## Architecture

```
Browser (static/index.html)
      │  JSON
      ▼
FastAPI (app/main.py)
  ├── POST /api/check ──► Rules engine (app/rules.py) ──► schemes.json rules
  └── POST /api/ask   ──► Hybrid retriever (app/retrieval.py)
                             ├── lexical: TF-IDF word + char n-grams
                             └── semantic: LSA or sentence-transformers
                          ──► Answer layer (app/answer.py): extractive or Ollama LLM
                          ──► optional per-source eligibility from the rules engine
```

Fusion score: `alpha × semantic + (1 − alpha) × lexical`, each min-max normalised (default `alpha = 0.5`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the app and http://localhost:8000/docs for the API.

### Optional upgrades

| Feature | How to enable |
| --- | --- |
| Transformer embeddings | `pip install sentence-transformers` then set `SAHAAYAK_EMBEDDINGS=st` |
| Local LLM answers | Install [Ollama](https://ollama.com), run `ollama pull qwen2.5:1.5b`, set `SAHAAYAK_LLM=ollama` (model via `OLLAMA_MODEL`) |

### Docker

```bash
docker build -t sahaayak .
docker run -p 8000:8000 sahaayak
```

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Status, scheme count, retrieval backend |
| GET | `/api/schemes?category=Health` | List schemes, optionally by category |
| GET | `/api/schemes/{id}` | One scheme's details |
| POST | `/api/check` | Profile → eligible / maybe / not-eligible schemes with reasons |
| POST | `/api/ask` | Question (+ optional profile) → answer, mode and cited sources |

Example:

```bash
curl -X POST localhost:8000/api/check -H "Content-Type: application/json" \
  -d '{"age": 45, "occupation": "farmer", "owns_agri_land": true, "is_income_tax_payer": false, "is_govt_employee": false}'

curl -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question": "kisan ko paisa kaise milega"}'
```

## Adding a scheme

Add an entry to `app/data/schemes.json`. Rules are data:

```json
{"field": "age", "op": "between", "value": [18, 40], "reason": "age must be between 18 and 40"}
{"any": [ {...}, {...} ], "reason": "must be a woman or belong to SC/ST"}
```

Operators: `eq`, `ne`, `in`, `not_in`, `gt`, `gte`, `lt`, `lte`, `between`, `is_true`, `is_false`.

## Tests

```bash
pytest -q
```

27 tests cover the rules engine (boundaries, any-groups, missing data), retrieval quality (including Hinglish and typo queries) and every API endpoint.

## Project structure

```
app/
  main.py        FastAPI app and endpoints
  rules.py       Deterministic eligibility engine
  retrieval.py   Hybrid TF-IDF + semantic retriever
  answer.py      Extractive / Ollama answer generation
  models.py      Pydantic request and response models
  data/schemes.json
static/index.html  Web UI
tests/             pytest suite
```

## Disclaimer

Scheme rules are simplified for guidance and may change. Always confirm eligibility on the official portal linked for each scheme before applying.
