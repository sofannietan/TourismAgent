# Cambodia Tourist Agent (RAG + Tools)

An academic project: a conversational tourism assistant for Cambodia, built with Python, LangChain and Google Gemini. It combines a **hybrid Retrieval-Augmented Generation (RAG)** pipeline over a curated travel corpus with an **agent** that can call practical tools (weather, currency conversion, best season to visit...). A router decides for each question whether to use the RAG chain or the agent.

> ⚠️ **Important: this project was built on the Google Gemini free tier.**
> Its capabilities are deliberately limited and heavily constrained by token and rate quotas. See [Limitations](#-limitations-free-api-tier) before judging the results.

---

## ✨ Features

- **Document ingestion** of `.txt` and `.pdf` files (Wikipedia pages, Wikivoyage export, travel guides), with cleaning and chunking.
- **Hybrid retrieval**: BM25 (lexical) + Chroma vector search (semantic, `gemini-embedding-001`) combined in an ensemble retriever, with optional **multi-query expansion**.
- **Query router**: sends each question either to the RAG chain or to the agent.
- **Tool-calling agent** with tools such as:
  - `search_in_corpus`: hybrid search in the knowledge base (returns sources)
  - `convert_currency`: currency conversion (e.g. EUR → KHR)
  - `get_weather`: forecast via [Open-Meteo](https://open-meteo.com/) (no API key needed)
  - `get_best_season`: best and worst months to visit a place
  - `get_current_date`: lets the agent resolve "next week", "in November"...
- **Conversation memory** per session.
- **Middleware**: input guardrails, logging callbacks, and a **semantic cache** (Chroma + `all-MiniLM-L6-v2`) to avoid repeated LLM calls.
- **Embedding cache on disk** (hash of model + task type + text) so re-indexing does not consume API quota again.

---

## 🏗️ Project Architecture

```text
Projet_2/
├── data/
│   ├── wiki_pages/          # Source documents (.txt, .pdf)
│   ├── data_collector.py    # Data collection script
│   └── data.py              # Static data (seasons, coordinates, ...)
├── src/
│   ├── agents/              # Agent logic and custom tools (agent.py, tools.py)
│   ├── chains/              # RAG and conversation chains (rag_chain.py, conversation.py)
│   ├── ingestion/           # Loading, chunking, embeddings (loader.py, chunker.py, embedder.py)
│   ├── memory/              # Conversation history management
│   ├── middleware/          # Logging, caching, guardrails (app_logging.py, cache.py, guardrails.py)
│   ├── output/              # Response structuring
│   └── retrieval/           # Hybrid search, query expansion, router
├── .gitignore
├── config.py                # Centralized settings (Pydantic Settings)
├── main.py                  # Main entry point
└── requirements.txt
```

### How a question flows

```text
User question
   │
   ├─► Semantic cache ──(hit)──► cached answer
   │
   ├─► Input guardrails
   │
   ├─► Router ──► "rag"   : contextualize → hybrid retrieval → LLM → answer
   │          └─► "agent" : LLM decides which tools to call → answer
   │
   └─► Cache write (only when the answer is safe to cache)
```

---

## 📚 Knowledge Base

The corpus is made of unstructured text, which is where RAG brings real value compared to a plain web search:

| Source | Type | Content |
|---|---|---|
| Wikipedia pages | `.txt` | Angkor Wat, Cambodia, cuisine, Kampot, Khmer Empire, Phnom Penh, Preah Vihear, Siem Reap, Sihanoukville, Tonle Sap |
| Wikivoyage – Cambodia | `.pdf` | Practical info: visas, transport, safety, health, etiquette, costs |
| Travel guides | `.pdf` | Itineraries, pre-departure advice, etiquette, destination facts |

Exact figures (prices, schedules, visa rules) in these documents age quickly and should be treated as **indicative**.

Structured data (coordinates, seasons) is **not** embedded: it is stored in `data/` and exposed through lookup tools, because exact lookups are more reliable than semantic search for this kind of information.

---

## 📋 Prerequisites

- **Python 3.10+**
- A **Google Gemini API Key** (free tier is enough to run the project, see limitations): [Google AI Studio](https://aistudio.google.com/)
- Internet access (Gemini API, Open-Meteo, exchange-rate API, first download of the `all-MiniLM-L6-v2` model)

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd Projet_2
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

---

## 💡 Usage

1. **Add documents** (`.txt` or `.pdf`) to `data/wiki_pages/` and list them in `config.py` (`file_paths`).

2. **Run the application:**
   ```bash
   python main.py
   ```
   The first run builds the vector store (`./chroma_db`) and the embedding cache (`./embedding_cache`). Later runs reuse them.

3. **Programmatic use:**
   ```python
   from src.chains.rag_chain import TourismAgentPipeline

   pipeline = TourismAgentPipeline(session_id="session_1")

   pipeline.query("What should I wear to visit temples?")                  # router decides
   pipeline.query("What's the weather in Siem Reap?", mode="agent")        # force the agent
   pipeline.query("Visa requirements for Cambodia?", mode="rag")           # force the RAG chain
   ```

4. **Rebuilding the index:** if you change documents or chunking parameters, delete `chroma_db/` (the embedding cache makes this cheap: only new or modified chunks call the API).

### Tips

- **Guardrails:** tune input policies in `src/middleware/guardrails.py`.
- **Configuration:** `chunk_size`, `chunk_overlap`, `top_k_retrieval`, and model names live in `config.py` or environment variables.
- **Caching:** the semantic cache in `src/middleware/cache.py` saves API calls, but answers that depend on real-time data (weather, exchange rates, dates) should not be cached for long.

---

## 🚧 Limitations (Free API Tier)

This project runs entirely on a **free Gemini API key**, which shapes almost every design decision and limits what the agent can do:

- **Very tight token and request quotas.** Every question can trigger several LLM calls (routing, query reformulation, multi-query expansion, tool calls, final answer). A single agent run that retrieves long passages can consume **thousands of tokens**, so only a limited number of questions can be asked before hitting a rate limit (HTTP 429).
- **Small retrieval budget.** Only a few chunks are retrieved per query and context is kept short. This reduces answer completeness, and some questions may lack the detail a paid setup would provide.
- **No heavy agent behavior.** Long multi-step plans (full itineraries with budgets, comparisons, many tool calls in a row) are not realistic under the quota. The agent is kept to short tool chains.
- **Query expansion is expensive.** Multi-query retrieval adds an LLM call and several embedding calls per question; it is the first thing to disable when quotas run out.
- **Caching is a necessity, not an optimization.** Both the disk embedding cache and the semantic response cache exist mainly to stay within the free quota.
- **Limited evaluation.** Large-scale evaluation (hundreds of questions, several configurations) is not feasible, so the evaluation set is small.
- **Model behavior may change.** Free-tier models, limits and availability can change at any time without notice.

Other limitations, independent of the API tier:

- The corpus is small (a few Wikipedia pages and a handful of guides), so coverage of regions, restaurants or hotels is partial.
- Prices, visa rules and security advisories in the documents may be outdated, and the Cambodia–Thailand border situation in particular changes quickly. Always check official sources before traveling.
- There is no web search tool, so the agent cannot answer breaking-news questions.

With a paid API key, the same architecture could support richer answers: longer context, more retrieved chunks, multi-query expansion by default, reranking, and longer agent reasoning chains.

---

## 🧭 Possible Improvements

- Replace the entire-page `.txt` retrieval with chunk-level retrieval everywhere (smaller context, fewer tokens).
- Add a reranker and metadata filters (province, category).
- Add a web-search fallback used only when retrieval finds nothing relevant.
- Build an evaluation set (question + expected source page) to compare: RAG only, hybrid vs. dense only, with/without query expansion.
- Add an itinerary tool and a budget estimator backed by structured data.

---

## 🛑 What to Git vs. What to Ignore

### Do NOT commit (in `.gitignore`):
* `.env` (secrets and API keys)
* `.venv/`
* `__pycache__/` and `*.pyc`
* `chroma_db/` and `embedding_cache/` (generated)
* `try-git.ipynb` (scratch notebooks)
* `data/wiki_pages/*` (local documents, **PDF guides are copyrighted and must not be redistributed**)

### DO commit:
* All source code in `src/`, `config.py`, `main.py`
* `requirements.txt`
* `.env.example`
* `.gitignore` and `README.md`

---

## 📄 Sources & Credits

- Wikipedia and Wikivoyage content under their respective Creative Commons licenses.
- Travel guides: Ayana Journeys, Sodha Travel, Khiri Travel (used for academic purposes only).
- Weather data: [Open-Meteo](https://open-meteo.com/).
- LLM and embeddings: Google Gemini.

*Academic project. Not intended for production or real travel decisions.*