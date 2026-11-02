# 🤖 LangGraph RAG Agentic Workflows

A conversational AI agent built with **LangGraph** that combines Retrieval-Augmented Generation (RAG) with local mathematical tool execution. The agent can answer domain-specific questions from a local database and perform precise calculations—all visualised dynamically with a live Streamlit trace UI.

---

## 💡 Why This Project? (The Core Idea)

Standard LLMs suffer from knowledge cutoff and math hallucinations. This project showcases how an **Agentic Loop** solves both issues by:
* **Retrieval (RAG):** Fetching context from local, structured knowledge bases using vector embeddings.
* **Tool-Use (Math):** Dynamically routing mathematical prompts to dedicated code execution tools rather than guessing the math.
* **Orchestration:** Using LangGraph's state machine to loop and re-evaluate answers until the target response is 100% accurate.

---

## ✨ Features At A Glance

* 📄 **RAG Document Search** — Fast semantic search over a FAISS vector index built from raw local `.txt` documents.
* 🧮 **On-The-Fly Arithmetic Tools** — Add, multiply, and divide tools executed directly by python logic.
* 🔄 **LangGraph Agent Loop** — An LLM-controlled system state that loops iteratively until it finds the optimal answer.
* 📊 **Live Sidebar Tool Trace** — A crisp Streamlit panel showing a complete breakdown of every tool invoked, its inputs, and outputs.
* ⚡ **High-Speed Inference** — Powered by `ChatGroq` for rapid model responses.
* 🧬 **Gemini Embeddings** — Generates rich context vectors using `gemini-embedding-001`.

---

## 📐 Architecture & Agent Flow

```text
       ┌──────────────────┐
       │   User Prompt    │
       └────────┬─────────┘
                │
                ▼
      ┌──────────────────┐  Tool Calls  ┌──────────────────┐
      │     llm_call     ├─────────────►│    tool_node     │
      │    (ChatGroq)    │◄─────────────┤ (Math / Search)  │
      └────────┬─────────┘ Tool Results └──────────────────┘
                │
                │ (No more tool calls needed)
                ▼
       ┌──────────────────┐
       │   Final Answer   │
       └──────────────────┘
```

### 🛠️ Agent Toolset

| Tool Name | Type | Description |
|:---|:---|:---|
| `search_docs` | Semantic | Performs similarity search over the vector knowledge base |
| `add` | Logic | Adds two integers cleanly |
| `multiply` | Logic | Multiplies two integers cleanly |
| `divide` | Logic | Divides two integers cleanly |

---

## 📂 Project Directory Structure

```text
.
├── agent.py               # Main Agent graph routing logic & state machine
├── app.py                 # Streamlit Chat Dashboard
├── ingest.py              # Ingestion pipeline to vectorize document chunks
├── main.py                # CLI Application trigger
├── sample_docs/           # Directory for domain knowledge raw texts
│   └── ai_concepts.txt    # Target clinical/AI domain concepts
├── faiss_index/           # Generated vector databases (Output)
├── pyproject.toml         # Dependencies lock file
└── .env.example           # Environment variables configuration guide
```

---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.12+
* Recommended: [`uv`](https://github.com/astral-sh/uv) (for ultra-fast dependency syncing) or standard `pip`

### 1. Clone & Download Dependencies
```bash
git clone <your-repo-url>
cd "LangGraph RAG"
uv sync
```

### 2. Configure Your API Keys
Generate a copy of your environment file:
```bash
cp .env.example .env
```
Open `.env` and fill in your private API keys (never commit these to Git!):
```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```
* 🔑 *Get a Google Embedding Key:* [Google AI Studio](https://aistudio.google.com/)
* 🔑 *Get a Groq Inference Key:* [Groq Console](https://console.groq.com/)

### 3. Initialize & Embed the Knowledge Base
Run this script once to chunk, embed, and store your local source documents inside the vector index:
```bash
uv run python ingest.py
```
*This splits `sample_docs/*.txt` files into ~500-character segments, runs them through Gemini Embeddings, and indexes them inside `faiss_index/`.*

### 4. Run the Streamlit Dashboard
```bash
uv run streamlit run app.py
```
Open up your browser to **http://localhost:8501** and start chatting!

---

## 📝 Example Prompts to Try

* **Pure Semantic Lookup:** *"What is RAG and why is it used?"*
* **Pure Math Tooling:** *"Calculate 42 multiplied by 7."*
* **Hybrid Agentic Reasoning:** *"If standard embeddings have 768 dimensions, how many float values is that for 15 vector documents?"*

---

## 📦 Core Dependencies

| Package | Purpose |
|:---|:---|
| `langgraph` | Direct Orchestration of State Machines & Loops |
| `langchain` | Wrapping LLMs, tools, and abstract interfaces |
| `langchain-groq` | Powering Groq inference endpoints |
| `langchain-google-genai` | Powering Google Gemini context embedding vectors |
| `faiss-cpu` | Managing vector space comparisons locally |
| `streamlit` | Lightweight web dashboard for runtime testing |
