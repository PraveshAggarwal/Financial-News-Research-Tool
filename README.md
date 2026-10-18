# 📈 Financial News Research Tool

A Streamlit-based web application that lets you research and query financial news articles using **Google Gemini LLM** and **FAISS vector search**. Paste up to 3 news article URLs, process them into a vector store, and ask natural language questions — the tool returns answers with cited sources.

---

## How It Works

```
URLs → Load → Split → Embed (Gemini) → FAISS Index → Query → Answer + Sources
```

1. You provide up to 3 news article URLs via the sidebar.
2. The app fetches and loads the article content using `UnstructuredURLLoader`.
3. Text is split into chunks using `RecursiveCharacterTextSplitter`.
4. Chunks are embedded with Google's `gemini-embedding-001` model and stored in a local FAISS index.
5. On query, the relevant chunks are retrieved and passed to `gemini-3.1-flash-lite` via a `RetrievalQAWithSourcesChain`.
6. The answer and source URLs are displayed in the UI.

---

## Project Structure

```
Financial(LLM)/
├── streamlit.py                    # Main Streamlit app
├── main.ipynb                      # Notebook: document loaders & text splitters exploration
├── faiss.ipynb                     # Notebook: FAISS vector store experiments
├── RetrievalQAwithSourcesChain.ipynb  # Notebook: RetrievalQA chain experiments
├── faiss_index/                    # Saved FAISS index (auto-generated)
│   ├── index.faiss
│   └── index.pkl
├── nvdia.txt                       # Sample text file used in notebooks
├── sample_text.csv                 # Sample CSV data
├── vector_index.pkl                # Pickled vector index (legacy)
├── vectorstore.pkl                 # Pickled vector store (legacy)
├── .env                            # Environment variables (API keys)
└── README.md
```

---

## Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key with access to Gemini models

---

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
cd "Financial(LLM)"
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install streamlit langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu unstructured python-dotenv
```

**4. Configure your API key**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Running the App

```bash
streamlit run streamlit.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. In the **sidebar**, paste up to 3 financial news article URLs.
2. Click **"Process URLs"** — the app will load, split, embed, and save the FAISS index locally.
3. Once processing is complete, type your question in the **"Question:"** input box.
4. The app returns an **answer** and the **source URLs** it referenced.

---

## Tech Stack

| Component | Library / Model |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (`gemini-3.1-flash-lite`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | FAISS (via `langchain-community`) |
| Document Loader | `UnstructuredURLLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| QA Chain | `RetrievalQAWithSourcesChain` |
| Env Management | `python-dotenv` |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `main.ipynb` | Explores `TextLoader`, `UnstructuredURLLoader`, `CharacterTextSplitter`, and `RecursiveCharacterTextSplitter` |
| `faiss.ipynb` | Experiments with building and querying FAISS vector stores |
| `RetrievalQAwithSourcesChain.ipynb` | Experiments with the `RetrievalQAWithSourcesChain` end-to-end |

---

## Notes

- The FAISS index is saved locally in the `faiss_index/` folder after processing. Re-processing URLs will overwrite it.
- `allow_dangerous_deserialization=True` is required when loading a local FAISS index — only load indexes you generated yourself.
- Keep your `.env` file out of version control. Add it to `.gitignore`:
  ```
  .env
  faiss_index/
  *.pkl
  ```
