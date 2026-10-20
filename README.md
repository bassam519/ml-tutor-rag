# RAG ML Tutor Assistant

A Retrieval-Augmented Generation (RAG) question-answering bot that acts as an AI and Machine Learning tutor. It answers questions using only the content of ML reference documents and cites the source file and page for every retrieved passage.

## Overview

Instead of relying on an LLM's memory, this project retrieves the most relevant passages from a custom knowledge base and passes them to a small open-source LLM as context. This keeps answers grounded in the source material and makes them traceable.

The default knowledge base contains:

- **Attention Is All You Need** (Vaswani et al., 2017)
- **Dive into Deep Learning** (d2l.ai)

You can swap in any PDFs you like.

## How It Works

```
PDFs → Text extraction → Chunking → Embeddings → FAISS index
                                                      │
User question → Query embedding → Top-k retrieval ────┘
                                        │
                          Retrieved context + question
                                        │
                              Qwen2.5-1.5B-Instruct
                                        │
                          Answer + sources (file, page, score)
```

1. **Ingestion**: PDFs are downloaded and parsed page by page with PyMuPDF.
2. **Chunking**: Each page is split into overlapping word-based chunks (150 words, 30-word overlap), keeping source and page metadata.
3. **Embedding**: Chunks are encoded with `intfloat/e5-base-v2` using the `passage:` / `query:` prefixes and normalized vectors.
4. **Indexing**: Embeddings are stored in a FAISS `IndexFlatIP` index (cosine similarity via inner product on normalized vectors).
5. **Retrieval**: The top-k most similar chunks (default 8) are fetched for each question.
6. **Generation**: `Qwen/Qwen2.5-1.5B-Instruct` answers using a tutor system prompt that restricts it to the provided context.
7. **Citations**: Each answer is followed by the source document, page number, and similarity score of the retrieved chunks.

## Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | PyMuPDF |
| Embedding model | `intfloat/e5-base-v2` (Sentence Transformers) |
| Vector search | FAISS (CPU) |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` (Hugging Face Transformers) |
| Runtime | Python, PyTorch (Google Colab compatible) |

## Getting Started

### Prerequisites

- Python 3.9+
- A GPU is recommended for faster generation, but not required

### Installation

```bash
git clone https://github.com/<your-username>/rag-ml-tutor-assistant.git
cd rag-ml-tutor-assistant
pip install pymupdf sentence-transformers faiss-cpu transformers accelerate torch requests
```

### Run

Open the notebook and run all cells:

```bash
jupyter notebook ml_assistant.ipynb
```

Or open it directly in Google Colab. The final cell launches an interactive chat loop in the terminal/notebook.

## Usage

```
Welcome to the RAG Q&A Bot!
Ask a question or type 'exit' to quit.

User: What is self-attention?

HelpBot: <answer grounded in the documents>

Sources:
- attention_is_all_you_need.pdf | Page: 3 | Score: 0.842
- ...
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 150 | Words per chunk |
| `overlap` | 30 | Overlapping words between chunks |
| `top_k` | 8 | Number of chunks retrieved per question |
| `max_new_tokens` | 150 | Maximum length of the generated answer |
| `max_length` | 4096 | Maximum prompt length in tokens |

### Adding your own documents

Add entries to the `documents` dictionary (filename → URL), or place PDF files directly in the `ml_documents/` folder, then re-run the chunking, embedding, and indexing cells.

### Saving the pipeline

The notebook includes a commented section for saving the LLM, tokenizer, embedding model, FAISS index, and chunk metadata so the system can be reloaded without rebuilding.

## Limitations

- Uses a small 1.5B-parameter model, so answers may be shallow on complex questions.
- Word-based chunking can split concepts across chunks, and PDF extraction may lose equations and tables.
- Answer length is capped at 150 new tokens.
- No reranking step; retrieval quality depends entirely on the embedding model.

## Future Improvements

- Add a reranker (e.g., a cross-encoder) on top of FAISS retrieval
- Use a larger or quantized LLM for better answers
- Sentence- or section-aware chunking
- Streamlit or Gradio web interface
- Retrieval and answer quality evaluation

## License

Released under the MIT License.
