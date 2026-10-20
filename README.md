# 🤖 RAG ML Tutor Assistant

An AI-powered **Machine Learning and Deep Learning tutor** built with Retrieval-Augmented Generation (RAG). It answers questions using trusted ML reference documents, retrieves relevant passages, and provides **source citations with page numbers** to make answers more traceable and grounded in the provided knowledge base.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/RAG-Powered-8A2BE2" alt="RAG" />
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--1.5B-green" alt="Qwen LLM" />
  <img src="https://img.shields.io/badge/Vector%20Search-FAISS-orange" alt="FAISS" />
  <img src="https://img.shields.io/badge/Embeddings-E5--Base--V2-red" alt="E5 Embeddings" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License" />
</p>

---

## 📌 Overview

Large Language Models (LLMs) can generate helpful explanations, but they may also produce inaccurate information or answers unsupported by a specific reference.

**RAG ML Tutor Assistant** addresses this challenge by retrieving relevant information from a custom collection of Machine Learning and Deep Learning documents before generating an answer.

Instead of relying entirely on the LLM's internal knowledge, the system uses retrieved passages as context and displays their source files, page numbers, and similarity scores.

### ✨ Key Features

- 📚 **Document-based Question Answering** — Ask questions about Machine Learning, Deep Learning, and related topics.
- 🔍 **Semantic Search** — Retrieve relevant passages using dense vector embeddings.
- 🧠 **LLM-powered Explanations** — Generate contextual answers using Qwen2.5-1.5B-Instruct.
- 📑 **Source Citations** — Display document names, page numbers, and retrieval scores.
- 🧩 **Custom Knowledge Base** — Add your own PDF reference materials.
- ⚡ **FAISS Vector Search** — Efficient similarity search over embedded document chunks.
- 🎯 **Context-grounded Responses** — Instruct the LLM to answer using the retrieved context and acknowledge when the information is insufficient.

---

## 🏗️ How It Works

The system follows a standard Retrieval-Augmented Generation pipeline:

```text
                📚 PDF Documents
                       |
                       ▼
               📄 Text Extraction
                  (PyMuPDF)
                       |
                       ▼
                  ✂️ Chunking
             (150 words, overlap 30)
                       |
                       ▼
                🧠 Embeddings
                (E5-Base-V2)
                       |
                       ▼
                 🗂️ FAISS Index
                       |
                       |
👤 User Question ──► Query Embedding
                       |
                       ▼
                🔎 Top-k Retrieval
                       |
                       ▼
             📖 Retrieved Context
                       |
                       ▼
             🤖 Qwen2.5-1.5B-Instruct
                       |
                       ▼
             💬 Answer + 📑 Sources
```

### 🔄 Pipeline Breakdown

**1. 📥 Document Ingestion**

PDF documents are downloaded or loaded locally and processed page by page using PyMuPDF. The extracted text retains its source filename and page number.

**2. ✂️ Text Chunking**

Extracted text is divided into overlapping word-based chunks. The default configuration uses 150 words per chunk with a 30-word overlap to preserve some context between consecutive chunks.

**3. 🧠 Embedding Generation**

The `intfloat/e5-base-v2` embedding model converts document passages into dense vector representations. It uses the recommended `passage:` prefix for document chunks and `query:` for user questions, with normalized embeddings.

**4. 🗂️ Vector Indexing**

The generated vectors are stored in a FAISS `IndexFlatIP` index. Because the vectors are normalized, inner-product search corresponds to cosine similarity.

**5. 🔎 Semantic Retrieval**

When a user submits a question, the system embeds the query and retrieves the top-k most similar chunks from the FAISS index.

**6. 🤖 Answer Generation**

The retrieved passages and user question are passed to `Qwen/Qwen2.5-1.5B-Instruct`. A tutor-oriented system prompt instructs the model to base its answer on the provided context rather than unsupported external knowledge.

**7. 📑 Source Attribution**

The system displays the retrieved source documents, page numbers, and similarity scores to help users inspect the supporting material.

---

## 📚 Default Knowledge Base

The project currently uses the following reference materials:

| Document | Description |
|---|---|
| 📄 *Attention Is All You Need* | The research paper introducing the Transformer architecture. |
| 📘 *Dive into Deep Learning* | An interactive textbook covering Deep Learning concepts, models, and implementations. |

The knowledge base is customizable, allowing you to add other Machine Learning, Deep Learning, NLP, or Computer Vision PDFs.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 📄 PyMuPDF | PDF text extraction |
| 🧠 Sentence Transformers | Document and query embeddings |
| 🔢 `intfloat/e5-base-v2` | Semantic embedding model |
| 🗂️ FAISS | Vector indexing and similarity search |
| 🤖 Qwen2.5-1.5B-Instruct | Context-based answer generation |
| 🔥 PyTorch | Deep Learning framework |
| 🤗 Hugging Face Transformers | LLM loading and inference |
| ⚡ Accelerate | Model execution and device support |

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.9 or later
- Git
- Sufficient RAM to load the embedding model and LLM
- A GPU is recommended for faster generation, but CPU execution may be possible with reduced performance

### 📥 1. Clone the Repository

```bash
git clone https://github.com/bassam519/rag-ml-tutor-assistant.git
cd rag-ml-tutor-assistant
```

### 📦 2. Install Dependencies

```bash
pip install pymupdf sentence-transformers faiss-cpu transformers accelerate torch requests
```

### ▶️ 3. Run the Notebook

Open the notebook in Jupyter:

```bash
jupyter notebook ml_assistant.ipynb
```

Alternatively, open the notebook directly in Google Colab and execute the cells in order.

The final notebook cell launches an interactive question-answering session.

---

## 💬 Example Usage

```text
Welcome to the RAG Q&A Bot!
Ask a question or type 'exit' to quit.

User: What is self-attention?

HelpBot:
Self-attention is a mechanism that allows a model to
determine how much importance each token should assign
to other tokens when building contextual representations.

Sources:
- attention_is_all_you_need.pdf | Page: 3 | Score: 0.842
- dive_into_deep_learning.pdf   | Page: 129 | Score: 0.791
```

*Example output for illustration; actual answers, retrieved pages, and scores depend on the documents and query.*

---

## ⚙️ Configuration

The following parameters control the default retrieval and generation behavior:

| Parameter | Default | Description |
|---|---:|---|
| `chunk_size` | `150` | Number of words in each chunk |
| `overlap` | `30` | Number of overlapping words between chunks |
| `top_k` | `8` | Number of chunks retrieved per question |
| `max_new_tokens` | `150` | Maximum number of new tokens generated |
| `max_length` | `4096` | Configured maximum prompt length in tokens |

Adjust these values to experiment with retrieval quality, contextual coverage, memory usage, and response length. Ensure the prompt fits within the model's actual context window.

---

## ➕ Adding Your Own Documents

You can customize the knowledge base with your own PDF reference materials.

1. 📂 Place your PDF files inside the `ml_documents/` directory, if supported by the notebook's ingestion code.
2. 🔄 Add their paths or download URLs to the document configuration, if required.
3. ✂️ Run the text extraction and chunking steps.
4. 🧠 Generate embeddings for the new chunks.
5. 🗂️ Rebuild the FAISS index and associated chunk metadata.
6. 💬 Ask questions about your new documents.

**Important:** Re-run the ingestion, embedding, and indexing steps whenever you change the knowledge base.

---

## 💾 Saving the Pipeline

The notebook includes a commented section for saving components of the RAG pipeline, including:

- 🤖 LLM and tokenizer
- 🧠 Embedding model
- 🗂️ FAISS index
- 📑 Chunk text and metadata

Saving these components can reduce startup time and avoid rebuilding the index every time you run the application.

For deployment, keep the index, metadata, and model configuration synchronized. Large model weights can be stored separately from GitHub when necessary.

---

## ⚠️ Limitations

- 🧠 **Model Capacity:** The 1.5B-parameter LLM may produce shallow or incomplete explanations for complex topics.
- 📄 **PDF Extraction:** Tables, equations, diagrams, and complex layouts may not be extracted accurately.
- ✂️ **Chunk Boundaries:** Word-based chunking can separate related concepts or split explanations across chunks.
- 🔎 **Retrieval Quality:** The system's performance depends on the embedding model, document quality, and retrieval settings.
- 🎯 **No Reranking:** The current pipeline does not include a dedicated reranking stage.
- 🛡️ **Grounding Is Not Guaranteed:** Prompt instructions encourage context-based answers but cannot completely prevent hallucinations.
- ⚡ **Hardware Requirements:** Loading the LLM and embedding model can require substantial RAM or GPU memory.

---

## 🔮 Future Improvements

Potential enhancements to make the project more robust and production-ready:

- 🖥️ Build a **Streamlit or Gradio** web interface.
- 🔄 Add a **Cross-Encoder Reranker** to improve retrieval relevance.
- 🔍 Experiment with **Hybrid Search**, combining semantic and keyword retrieval.
- ✂️ Implement sentence-aware or section-aware chunking.
- 📊 Evaluate retrieval and answer quality using appropriate metrics and test questions.
- 🧪 Add automated tests for retrieval, source attribution, and answer generation.
- ⚡ Optimize inference using quantization or other memory-efficient techniques.
- 💾 Add persistent storage and automatic index loading.
- 💬 Support conversation history and follow-up questions.
- 📑 Improve citation formatting and expose the exact retrieved passages supporting each answer.

---

## 🎯 Learning Objectives

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Text preprocessing and chunking
- Dense vector embeddings
- Semantic similarity search
- Vector databases and FAISS indexing
- Prompt engineering and context-grounded generation
- PDF document processing and metadata management
- Building an end-to-end AI question-answering pipeline

---

## 👨‍💻 Author

**Bassam Omar Ibrahim**

Aspiring Machine Learning Engineer from Egypt 🇪🇬

- 🐙 GitHub: [@bassam519](https://github.com/bassam519)
- 💡 Interests: Machine Learning, Deep Learning, NLP, Computer Vision, and Generative AI

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

⭐ **If you find this project useful, consider giving it a star!**

Contributions, suggestions, and feedback are welcome.
