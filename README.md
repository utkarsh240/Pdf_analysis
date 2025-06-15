# 📄 PDF Q&A with LangChain, OpenAI, and Qdrant

A **Streamlit app** that lets you upload a PDF and ask natural language questions about its content. It uses **LangChain**, **OpenAI**, and **Qdrant** to provide accurate answers based solely on the uploaded document.

---

## 🚀 **Features**

- **Upload and parse PDFs**
- **Split PDF text into smart chunks**
- **Embed and store content using Qdrant**
- **Ask context-aware questions powered by GPT-4**
- **Answers grounded in the uploaded PDF only**

---

## 🧰 **Tech Stack**

- **[Streamlit](https://streamlit.io/)** – Interactive Web UI
- **[LangChain](https://www.langchain.com/)** – Text splitting, Vector store handling, QA chain
- **[OpenAI](https://openai.com/)** – Embeddings + LLM (GPT-4)
- **[Qdrant](https://qdrant.tech/)** – High-performance vector database
- **[PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/en/latest/)** – PDF parsing and text extraction

---

## 📦 **Installation**

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
