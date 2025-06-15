📄 PDF Q&A with LangChain, OpenAI, and Qdrant
This Streamlit app allows users to upload a PDF file and ask questions about its content using natural language. It utilizes LangChain for LLM orchestration, OpenAI for embeddings and answering, and Qdrant for storing vectorized document chunks.

🚀 Features
Upload and parse PDFs

Chunk PDF content using LangChain's recursive splitter

Store and search document chunks using Qdrant vector store

Ask context-aware questions powered by GPT-4

Custom prompt to ensure grounded, PDF-based answers

🧰 Tech Stack
Streamlit – Web UI

LangChain – Chunking, Vector DB interface, QA chain

OpenAI – Embeddings and LLM

Qdrant – Vector storage and retrieval

PyMuPDF (fitz) – PDF text extraction

📦 Requirements
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
requirements.txt
shell
Copy
Edit
streamlit
pymupdf
openai
qdrant-client
tiktoken
python-dotenv
langchain>=0.1.13
langchain-openai
langchain-community
⚙️ Environment Setup
Create a .env file in your project root:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
Ensure Qdrant is running locally on port 6333 (you can use Docker):

yaml
Copy
Edit
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
🧠 How It Works
Upload PDF: The user uploads a PDF.

Text Extraction: Pages are parsed with fitz (PyMuPDF).

Chunking: The text is split into overlapping chunks.

Embedding + Vector Store: Chunks are embedded via OpenAI and stored in Qdrant.

Retrieval + QA: When a question is asked, relevant chunks are retrieved and passed to GPT-4 for answering.

▶️ Run the App
arduino
Copy
Edit
streamlit run app.py
📌 Notes
Uses RetrievalQA chain with mmr search for diverse, high-relevance results.

Replace gpt-4.1 with another model if needed.

Collection is recreated each run (force_recreate=True), which clears previous data.

🧑‍💻 Author
Made with ❤️ by Utkarsh Gupta (utk24g@gmail.com)
