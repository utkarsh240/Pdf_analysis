import streamlit as st
import os
import fitz  # this works after installing pymupdf
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate



from qdrant_client import QdrantClient

# Load environment
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Config
COLLECTION_NAME = "pdf_qna"

# Streamlit setup
st.set_page_config(page_title="PDF Q&A with LangChain + Qdrant")
st.title("📄 Ask Questions About Your PDF")
st.markdown("Upload a PDF and ask questions. Powered by LangChain, OpenAI, and Qdrant.")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

@st.cache_data
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

@st.cache_resource
def load_vectorstore(chunks):
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    # Qdrant client and vectorstore
    qdrant_client = QdrantClient(host="localhost", port=6333)

    # Create vector store
    vectorstore = Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        location="http://localhost:6333",
        collection_name=COLLECTION_NAME,
        force_recreate=True,  # Replace collection each time
    )
    return vectorstore

@st.cache_data
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def get_qa_chain(vectorstore):
    custom_prompt = """
You are a helpful assistant. Your job is to answer questions using the provided PDF content only.

- If the answer is directly stated in the context, quote or summarize it.
- If the answer is **implied** or **related**, explain it clearly using logical reasoning.
- If the answer is **not in the context**, reply with: "I couldn't find the answer in the provided PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt
    )

    llm = ChatOpenAI(
        model_name="gpt-4.1",
        temperature=0.3,
        openai_api_key=openai_key,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

if pdf_file:
    with st.spinner("Extracting text..."):
        text = extract_text(pdf_file)

    with st.spinner("Splitting into chunks..."):
        chunks = split_into_chunks(text)

    with st.spinner("Storing chunks in Qdrant..."):
        vectorstore = load_vectorstore(chunks)

    st.success("✅ PDF processed and stored!")

    # Ask a question
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):
            qa_chain = get_qa_chain(vectorstore)
            result = qa_chain.run(question)

        st.write("### 📘 Answer:")
        st.write(result)
