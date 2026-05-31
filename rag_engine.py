# rag_engine.py (Gemini Version - Modern LangChain LCEL)

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# ─────────────────────────────────────────
# STEP 1: LOAD PDF
# ─────────────────────────────────────────
def load_document(pdf_path):
    print("📄 Loading document...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    return documents


# ─────────────────────────────────────────
# STEP 2: SPLIT INTO CHUNKS
# ─────────────────────────────────────────
def split_documents(documents):
    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────
# STEP 3: EMBEDDINGS (Free - HuggingFace)
# ─────────────────────────────────────────
def get_embeddings():
    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embeddings


# ─────────────────────────────────────────
# STEP 4: CREATE VECTOR DATABASE
# ─────────────────────────────────────────
def create_vector_store(chunks):
    print("💾 Creating vector store...")
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("✅ Vector store created!")
    return vector_store


# ─────────────────────────────────────────
# STEP 5: LOAD EXISTING VECTOR DATABASE
# ─────────────────────────────────────────
def load_vector_store():
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vector_store


# ─────────────────────────────────────────
# STEP 6: BUILD QA CHAIN (Using Gemini)
# ─────────────────────────────────────────
def build_qa_chain(vector_store):
    print("⛓️  Building QA chain...")

    prompt = PromptTemplate(
        template="""
You are a helpful assistant. Use the following context
to answer the question clearly and simply.

If you don't know the answer from the context,
just say "I couldn't find that in the document."

Don't make up answers.

Context:
{context}

Question: {question}

Answer:
""",
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain — no RetrievalQA needed
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ QA Chain ready!")
    # Return both chain and retriever as a dict
    return {"chain": chain, "retriever": retriever}


# ─────────────────────────────────────────
# STEP 7: ASK QUESTION
# ─────────────────────────────────────────
def ask_question(qa_chain, question):
    answer = qa_chain["chain"].invoke(question)

    # Get source docs for page numbers
    source_docs = qa_chain["retriever"].invoke(question)

    pages_used = []
    for doc in source_docs:
        page = doc.metadata.get("page", "unknown")
        pages_used.append(page + 1 if isinstance(page, int) else page)

    return answer, pages_used


# ─────────────────────────────────────────
# MAIN — Process new PDF
# ─────────────────────────────────────────
def process_pdf(pdf_path):
    documents = load_document(pdf_path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = build_qa_chain(vector_store)
    return qa_chain
