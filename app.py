# app.py

import streamlit as st
import os
import tempfile
from rag_engine import process_pdf, load_vector_store, build_qa_chain

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG - Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A System")
st.caption("Upload a PDF and ask questions about it")

# ─────────────────────────────────────────
# SESSION STATE
# (Keeps QA chain alive between questions)
# ─────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────
# SIDEBAR — PDF Upload
# ─────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process Document", type="primary"):
            
            with st.spinner("Processing your document..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process the PDF
                st.session_state.qa_chain = process_pdf(tmp_path)
                
                # Clear old chat
                st.session_state.chat_history = []
                
                # Clean up temp file
                os.unlink(tmp_path)
            
            st.success("✅ Document processed! Ask anything.")
    
    # Load existing vector store
    st.divider()
    if st.button("📂 Load Previous Document"):
        if os.path.exists("chroma_db"):
            with st.spinner("Loading..."):
                vector_store = load_vector_store()
                st.session_state.qa_chain = build_qa_chain(
                    vector_store
                )
            st.success("✅ Previous document loaded!")
        else:
            st.error("No previous document found.")

# ─────────────────────────────────────────
# MAIN AREA — Chat Interface
# ─────────────────────────────────────────
if st.session_state.qa_chain is None:
    # Show instructions if no document loaded
    st.info("👈 Upload a PDF from the sidebar to get started")
    
    st.markdown("""
    ### How it works:
    1. 📤 Upload any PDF document
    2. ⚙️ Click **Process Document**
    3. ❓ Ask questions about it
    4. 🤖 Get instant AI answers
    
    ### Good PDFs to try:
    - Medical reports
    - Research papers
    - Legal contracts
    - Company reports
    - Textbooks
    """)

else:
    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            
            if chat["pages"]:
                st.caption(
                    f"📖 Source pages: {chat['pages']}"
                )
    
    # Question input
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Show user question
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from rag_engine import ask_question
                answer, pages = ask_question(
                    st.session_state.qa_chain,
                    question
                )
            
            st.write(answer)
            
            if pages:
                st.caption(f"📖 Source pages: {pages}")
        
        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "pages": pages
        })