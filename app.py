"""
Streamlit UI for RAG Application with LangChain & Pinecone.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import PineconeVectorStoreManager
from rag_chain import RAGChain
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Application with LangChain & Pinecone",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None


def initialize_app():
    """Initialize the application with API keys and configurations."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-pinecone-docs")

    if not openai_api_key or not pinecone_api_key:
        st.error("⚠️ Please set OPENAI_API_KEY and PINECONE_API_KEY in your .env file")
        st.stop()

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize vector store manager
    if st.session_state.vector_store_manager is None:
        try:
            st.session_state.vector_store_manager = PineconeVectorStoreManager(
                api_key=pinecone_api_key,
                index_name=index_name,
                embedding_model="text-embedding-3-large"
            )

            # Check if documents already exist in the index
            stats = st.session_state.vector_store_manager.get_index_stats()
            if stats and stats.get('total_vector_count', 0) > 0:
                # Documents exist, retrieve ALL documents for hybrid search
                vector_store = st.session_state.vector_store_manager.get_vector_store()

                # Fetch documents from vector store for hybrid retrieval
                try:
                    # Retrieve documents from vector store to enable BM25
                    print("Fetching documents from Pinecone for hybrid search...")

                    # Use a broad query to fetch multiple documents for BM25 indexing
                    # Fetch MANY documents to ensure better coverage for 219 documents
                    # Using k=500 to ensure we get sufficient coverage for BM25
                    sample_docs = vector_store.similarity_search("government order transfer posting", k=500)

                    if sample_docs and len(sample_docs) > 0:
                        print(f"Retrieved {len(sample_docs)} documents for hybrid search initialization")
                        st.session_state.document_chunks = sample_docs

                        st.session_state.rag_chain = RAGChain(
                            vector_store=vector_store,
                            documents=sample_docs,  # Pass fetched documents for hybrid retrieval
                            model_name="gpt-4o-mini",
                            temperature=0.0,
                            k=25,  # Increased to 25 to ensure complete G.O. retrieval (each G.O. ~2-3 chunks)
                            use_hybrid=True  # ENABLE hybrid search!
                        )
                        print("✓ Hybrid search enabled with fetched documents")
                    else:
                        # Fallback to semantic only if fetch fails
                        st.session_state.rag_chain = RAGChain(
                            vector_store=vector_store,
                            documents=None,
                            model_name="gpt-4o-mini",
                            temperature=0.0,
                            k=25,
                            use_hybrid=False
                        )
                        print("⚠ Semantic-only search (document fetch returned empty)")

                    st.session_state.documents_processed = True
                except Exception as e:
                    print(f"Note: Error fetching documents. Using semantic-only search. ({str(e)})")
                    st.session_state.rag_chain = RAGChain(
                        vector_store=vector_store,
                        documents=None,
                        model_name="gpt-4o-mini",
                        temperature=0.0,
                        k=15,
                        use_hybrid=False
                    )
                    st.session_state.documents_processed = True

        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the vector store."""
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return False

    try:
        with st.spinner("Processing documents..."):
            # Create temporary directory to save uploaded files
            temp_dir = tempfile.mkdtemp()
            file_paths = []

            # Save uploaded files to temporary directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            # Process documents with optimized settings for short documents
            doc_processor = DocumentProcessor(chunk_size=1500, chunk_overlap=300)
            chunks = doc_processor.process_documents(file_paths)

            if not chunks:
                st.error("No text could be extracted from the uploaded documents.")
                return False

            # Add documents to vector store
            vector_store = st.session_state.vector_store_manager.add_documents(chunks)

            # Store chunks for hybrid retrieval
            st.session_state.document_chunks = chunks

            # Initialize RAG chain with hybrid retrieval for accurate name/date/ID matching
            st.session_state.rag_chain = RAGChain(
                vector_store=vector_store,
                documents=chunks,  # Pass documents for hybrid retrieval
                model_name="gpt-4o-mini",
                temperature=0.0,
                k=15,  # Increased for better coverage
                use_hybrid=True  # Enable hybrid search for exact matches
            )

            st.session_state.documents_processed = True

            # Display success with retrieval mode info
            retrieval_mode = "Hybrid (Semantic + Keyword)" if st.session_state.rag_chain.use_hybrid else "Semantic Only"
            st.success(f"✅ Successfully processed {len(chunks)} document chunks!\n\n🔍 Retrieval Mode: **{retrieval_mode}**")

            # Clean up temporary files
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass

            return True

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False


def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []


def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(content)


def main():
    """Main application function."""
    # Initialize app
    initialize_app()

    # Header
    st.title("📚 RAG Application with LangChain & Pinecone")
    st.markdown("Upload your documents and ask questions to get AI-powered answers based on your content.")

    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Documents")
        st.info("📄 Supported formats: PDF, TXT, DOCX\n\n📦 Limit: 200MB per file")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Drag and drop files here"
        )

        if st.button("Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                process_uploaded_files(uploaded_files)
            else:
                st.warning("Please upload at least one document.")

        st.divider()

        if st.button("Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.rerun()

        # Display index stats
        if st.session_state.vector_store_manager:
            st.divider()
            st.subheader("Index Statistics")
            with st.spinner("Loading stats..."):
                stats = st.session_state.vector_store_manager.get_index_stats()
                if stats:
                    st.metric("Total Vectors", stats.get('total_vector_count', 0))
                    st.metric("Index Name", st.session_state.vector_store_manager.index_name)

    # Main chat interface
    if st.session_state.documents_processed:
        # Check if using hybrid or semantic only mode
        retrieval_mode = "Hybrid (Semantic + Keyword)" if (st.session_state.rag_chain and st.session_state.rag_chain.use_hybrid) else "Semantic Only"
        st.success(f"✅ Documents are ready! Ask your questions below.\n\n🔍 Retrieval Mode: **{retrieval_mode}**")

        # Display chat history
        display_chat_history()

        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")

        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })

            # Display user message
            with st.chat_message("user", avatar="🧑"):
                st.write(user_question)

            # Get response from RAG chain
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_chain.ask(user_question)
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])

                    st.write(answer)

                    # Display source documents if available
                    if source_docs:
                        with st.expander(f"📄 View Source Documents ({len(source_docs)} found)"):
                            for i, doc in enumerate(source_docs, 1):
                                source_file = doc.metadata.get('source_file', 'Unknown')
                                st.markdown(f"**Source {i}:** `{source_file}`")
                                st.text(doc.page_content[:400] + "...")
                                st.divider()

                    # Show retrieval mode info
                    retrieval_info = "🔍 **Retrieval Mode:** "
                    if st.session_state.rag_chain and st.session_state.rag_chain.use_hybrid:
                        retrieval_info += "Hybrid (Semantic + Keyword BM25) ✓"
                    else:
                        retrieval_info += "Semantic Only (Consider uploading documents for hybrid search)"
                    with st.expander("ℹ️ Retrieval Info"):
                        st.info(retrieval_info)
                        st.markdown(f"**Documents Retrieved:** {len(source_docs) if source_docs else 0}")
                        st.markdown(f"**k value:** {st.session_state.rag_chain.k if st.session_state.rag_chain else 'N/A'}")

            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

    else:
        # Welcome message
        st.info("👈 Please upload your documents using the sidebar to get started.")

        # Display example questions
        st.subheader("How it works:")
        st.markdown("""
        1. **Upload Documents**: Use the sidebar to upload PDF, TXT, or DOCX files
        2. **Process Documents**: Click the 'Process Documents' button to index your files
        3. **Ask Questions**: Type your questions in the chat input below
        4. **Get Answers**: Receive AI-powered answers based on your document content
        """)

        st.divider()

        st.subheader("Example Questions:")
        st.markdown("""
        - What is the main topic of the documents?
        - Can you summarize the key points?
        - What information do you have about [specific topic]?
        - Explain [concept] mentioned in the documents
        """)


if __name__ == "__main__":
    main()
