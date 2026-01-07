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

                    # ROBUST APPROACH: Use random vector queries to fetch documents
                    # This works for ANY content type (G.O., Mahabharata, technical docs, etc.)
                    import numpy as np
                    from langchain.schema import Document
                    from pinecone import Pinecone

                    sample_docs = []
                    seen_ids = set()

                    try:
                        # Get Pinecone index directly
                        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

                        # Fetch using 5 random vectors to get diverse samples
                        for _ in range(5):
                            # Create random vector (3072 dims for text-embedding-3-large)
                            random_vector = np.random.randn(3072).tolist()

                            # Query Pinecone with random vector
                            results = index.query(
                                vector=random_vector,
                                top_k=100,
                                include_metadata=True,
                                namespace="default"  # CRITICAL: Must specify namespace!
                            )

                            # Convert to LangChain documents
                            for match in results.get('matches', []):
                                if match['id'] not in seen_ids:
                                    seen_ids.add(match['id'])
                                    text = match.get('metadata', {}).get('text', '')
                                    if text:
                                        doc = Document(
                                            page_content=text,
                                            metadata=match.get('metadata', {})
                                        )
                                        sample_docs.append(doc)

                            if len(sample_docs) >= 500:
                                break

                        sample_docs = sample_docs[:500]
                        print(f"✓ Fetched {len(sample_docs)} documents using random vector sampling")

                    except Exception as e:
                        print(f"Warning: Could not fetch with random vectors - {str(e)[:100]}")
                        print("Attempting fallback to generic semantic queries...")

                        # Fallback: try semantic search with generic queries
                        generic_queries = ["document text", "information content", "data records"]
                        for query in generic_queries:
                            try:
                                docs = vector_store.similarity_search(query, k=200)
                                for doc in docs:
                                    doc_id = id(doc.page_content)
                                    if doc_id not in seen_ids:
                                        seen_ids.add(doc_id)
                                        sample_docs.append(doc)
                            except:
                                continue

                        sample_docs = sample_docs[:500]

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
    """Display the chat history with sources as collapsible dropdown with 'See More' functionality."""
    # Initialize session state for tracking how many chunks to show per message
    if "chunks_to_show" not in st.session_state:
        st.session_state.chunks_to_show = {}

    for idx, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(content)

                # Display sources in a collapsible expander if available
                source_docs = message.get("source_docs", [])
                if source_docs:
                    # Initialize chunks to show for this message (default: first 3)
                    msg_key = f"msg_{idx}"
                    if msg_key not in st.session_state.chunks_to_show:
                        st.session_state.chunks_to_show[msg_key] = 3

                    with st.expander(f"📚 View {len(source_docs)} Source Chunks", expanded=False):
                        st.caption(f"These are the exact document chunks used to generate this answer")
                        st.markdown("---")

                        # Get how many chunks to display
                        num_to_show = st.session_state.chunks_to_show[msg_key]

                        # Display limited chunks (prevents auto-scroll issue)
                        for i, doc in enumerate(source_docs[:num_to_show], 1):
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            chunk_length = len(doc.page_content)

                            # Header for each chunk
                            st.markdown(f"#### 📄 Source {i}")

                            # Show metadata in columns
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.caption(f"**Document:** {source_file}")
                            with col2:
                                st.caption(f"**Length:** {chunk_length:,} chars")
                            with col3:
                                st.caption(f"**Chunk {i}/{len(source_docs)}**")

                            # Show preview only (no full content to reduce length)
                            preview_length = 200
                            preview = doc.page_content[:preview_length]
                            st.text(preview + ("..." if chunk_length > preview_length else ""))

                            # Optional: Show full content in smaller text area
                            if st.checkbox(f"Show full content", key=f"show_full_{idx}_{i}"):
                                st.text_area(
                                    f"Full content of chunk {i}",
                                    value=doc.page_content,
                                    height=150,
                                    key=f"history_{idx}_chunk_{i}_{hash(doc.page_content[:50])}",
                                    label_visibility="collapsed"
                                )

                            # Divider between chunks
                            if i < num_to_show:
                                st.markdown("---")

                        # "See More" button if there are more chunks
                        if num_to_show < len(source_docs):
                            remaining = len(source_docs) - num_to_show
                            if st.button(f"📖 See {min(3, remaining)} More Chunks ({remaining} remaining)", key=f"see_more_{idx}"):
                                st.session_state.chunks_to_show[msg_key] += 3
                                st.rerun()
                        elif num_to_show >= len(source_docs) and len(source_docs) > 3:
                            # Show "Show Less" button
                            if st.button("🔼 Show Less", key=f"show_less_{idx}"):
                                st.session_state.chunks_to_show[msg_key] = 3
                                st.rerun()


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

                    # ========== TERMINAL LOGGING OF RETRIEVED CHUNKS ==========
                    print("\n" + "="*80)
                    print(f"🔍 QUERY: {user_question}")
                    print("="*80)
                    print(f"\n📄 RETRIEVED CHUNKS: {len(source_docs)} chunks")
                    print("="*80)

                    for i, doc in enumerate(source_docs, 1):
                        print(f"\n--- Chunk {i}/{len(source_docs)} ---")
                        print(f"Length: {len(doc.page_content)} chars")

                        # Show metadata
                        if doc.metadata:
                            print(f"Metadata: {doc.metadata}")

                        # Show full content (not just preview)
                        print(f"\nFull Content:")
                        print(doc.page_content)
                        print("-"*80)

                    # Show total context sent to LLM
                    context = "\n\n---\n\n".join([doc.page_content for doc in source_docs])
                    print(f"\n{'='*80}")
                    print(f"📊 CONTEXT SUMMARY")
                    print(f"{'='*80}")
                    print(f"Total context length: {len(context):,} characters")
                    print(f"Number of chunks: {len(source_docs)}")
                    print(f"Estimated tokens: ~{len(context) // 4:,}")
                    print(f"{'='*80}\n")
                    # ========== END TERMINAL LOGGING ==========

                    st.write(answer)

            # Add assistant message to chat history WITH source docs
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "source_docs": source_docs  # Store sources with the message
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
