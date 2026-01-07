"""
Complete RAG Debug Tool - Shows retrieval AND answer generation.
See exactly what chunks are retrieved and how the LLM uses them.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rag_chain import RAGChain

# Load environment
load_dotenv()

def debug_rag_query(query: str):
    """
    Debug a RAG query end-to-end.

    Args:
        query: The question to ask
    """
    print("\n" + "="*80)
    print("🔍 RAG QUERY DEBUG - COMPLETE PIPELINE")
    print("="*80)
    print(f"\n📝 Query: \"{query}\"")
    print("="*80)

    # Initialize
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Connect to Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # Get stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)

    print(f"\n📊 STEP 1: PINECONE CONNECTION")
    print(f"   • Index: {index_name}")
    print(f"   • Total vectors: {total_vectors:,}")

    # Create vector store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Fetch documents for hybrid search - ROBUST APPROACH
    print(f"\n📊 STEP 2: HYBRID SEARCH INITIALIZATION")

    # Strategy: Use dummy vector query to fetch ANY documents from index
    # This works regardless of content type (G.O., Mahabharata, etc.)
    import numpy as np

    sample_docs = []
    try:
        # Create multiple random vectors to fetch diverse documents
        print(f"   • Using random vector queries to fetch documents...")
        seen_ids = set()

        # DEBUG: First query to inspect structure
        random_vector = np.random.randn(3072).tolist()
        test_results = index.query(
            vector=random_vector,
            top_k=5,
            include_metadata=True,
            namespace="default"  # CRITICAL: Must specify namespace!
        )

        print(f"   • DEBUG: Test query returned {len(test_results.get('matches', []))} matches")
        if test_results.get('matches'):
            first_match = test_results['matches'][0]
            print(f"   • DEBUG: First match ID: {first_match.get('id', 'NO_ID')}")
            print(f"   • DEBUG: Metadata keys: {list(first_match.get('metadata', {}).keys())}")
            print(f"   • DEBUG: Has 'text' key: {'text' in first_match.get('metadata', {})}")

            # Check what the actual key is
            metadata = first_match.get('metadata', {})
            if metadata:
                print(f"   • DEBUG: Sample metadata: {str(metadata)[:200]}...")

        # Fetch using 5 random vectors to get diverse samples
        for _ in range(5):
            # Create a random vector matching the embedding dimension (3072 for text-embedding-3-large)
            random_vector = np.random.randn(3072).tolist()

            # Query Pinecone directly with random vector
            results = index.query(
                vector=random_vector,
                top_k=100,
                include_metadata=True,
                namespace="default"  # CRITICAL: Must specify namespace!
            )

            # Convert Pinecone results to LangChain documents
            for match in results.get('matches', []):
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    # Extract text from metadata - try 'text' key
                    text = match.get('metadata', {}).get('text', '')
                    if text:
                        from langchain.schema import Document
                        doc = Document(
                            page_content=text,
                            metadata=match.get('metadata', {})
                        )
                        sample_docs.append(doc)

            # Stop if we have enough documents
            if len(sample_docs) >= 500:
                break

        sample_docs = sample_docs[:500]
        print(f"   • Fetched {len(sample_docs)} documents for BM25")

    except Exception as e:
        print(f"   ⚠️  Warning: Could not fetch documents - {str(e)[:100]}")
        print(f"   • Hybrid search will be disabled, using semantic-only")
        sample_docs = []

    # Initialize RAG chain
    print(f"\n📊 STEP 3: INITIALIZING RAG CHAIN")
    rag_chain = RAGChain(
        vector_store=vector_store,
        documents=sample_docs,
        model_name="gpt-4o-mini",
        temperature=0.0,
        k=25,  # Retrieve top 25 chunks
        use_hybrid=True
    )
    print(f"   • Model: gpt-4o-mini")
    print(f"   • Temperature: 0.0")
    print(f"   • Retrieval k: 25")
    print(f"   • Hybrid search: ENABLED")

    # Ask question and get response
    print(f"\n📊 STEP 4: RETRIEVING DOCUMENTS")
    print(f"   • Searching for relevant chunks...")

    result = rag_chain.ask(query)
    answer = result.get("answer", "No answer generated")
    source_docs = result.get("source_documents", [])

    print(f"   • Retrieved {len(source_docs)} chunks")

    # Show retrieved chunks
    print(f"\n{'='*80}")
    print(f"📄 RETRIEVED CHUNKS ({len(source_docs)} total)")
    print(f"{'='*80}")

    for i, doc in enumerate(source_docs, 1):
        print(f"\n--- Chunk {i}/{len(source_docs)} ---")
        print(f"Length: {len(doc.page_content)} chars")

        # Show metadata
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")

        # Show preview
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"Preview: {preview}")
        print("-"*80)

    # Show what goes to LLM
    print(f"\n{'='*80}")
    print(f"📤 CONTEXT SENT TO LLM")
    print(f"{'='*80}")
    context = "\n\n---\n\n".join([doc.page_content for doc in source_docs])
    print(f"   • Total context length: {len(context):,} characters")
    print(f"   • Number of chunks: {len(source_docs)}")
    print(f"   • Estimated tokens: ~{len(context) // 4:,}")

    # Show answer
    print(f"\n{'='*80}")
    print(f"💬 LLM GENERATED ANSWER")
    print(f"{'='*80}")
    print(answer)
    print(f"{'='*80}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 COMPLETE SUMMARY")
    print(f"{'='*80}")
    print(f"   • Query: \"{query}\"")
    print(f"   • Retrieved chunks: {len(source_docs)}")
    print(f"   • Total context chars: {len(context):,}")
    print(f"   • Answer length: {len(answer)} chars")
    print(f"   • Retrieval method: Hybrid (BM25 + Semantic)")
    print(f"{'='*80}")

    # Offer to show full chunks
    print("\n💡 TIP: Scroll up to see all retrieved chunks and their content")

    return result


def main():
    """Main function."""
    print("\n" + "="*80)
    print("🔍 RAG COMPLETE DEBUG TOOL")
    print("="*80)
    print("\nThis tool shows you:")
    print("   1. Which chunks are retrieved")
    print("   2. What context is sent to the LLM")
    print("   3. What answer the LLM generates")
    print("="*80)

    # # Example queries
    # print("\n📝 Example queries:")
    # print("   1. What are the orders in G.O.Rt.No. 189?")
    # print("   2. Who signed G.O.Rt.No. 1447?")
    # print("   3. Who is PEEYUSH KUMAR?")
    # print("   4. List transfers in G.O.Rt.No. 1213")

    # Get user input
    print("\n" + "="*80)
    query = input("\n🔍 Enter your query (or press Enter for example): ").strip()

    if not query:
        query = "What are the orders in G.O.Rt.No. 189?"
        print(f"Using example query: \"{query}\"")

    # Debug the query
    result = debug_rag_query(query)

    # Offer to show specific chunk
    source_docs = result.get("source_documents", [])
    if source_docs:
        print("\n" + "="*80)
        view = input(f"\n👁️  View a specific chunk in full? (1-{len(source_docs)} or 'n'): ").strip()

        if view.isdigit():
            chunk_num = int(view)
            if 1 <= chunk_num <= len(source_docs):
                doc = source_docs[chunk_num - 1]
                print(f"\n{'='*80}")
                print(f"📄 CHUNK {chunk_num} - FULL CONTENT")
                print(f"{'='*80}")
                if doc.metadata:
                    print(f"\nMetadata: {doc.metadata}\n")
                print(doc.page_content)
                print(f"{'='*80}")

    print("\n✅ Debug complete!")


if __name__ == "__main__":
    main()
