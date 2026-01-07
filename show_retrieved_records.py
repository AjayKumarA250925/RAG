"""
Show all retrieved records/chunks for a query.
This helps you see exactly what the RAG system fetches before generating an answer.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from hybrid_retriever import AdaptiveHybridRetriever

# Load environment
load_dotenv()

def show_retrieved_records(query: str, k: int = 25):
    """
    Show all records retrieved for a given query.

    Args:
        query: The question to search for
        k: Number of top chunks to retrieve (default: 25)
    """
    print("="*80)
    print(f"🔍 SHOWING RETRIEVED RECORDS FOR QUERY")
    print("="*80)
    print(f"\n📝 Query: \"{query}\"")
    print(f"📊 Retrieving top {k} chunks")
    print("="*80)

    # Initialize
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Connect to Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    print(f"\n📊 Pinecone Index Stats:")
    print(f"   • Total vectors in index: {total_vectors:,}")

    # Create vector store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Fetch documents for hybrid search - ROBUST APPROACH
    print(f"\n🔄 Fetching documents for hybrid search...")

    # Strategy: Use random vector queries to fetch ANY documents from index
    # This works regardless of content type (G.O., Mahabharata, etc.)
    import numpy as np

    sample_docs = []
    try:
        print(f"   • Using random vector queries to fetch documents...")
        seen_ids = set()

        # Fetch using 5 random vectors to get diverse samples
        for _ in range(5):
            # Create random vector (3072 dims for text-embedding-3-large)
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
                    text = match.get('metadata', {}).get('text', '')
                    if text:
                        from langchain.schema import Document
                        doc = Document(
                            page_content=text,
                            metadata=match.get('metadata', {})
                        )
                        sample_docs.append(doc)

            if len(sample_docs) >= 500:
                break

        sample_docs = sample_docs[:500]
        print(f"   • Retrieved {len(sample_docs)} documents for BM25 indexing")

    except Exception as e:
        print(f"   ⚠️  Warning: Could not fetch documents - {str(e)[:100]}")
        print(f"   • Hybrid search will be disabled, using semantic-only")
        sample_docs = []

    # Initialize hybrid retriever
    print(f"\n🔧 Initializing Hybrid Retriever (BM25 + Semantic)...")
    hybrid_retriever = AdaptiveHybridRetriever(
        vector_store=vector_store,
        documents=sample_docs,
        k=k
    )

    # Get retriever
    retriever = hybrid_retriever.get_retriever()

    # Retrieve documents
    print(f"\n🔍 Retrieving documents for query...")
    retrieved_docs = retriever.invoke(query)

    print(f"\n✅ Retrieved {len(retrieved_docs)} chunks")
    print("="*80)

    # Display each retrieved chunk
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n{'='*80}")
        print(f"📄 CHUNK {i}/{len(retrieved_docs)}")
        print(f"{'='*80}")

        # Show metadata
        print(f"\n📋 Metadata:")
        if doc.metadata:
            for key, value in doc.metadata.items():
                print(f"   • {key}: {value}")
        else:
            print("   • No metadata")

        # Show content
        print(f"\n📝 Content ({len(doc.page_content)} chars):")
        print("-"*80)
        print(doc.page_content)
        print("-"*80)

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 RETRIEVAL SUMMARY")
    print(f"{'='*80}")
    print(f"   • Query: \"{query}\"")
    print(f"   • Total chunks retrieved: {len(retrieved_docs)}")
    print(f"   • Retrieval method: Hybrid (BM25 + Semantic)")
    print(f"   • Total characters: {sum(len(doc.page_content) for doc in retrieved_docs):,}")
    print(f"   • Average chunk size: {sum(len(doc.page_content) for doc in retrieved_docs) // len(retrieved_docs) if retrieved_docs else 0} chars")
    print(f"{'='*80}")

    return retrieved_docs


def main():
    """Main function."""
    print("\n" + "="*80)
    print("🔍 RAG RETRIEVAL INSPECTOR")
    print("="*80)
    print("\nThis tool shows you ALL chunks retrieved for your query")
    print("before the LLM generates an answer.")
    print("="*80)

    # Example queries
    print("\n📝 Example queries:")
    print("   1. What are the orders in G.O.Rt.No. 189?")
    print("   2. Who signed G.O.Rt.No. 1447?")
    print("   3. Who is PEEYUSH KUMAR?")
    print("   4. List transfers in G.O.Rt.No. 1213")

    # Get user input
    print("\n" + "="*80)
    query = input("\n🔍 Enter your query (or press Enter for example): ").strip()

    if not query:
        query = "What are the orders in G.O.Rt.No. 189?"
        print(f"Using example query: \"{query}\"")

    # Get k value
    k_input = input("\n📊 Number of chunks to retrieve (default 25, press Enter to use default): ").strip()
    k = int(k_input) if k_input.isdigit() else 25

    # Show retrieved records
    retrieved_docs = show_retrieved_records(query, k)

    # Offer to show specific chunk
    print("\n" + "="*80)
    view_specific = input("\n👁️  View a specific chunk in detail? (Enter chunk number or 'n' to exit): ").strip()

    if view_specific.isdigit():
        chunk_num = int(view_specific)
        if 1 <= chunk_num <= len(retrieved_docs):
            doc = retrieved_docs[chunk_num - 1]
            print(f"\n{'='*80}")
            print(f"📄 DETAILED VIEW - CHUNK {chunk_num}")
            print(f"{'='*80}")
            print(f"\n📋 Metadata:")
            if doc.metadata:
                for key, value in doc.metadata.items():
                    print(f"   • {key}: {value}")
            print(f"\n📝 Full Content:")
            print("-"*80)
            print(doc.page_content)
            print("-"*80)
            print(f"\n📊 Stats:")
            print(f"   • Characters: {len(doc.page_content)}")
            print(f"   • Words: ~{len(doc.page_content.split())}")
            print(f"   • Lines: {len(doc.page_content.splitlines())}")

    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == "__main__":
    main()
