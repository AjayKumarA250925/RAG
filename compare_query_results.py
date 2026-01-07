"""
Compare multiple queries to understand why RAG gives different answers.
Helps debug issues like inconsistent answers for similar questions.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rag_chain import RAGChain
import numpy as np
from langchain.schema import Document

# Load environment
load_dotenv()


def analyze_query(query: str, rag_chain: RAGChain, query_num: int):
    """
    Analyze a single query and show what chunks were retrieved.

    Args:
        query: The question to ask
        rag_chain: The RAG chain to use
        query_num: Query number for display
    """
    print("\n" + "="*80)
    print(f"📝 QUERY {query_num}: \"{query}\"")
    print("="*80)

    # Get response
    result = rag_chain.ask(query)
    answer = result.get("answer", "No answer generated")
    source_docs = result.get("source_documents", [])

    print(f"\n✅ ANSWER:")
    print("-"*80)
    print(answer)
    print("-"*80)

    print(f"\n📄 RETRIEVED {len(source_docs)} CHUNKS:")
    print("="*80)

    # Track unique content to detect duplicates
    seen_content = set()
    duplicate_count = 0

    for i, doc in enumerate(source_docs, 1):
        content_hash = hash(doc.page_content)
        is_duplicate = content_hash in seen_content
        seen_content.add(content_hash)

        if is_duplicate:
            duplicate_count += 1

        print(f"\n{'🔁 ' if is_duplicate else ''}Chunk {i}/{len(source_docs)}")
        print(f"Length: {len(doc.page_content)} chars")

        # Show metadata
        if doc.metadata:
            source_file = doc.metadata.get('source_file', 'Unknown')
            print(f"Source: {source_file}")

        # Show preview (first 300 chars)
        preview = doc.page_content[:300].replace('\n', ' ')
        if len(doc.page_content) > 300:
            preview += "..."
        print(f"Preview: {preview}")
        print("-"*40)

    if duplicate_count > 0:
        print(f"\n⚠️ WARNING: Found {duplicate_count} duplicate chunks!")

    # Show context summary
    context = "\n\n---\n\n".join([doc.page_content for doc in source_docs])
    print(f"\n📊 CONTEXT SUMMARY:")
    print(f"   • Total characters: {len(context):,}")
    print(f"   • Unique chunks: {len(seen_content)}")
    print(f"   • Duplicate chunks: {duplicate_count}")
    print(f"   • Estimated tokens: ~{len(context) // 4:,}")

    return result


def compare_queries(queries: list):
    """
    Compare multiple queries side-by-side to understand differences.

    Args:
        queries: List of query strings to compare
    """
    print("\n" + "="*80)
    print("🔍 RAG QUERY COMPARISON TOOL")
    print("="*80)
    print(f"\nComparing {len(queries)} queries to understand retrieval differences...")

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

    print(f"\n📊 INDEX INFO:")
    print(f"   • Index: {index_name}")
    print(f"   • Total vectors: {total_vectors:,}")

    # Create vector store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Fetch documents for hybrid search
    print(f"\n⏳ Fetching documents for hybrid search...")
    sample_docs = []
    seen_ids = set()

    try:
        # Fetch using random vectors
        for _ in range(5):
            random_vector = np.random.randn(3072).tolist()

            results = index.query(
                vector=random_vector,
                top_k=100,
                include_metadata=True,
                namespace="default"
            )

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
        print(f"   ✓ Fetched {len(sample_docs)} documents")

    except Exception as e:
        print(f"   ⚠️ Warning: Could not fetch documents - {str(e)[:100]}")
        sample_docs = []

    # Initialize RAG chain
    print(f"\n⏳ Initializing RAG chain...")
    rag_chain = RAGChain(
        vector_store=vector_store,
        documents=sample_docs,
        model_name="gpt-4o-mini",
        temperature=0.0,
        k=25,
        use_hybrid=True
    )
    print(f"   ✓ RAG chain ready (k={rag_chain.k}, hybrid={rag_chain.use_hybrid})")

    # Analyze each query
    results = []
    for i, query in enumerate(queries, 1):
        result = analyze_query(query, rag_chain, i)
        results.append(result)

    # Show comparison summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)

    for i, (query, result) in enumerate(zip(queries, results), 1):
        source_docs = result.get("source_documents", [])
        answer = result.get("answer", "")

        print(f"\nQuery {i}: \"{query}\"")
        print(f"   • Retrieved chunks: {len(source_docs)}")
        print(f"   • Answer length: {len(answer)} chars")
        print(f"   • Answer preview: {answer[:100]}...")

    print("\n" + "="*80)
    print("💡 ANALYSIS TIPS:")
    print("="*80)
    print("1. Compare the retrieved chunks for each query")
    print("2. Check if different chunks lead to different answers")
    print("3. Look for duplicate or irrelevant chunks")
    print("4. Verify if the correct chunks are being retrieved")
    print("5. Check if chunk overlap causes information mixing")
    print("="*80)


def main():
    """Main function."""
    print("\n" + "="*80)
    print("🔍 RAG QUERY COMPARISON TOOL")
    print("="*80)
    print("\nThis tool helps you understand why RAG gives different answers")
    print("for similar queries by comparing retrieved chunks side-by-side.")
    print("="*80)

    # Example: Compare the Abhimanyu grandparents queries
    print("\n📝 EXAMPLE: Comparing Abhimanyu grandparents queries")
    print("(You can modify this script to compare your own queries)")

    queries_to_compare = [
        "who is abhimanyu's grand parents",
        "who is abhimanyu's grand parents?",
        "who are the grandparents of Abhimanyu",
        "tell me about Abhimanyu's grandparents",
    ]

    # Allow user to input custom queries
    print("\n" + "="*80)
    print("Enter queries to compare (or press Enter to use example queries):")
    print("Type each query and press Enter. Type 'done' when finished.")
    print("="*80)

    custom_queries = []
    while True:
        query = input(f"\nQuery {len(custom_queries) + 1} (or 'done'): ").strip()
        if query.lower() == 'done':
            break
        if query:
            custom_queries.append(query)

    # Use custom queries if provided, otherwise use examples
    queries = custom_queries if custom_queries else queries_to_compare

    if not queries:
        print("\n⚠️ No queries to compare!")
        return

    print(f"\n✓ Comparing {len(queries)} queries...")

    # Run comparison
    compare_queries(queries)

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
