"""
Specific investigation tool for the Abhimanyu grandparents query confusion.
This will help identify why different answers appear for the same question.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rag_chain import RAGChain
import numpy as np
from langchain.schema import Document

load_dotenv()


def search_for_abhimanyu_info():
    """Search for all chunks containing 'Abhimanyu' in the vector store."""
    print("\n" + "="*80)
    print("🔍 SEARCHING FOR ABHIMANYU INFORMATION IN DATABASE")
    print("="*80)

    # Initialize
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Search for Abhimanyu
    print("\n⏳ Searching for 'Abhimanyu'...")
    abhimanyu_docs = vector_store.similarity_search("Abhimanyu", k=50)

    print(f"\n✓ Found {len(abhimanyu_docs)} chunks mentioning or related to Abhimanyu")

    # Filter to chunks that actually contain "Abhimanyu"
    direct_mentions = []
    for doc in abhimanyu_docs:
        if "abhimanyu" in doc.page_content.lower():
            direct_mentions.append(doc)

    print(f"✓ {len(direct_mentions)} chunks directly mention 'Abhimanyu'")

    # Show each chunk
    print("\n" + "="*80)
    print("📄 CHUNKS MENTIONING ABHIMANYU:")
    print("="*80)

    for i, doc in enumerate(direct_mentions, 1):
        print(f"\n{'='*80}")
        print(f"CHUNK {i}/{len(direct_mentions)}")
        print('='*80)

        # Show metadata
        if doc.metadata:
            source = doc.metadata.get('source_file', 'Unknown')
            print(f"Source: {source}")

        # Show full content
        print(f"\nFull Content ({len(doc.page_content)} chars):")
        print("-"*80)
        print(doc.page_content)
        print("-"*80)

        # Analyze grandparent mentions
        content_lower = doc.page_content.lower()
        grandparent_terms = ['grandfather', 'grandmother', 'grandparent', 'pandu', 'kunti',
                           'vasudeva', 'devaki', 'rohini']

        found_terms = [term for term in grandparent_terms if term in content_lower]
        if found_terms:
            print(f"\n🔍 Grandparent-related terms found: {', '.join(found_terms)}")

    return direct_mentions


def test_query_variations():
    """Test different query variations for Abhimanyu's grandparents."""
    print("\n" + "="*80)
    print("🔍 TESTING QUERY VARIATIONS")
    print("="*80)

    # Initialize
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Fetch documents for hybrid search
    print(f"\n⏳ Fetching documents for hybrid search...")
    sample_docs = []
    seen_ids = set()

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

    # Initialize RAG chain
    rag_chain = RAGChain(
        vector_store=vector_store,
        documents=sample_docs,
        model_name="gpt-4o-mini",
        temperature=0.0,
        k=25,
        use_hybrid=True
    )

    # Test different query variations
    queries = [
        "who is abhimanyu's grand parents",
        "who is abhimanyu's grand parents?",
        "who are the grandparents of Abhimanyu?",
        "tell me about Abhimanyu's grandfather and grandmother",
        "Abhimanyu grandparents Mahabharata",
    ]

    results = []
    for i, query in enumerate(queries, 1):
        print("\n" + "="*80)
        print(f"QUERY {i}: \"{query}\"")
        print("="*80)

        result = rag_chain.ask(query)
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        print(f"\n✅ ANSWER:")
        print("-"*80)
        print(answer)
        print("-"*80)

        print(f"\n📄 Retrieved {len(source_docs)} chunks")

        # Check which chunks mention key names
        key_names = ['pandu', 'kunti', 'vasudeva', 'devaki', 'rohini', 'arjuna', 'subhadra']
        name_counts = {name: 0 for name in key_names}

        for doc in source_docs:
            content_lower = doc.page_content.lower()
            for name in key_names:
                if name in content_lower:
                    name_counts[name] += 1

        print(f"\n🔍 Key names mentioned in retrieved chunks:")
        for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"   • {name.capitalize()}: {count} chunks")

        results.append({
            'query': query,
            'answer': answer,
            'source_docs': source_docs,
            'name_counts': name_counts
        })

    # Summary comparison
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: \"{result['query']}\"")
        answer_preview = result['answer'][:150].replace('\n', ' ')
        print(f"Answer: {answer_preview}...")

        # Check for key names in answer
        answer_lower = result['answer'].lower()
        names_in_answer = [name for name in key_names if name in answer_lower]
        if names_in_answer:
            print(f"Names in answer: {', '.join([n.capitalize() for n in names_in_answer])}")

    return results


def diagnose_confusion():
    """Main diagnostic function."""
    print("\n" + "="*80)
    print("🔬 ABHIMANYU GRANDPARENTS QUERY DIAGNOSTIC")
    print("="*80)
    print("\nThis tool will help identify why you're getting different answers")
    print("for the same question about Abhimanyu's grandparents.")
    print("="*80)

    # Step 1: Search for all Abhimanyu chunks
    print("\n\n🔹 STEP 1: Finding all chunks about Abhimanyu")
    abhimanyu_chunks = search_for_abhimanyu_info()

    # Step 2: Test query variations
    print("\n\n🔹 STEP 2: Testing different query variations")
    query_results = test_query_variations()

    # Step 3: Analysis and recommendations
    print("\n" + "="*80)
    print("💡 DIAGNOSTIC FINDINGS & RECOMMENDATIONS")
    print("="*80)

    print("\n1. DATABASE CONTENT:")
    print(f"   • Found {len(abhimanyu_chunks)} chunks mentioning Abhimanyu")
    print("   • Check if these chunks contain correct grandparent information")

    print("\n2. QUERY VARIATIONS:")
    print(f"   • Tested {len(query_results)} different query variations")
    print("   • Compare if different queries retrieve different chunks")

    print("\n3. POSSIBLE CAUSES OF CONFUSION:")
    print("   • Different chunks may contain conflicting information")
    print("   • Retrieval may be pulling wrong or partial chunks")
    print("   • LLM may be mixing information from different chunks")
    print("   • Temperature=0.0 should give consistent answers, but chunks may vary")

    print("\n4. RECOMMENDED ACTIONS:")
    print("   • Review the chunks above to verify correct information")
    print("   • Check if database has conflicting or incorrect source data")
    print("   • Consider improving chunk overlap or size for better context")
    print("   • Ensure source documents (Mahabharata text) are accurate")

    print("\n" + "="*80)
    print("✅ Diagnostic complete! Review the output above to identify the issue.")
    print("="*80)


if __name__ == "__main__":
    diagnose_confusion()
