"""
Debug script to check which G.O. numbers are in Pinecone.
"""

import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment
load_dotenv()

def check_go_numbers():
    """Check which G.O. numbers exist in Pinecone."""

    print("="*60)
    print("🔍 CHECKING G.O. NUMBERS IN PINECONE")
    print("="*60)

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

    print(f"\n📊 Index Statistics:")
    print(f"   Total Vectors: {total_vectors:,}")

    # Create vector store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Search for G.O. numbers
    print(f"\n🔍 Searching for G.O. numbers...")

    # Fetch many documents
    results = vector_store.similarity_search("G.O.Rt.No government order", k=100)

    # Extract G.O. numbers
    go_numbers = set()
    go_pattern = re.compile(r'G\.?\s*O\.?\s*R[tT]\.?\s*N[oO]\.?\s*(\d+)', re.IGNORECASE)

    for doc in results:
        matches = go_pattern.findall(doc.page_content)
        for match in matches:
            go_numbers.add(match)

    # Sort numerically
    go_numbers_sorted = sorted(go_numbers, key=lambda x: int(x))

    print(f"\n✅ Found {len(go_numbers_sorted)} unique G.O. numbers:")
    print()

    # Display in columns
    for i, go_num in enumerate(go_numbers_sorted):
        print(f"   G.O. {go_num:>4}", end="")
        if (i + 1) % 5 == 0:
            print()  # New line every 5 items

    print()  # Final newline

    # Show sample documents to understand what's in Pinecone
    print()
    print("="*60)
    print("📋 SAMPLE DOCUMENTS IN PINECONE:")
    print("="*60)

    if results:
        print(f"\nShowing first 3 documents:\n")
        for i, doc in enumerate(results[:3], 1):
            print(f"\n--- Document {i} ---")
            print(f"Content length: {len(doc.page_content)} chars")
            print(f"First 300 chars:")
            print(doc.page_content[:300])
            print(f"\nMetadata: {doc.metadata}")
            print("-" * 40)

    # Check for G.O. 256 specifically
    print()
    print("="*60)
    if "256" in go_numbers:
        print("✅ G.O. 256 FOUND in Pinecone!")

        # Get chunks for G.O. 256
        go_256_results = vector_store.similarity_search("G.O.Rt.No. 256", k=10)

        print(f"\n📄 G.O. 256 has {len([r for r in go_256_results if '256' in r.page_content])} chunks")

        # Show first chunk
        for doc in go_256_results:
            if '256' in doc.page_content:
                print(f"\n📋 Sample content:")
                print(doc.page_content[:500])
                break
    else:
        print("❌ G.O. 256 NOT FOUND in Pinecone")
        print()
        print("Possible reasons:")
        print("   1. G.O. 256 was not uploaded")
        print("   2. Document processing failed for G.O. 256")
        print("   3. OCR failed to extract text from scanned PDF")
        print("   4. Document is labeled differently (check filename)")
        print()
        print("Solution:")
        print("   - Check if G.O. 256 PDF exists in your files")
        print("   - Re-upload G.O. 256 specifically")
        print("   - Verify it's not a corrupted or empty PDF")

    print("="*60)

if __name__ == "__main__":
    check_go_numbers()
