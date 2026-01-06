"""
Utility script to clear all vectors from Pinecone index.
Use this to start fresh with new document ingestion.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def clear_pinecone_index():
    """Clear all data from Pinecone index."""

    # Load environment variables
    load_dotenv()

    # Get Pinecone configuration
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        print("❌ Error: PINECONE_API_KEY or PINECONE_INDEX_NAME not found in .env file")
        return False

    try:
        # Initialize Pinecone
        print(f"🔧 Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)

        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"⚠️  Index '{index_name}' does not exist. Nothing to clear.")
            return True

        # Get index
        index = pc.Index(index_name)

        # Get index stats before deletion
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)

        print(f"\n📊 Current Index Stats:")
        print(f"  • Index Name: {index_name}")
        print(f"  • Total Vectors: {total_vectors:,}")
        print(f"  • Namespaces: {stats.get('namespaces', {})}")

        if total_vectors == 0:
            print("\n✅ Index is already empty. Nothing to clear.")
            return True

        # Confirm deletion
        print(f"\n⚠️  WARNING: This will DELETE all {total_vectors:,} vectors from '{index_name}'")
        response = input("   Type 'yes' to confirm: ")

        if response.lower() != 'yes':
            print("\n❌ Operation cancelled.")
            return False

        # Delete all vectors from all namespaces
        print(f"\n🗑️  Deleting all vectors from index '{index_name}'...")

        # Get all namespaces
        namespaces = stats.get('namespaces', {})

        if namespaces:
            # Delete from each namespace
            for namespace_name in namespaces.keys():
                print(f"   • Deleting from namespace: '{namespace_name}'...")
                index.delete(delete_all=True, namespace=namespace_name)
        else:
            # Try default namespace
            print(f"   • Deleting from default namespace...")
            index.delete(delete_all=True, namespace="")

        # Verify deletion
        print("⏳ Waiting for deletion to complete...")
        import time
        time.sleep(2)  # Wait for deletion to propagate

        stats_after = index.describe_index_stats()
        vectors_after = stats_after.get('total_vector_count', 0)

        if vectors_after == 0:
            print(f"\n✅ SUCCESS! All data cleared from '{index_name}'")
            print(f"   • Vectors deleted: {total_vectors:,}")
            print(f"   • Current vectors: 0")
            print(f"\n📝 You can now upload fresh documents in the Streamlit app.")
            return True
        else:
            print(f"\n⚠️  Warning: {vectors_after} vectors still remain. Try again in a few seconds.")
            return False

    except Exception as e:
        print(f"\n❌ Error clearing Pinecone index: {str(e)}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("🧹 PINECONE INDEX CLEANUP UTILITY")
    print("=" * 60)

    success = clear_pinecone_index()

    if success:
        print("\n" + "=" * 60)
        print("✅ READY FOR FRESH INGESTION")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Upload your 219 documents")
        print("3. Wait for processing to complete")
        print("4. Start querying with high accuracy!")
    else:
        print("\n" + "=" * 60)
        print("❌ CLEANUP FAILED")
        print("=" * 60)

if __name__ == "__main__":
    main()
