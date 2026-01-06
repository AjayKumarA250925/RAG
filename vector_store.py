import os
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

class PineconeVectorStoreManager:
    """Manages Pinecone vector store operations for RAG application."""

    def __init__(
        self,
        api_key: str,
        index_name: str = "rag-pinecone-docs",
        embedding_model: str = "text-embedding-3-large",
        dimension: int = 3072  # text-embedding-3-large dimension
    ):
        """
        Initialize the Pinecone vector store manager.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding_model: OpenAI embedding model name
            dimension: Embedding dimension (3072 for text-embedding-3-large)
        """
        self.api_key = api_key
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.dimension = dimension

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize or get existing index
        self._initialize_index()

    def _initialize_index(self):
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{self.index_name}' created successfully!")
        else:
            print(f"Using existing index: {self.index_name}")

        self.index = self.pc.Index(self.index_name)

    def add_documents(self, documents: List[Document], namespace: str = "default", batch_size: int = 100):
        """
        Add documents to the Pinecone vector store with BATCH PROCESSING.
        Handles large documents by processing in smaller batches to avoid token limits.

        Args:
            documents: List of LangChain Document objects
            namespace: Namespace for organizing documents
            batch_size: Number of documents to process per batch (default: 100)

        Returns:
            PineconeVectorStore instance
        """
        if not documents:
            raise ValueError("No documents provided to add to vector store")

        total_docs = len(documents)
        print(f"\n{'='*60}")
        print(f"📤 UPLOADING TO PINECONE")
        print(f"{'='*60}")
        print(f"  • Total chunks: {total_docs}")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Total batches: {(total_docs + batch_size - 1) // batch_size}")
        print(f"{'='*60}\n")

        # Process in batches to avoid token limit errors
        vector_store = None
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")

            try:
                if i == 0:
                    # First batch: create vector store
                    vector_store = PineconeVectorStore.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        index_name=self.index_name,
                        namespace=namespace
                    )
                else:
                    # Subsequent batches: add to existing vector store
                    vector_store.add_documents(batch)

                print(f"✅ Success")

            except Exception as e:
                error_msg = str(e)
                if "max_tokens_per_request" in error_msg:
                    print(f"\n❌ Token limit exceeded for batch {batch_num}!")
                    print(f"   Batch has {len(batch)} chunks - try reducing batch_size")
                    print(f"   Error: {error_msg[:100]}...")
                    raise
                else:
                    print(f"❌ Error: {error_msg[:100]}...")
                    raise

        print(f"\n{'='*60}")
        print(f"✅ UPLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"  • Total chunks uploaded: {total_docs}")
        print(f"  • Batches processed: {total_batches}")
        print(f"{'='*60}\n")

        return vector_store

    def get_vector_store(self, namespace: str = "default"):
        """
        Get an existing vector store instance.

        Args:
            namespace: Namespace to retrieve documents from

        Returns:
            PineconeVectorStore instance
        """
        vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )
        return vector_store

    def delete_index(self):
        """Delete the Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully!")
        except Exception as e:
            print(f"Error deleting index: {str(e)}")

    def get_index_stats(self):
        """
        Get statistics about the index.

        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting index stats: {str(e)}")
            return None

    def clear_namespace(self, namespace: str = "default"):
        """
        Clear all vectors in a specific namespace.

        Args:
            namespace: Namespace to clear
        """
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            print(f"Cleared namespace '{namespace}' successfully!")
        except Exception as e:
            print(f"Error clearing namespace: {str(e)}")
