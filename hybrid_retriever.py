"""
Hybrid retriever combining semantic search (Pinecone) with keyword search (BM25).
This ensures accurate retrieval of exact matches (names, dates, IDs) and contextual matches.
"""

from typing import List
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore


class HybridRetriever:
    """
    Combines semantic (vector) search with keyword (BM25) search for optimal retrieval.

    Benefits:
    - Semantic search: Finds contextually similar content
    - BM25 search: Finds exact matches for names, dates, IDs, numbers
    - Weighted combination: Best of both worlds
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        documents: List[Document],
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        k: int = 4
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Pinecone vector store for semantic search
            documents: All documents for BM25 indexing
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.documents = documents
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.k = k

        # Initialize retrievers
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize both semantic and keyword retrievers."""
        # Semantic retriever (Pinecone)
        self.semantic_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

        # Keyword retriever (BM25)
        self.keyword_retriever = BM25Retriever.from_documents(
            documents=self.documents
        )
        self.keyword_retriever.k = self.k

        # Ensemble retriever combining both
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.keyword_retriever],
            weights=[self.semantic_weight, self.keyword_weight]
        )

    def get_retriever(self):
        """
        Get the ensemble retriever.

        Returns:
            EnsembleRetriever instance
        """
        return self.ensemble_retriever

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Search using hybrid retrieval.

        Args:
            query: Search query
            k: Number of documents to retrieve (optional)

        Returns:
            List of relevant documents
        """
        if k is not None:
            self.keyword_retriever.k = k
            self.semantic_retriever.search_kwargs["k"] = k

        results = self.ensemble_retriever.get_relevant_documents(query)
        return results

    def update_weights(self, semantic_weight: float, keyword_weight: float):
        """
        Update the weights for semantic and keyword search.

        Args:
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self._initialize_retrievers()


class AdaptiveHybridRetriever(HybridRetriever):
    """
    Advanced hybrid retriever that adapts weights based on query type.

    - If query contains specific identifiers (dates, IDs, numbers), favor keyword search
    - If query is conceptual, favor semantic search
    """

    def __init__(self, vector_store: PineconeVectorStore, documents: List[Document], k: int = 4):
        super().__init__(vector_store, documents, k=k)

    def _detect_query_type(self, query: str) -> tuple[float, float]:
        """
        Detect query type and return optimal weights.

        Returns:
            Tuple of (semantic_weight, keyword_weight)
        """
        import re

        # Patterns for exact match queries
        has_date = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}', query))

        # Enhanced G.O. number detection - handles multiple formats:
        # G.O.Rt.No. 1447, G.O.RT.No. 338, G O Rt No 506, GO.RT.NO.1213
        has_id = bool(
            re.search(r'G\.?\s*O\.?\s*R[tT]\.?\s*N[oO]\.?\s*\d+', query, re.IGNORECASE) or  # All G.O. variations
            re.search(r'\b[A-Z]\.[A-Z]\.([A-Z]+\.)*[A-Z]+\.\s*\d+\b', query)  # Generic format
        )

        has_number = bool(re.search(r'\b\d{3,}\b', query))

        # Enhanced name patterns to handle:
        # - "John Smith" (standard format)
        # - "S. SURESH KUMAR" (initial + uppercase names)
        # - "S.SURESH KUMAR" (no space after initial)
        # - "Dr. John Smith", "Mr. Smith" (titles)
        # - Multiple initials: "S.S. KUMAR"
        has_name_pattern = bool(
            re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query) or  # John Smith
            re.search(r'\b[A-Z]\.?\s*[A-Z]+\s+[A-Z]+\b', query) or  # S. SURESH KUMAR, S.SURESH KUMAR
            re.search(r'\b([A-Z]\.?\s*)+[A-Z]+\b', query) or  # S.S. KUMAR
            re.search(r'\b(Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+', query, re.IGNORECASE)  # Dr. Smith
        )

        # Count exact match indicators
        exact_indicators = sum([has_date, has_id, has_number, has_name_pattern])

        if exact_indicators >= 2:
            # Strong preference for keyword search
            return (0.2, 0.8)
        elif exact_indicators == 1:
            # Moderate preference for keyword search
            return (0.3, 0.7)
        else:
            # Balanced or slight semantic preference
            return (0.6, 0.4)

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Adaptive search that adjusts weights based on query type.

        Args:
            query: Search query
            k: Number of documents to retrieve (optional)

        Returns:
            List of relevant documents
        """
        # Detect optimal weights for this query
        semantic_weight, keyword_weight = self._detect_query_type(query)

        # Update weights temporarily
        original_weights = (self.semantic_weight, self.keyword_weight)
        self.update_weights(semantic_weight, keyword_weight)

        # Perform search
        if k is not None:
            self.keyword_retriever.k = k
            self.semantic_retriever.search_kwargs["k"] = k

        results = self.ensemble_retriever.get_relevant_documents(query)

        # Restore original weights
        self.update_weights(*original_weights)

        return results
