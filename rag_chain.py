"""
RAG chain module for question answering using LangChain and OpenAI.
"""

from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document


class RAGChain:
    """Manages the RAG chain for question answering."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        documents: Optional[List[Document]] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        k: int = 4,
        use_hybrid: bool = True,
        fetch_all_for_hybrid: bool = False
    ):
        """
        Initialize the RAG chain.

        Args:
            vector_store: Pinecone vector store instance
            documents: All documents for hybrid retrieval (optional)
            model_name: OpenAI model name
            temperature: Model temperature for responses
            k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid retrieval (semantic + keyword)
            fetch_all_for_hybrid: Fetch documents from vector store for hybrid search
        """
        self.vector_store = vector_store
        self.documents = documents
        self.k = k
        self.use_hybrid = use_hybrid
        self.fetch_all_for_hybrid = fetch_all_for_hybrid

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

        # Create enhanced prompt template for better accuracy
        self.prompt_template = """You are a precise government document analyst. Use the following context to answer the question accurately.

CRITICAL INSTRUCTIONS:
1. THOROUGH SEARCH: Carefully examine ALL provided context chunks. Information may be scattered across multiple sections.

2. EXACT MATCHING: When searching for specific items, look for variations:
   - Names: "S. SURESH KUMAR", "S.SURESH KUMAR", "SURESH KUMAR", "Suresh Kumar", "B.SAM BOB", "B. SAM BOB"
   - Government Orders: "G.O.Rt.No.", "G.O.RT.No.", "GO Rt No", "G.O. Rt. No."
   - Dates: Different formats (DD/MM/YYYY, DD-MM-YYYY, Month DD, YYYY)
   - Numbers: With/without commas, different spacing

3. **ABSOLUTE RULE - NO MIXING OR HALLUCINATION**:
   **YOU MUST NEVER:**
   - Mix information from different G.O. numbers
   - Invent or guess signatures, names, or dates not in the context
   - Use patterns from one G.O. to fill gaps in another G.O.
   - Combine information from different documents

   **YOU MUST ALWAYS:**
   - Each G.O. number is a SEPARATE document - treat it independently
   - If signature is not in retrieved context, say "Signature information not available in retrieved context"
   - If information is incomplete, acknowledge it rather than filling gaps
   - Only combine information from chunks of the SAME G.O. number

   **EXAMPLE OF WRONG BEHAVIOR:**
   - Context has G.O. 189 without signature, and G.O. 200 signed by "DR.K.S.JAWAHAR REDDY"
   - Question: "Who signed G.O. 189?"
   - WRONG: "DR.K.S.JAWAHAR REDDY" (this is hallucination - using G.O. 200's signature for G.O. 189)
   - CORRECT: "The signature is not available in the retrieved context for G.O. 189"

4. SPECIFIC G.O. NUMBER QUERIES: **CRITICAL** - When asked about a SPECIFIC G.O. number:
   - Find ONLY that exact G.O. number in the context
   - Extract ONLY information that appears WITHIN that G.O. number's text
   - DO NOT include information from other G.O. numbers
   - If the specific G.O. number is not found, say so clearly
   - If parts are missing, explicitly state what's missing
   - Example: If asked about "G.O.Rt.No. 1447", ONLY extract info from G.O.Rt.No. 1447, NOT from 1566, 338, or any others

5. QUERIES ABOUT SPECIFIC PEOPLE: When asked about orders someone issued/signed:
   - Find documents WHERE THEIR NAME APPEARS in the signature/issuer section
   - DO NOT attribute orders to people who don't appear in that document
   - If Person X signed G.O. 100 but not G.O. 200, only mention G.O. 100

6. COMPREHENSIVE EXTRACTION: When you find relevant information:
   - Extract ALL related details (names, dates, positions, departments, order numbers)
   - Quote exact text where possible
   - Provide complete context around the information
   - Verify all details are from the SAME source document

7. CASE SENSITIVITY: Treat uppercase and lowercase as equivalent for matching
   - "KUMAR" = "Kumar" = "kumar"
   - "BOB" = "Bob" = "bob"

8. PARTIAL MATCHES: If exact match not found, look for partial matches:
   - "S. SURESH KUMAR" might appear as just "SURESH KUMAR" or "S.SURESH"
   - "B.SAM BOB" might appear as "B. SAM BOB" or "SAM BOB"

9. ORDERS & ACTIONS: When asked about orders someone passed or actions they took:
   - Look for their name followed by words like: "order", "issued", "passed", "directed", "sanctioned"
   - Check signatures at the end of orders (e.g., "BY ORDER AND IN THE NAME OF... [Name]")
   - Look for G.O. numbers, dates, and subjects of orders
   - ONLY report orders WHERE THEIR NAME ACTUALLY APPEARS

10. CERTAINTY: Only say information is not available if you've thoroughly checked ALL context and found NOTHING related.

11. **NAME VARIATIONS - CONSISTENCY REQUIRED**:
   - Same person may have multiple names: "Balarama" = "Baladeva", "Krishna" = "Vasudeva"
   - When answering, use the name that appears MOST in the context
   - If context mentions multiple variations, acknowledge them: "Balarama (also known as Baladeva)"
   - CRITICAL: Give consistent answer regardless of which name variation user asks about

12. **FAMILY RELATIONSHIPS - NO INCONSISTENCY**:
   - When asked about someone's grandparents/parents/children multiple times
   - YOU MUST give the EXACT SAME ANSWER every time
   - Check ALL chunks for family relationships before answering
   - If you find conflicting information in chunks, state: "The sources contain conflicting information..."
   - NEVER give different family members in different responses for the same person

Context:
{context}

Question: {question}

Detailed Answer:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        # Initialize hybrid retriever if documents provided
        self.hybrid_retriever = None
        if self.use_hybrid and self.documents:
            try:
                from hybrid_retriever import AdaptiveHybridRetriever
                self.hybrid_retriever = AdaptiveHybridRetriever(
                    vector_store=self.vector_store,
                    documents=self.documents,
                    k=self.k
                )
            except ImportError:
                print("Warning: hybrid_retriever module not found. Using semantic search only.")
                self.use_hybrid = False

        # Create the retrieval chain
        self._initialize_chain()

    def _initialize_chain(self):
        """Initialize the RetrievalQA chain with optimized retrieval."""
        # Use hybrid retriever if available, otherwise use semantic only
        if self.hybrid_retriever:
            retriever = self.hybrid_retriever.get_retriever()
        else:
            # Enhanced semantic retrieval
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k}
            )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Stuff all retrieved docs into context
            retriever=retriever,
            chain_type_kwargs={"prompt": self.PROMPT},
            return_source_documents=True
        )

    def _is_conversational(self, question: str) -> Optional[str]:
        """
        Check if the question is conversational (greeting, thanks, etc.) and return appropriate response.

        Args:
            question: Question to check

        Returns:
            Response string if conversational, None otherwise
        """
        question_lower = question.lower().strip()

        # Remove punctuation for matching
        question_clean = question_lower.rstrip('!?.,:;')

        # Greetings
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if question_clean in greetings or any(question_clean.startswith(g) for g in greetings):
            return "Hello! I'm here to help you find information from your documents. What would you like to know?"

        # Thanks
        thanks = ['thank you', 'thanks', 'thank', 'thx', 'ty']
        if question_clean in thanks or any(question_clean.startswith(t) for t in thanks):
            return "You're welcome! Feel free to ask me anything about your documents."

        # How are you
        how_are_you = ['how are you', 'how are you doing', 'how do you do', 'whats up', "what's up"]
        if question_clean in how_are_you:
            return "I'm doing great, thank you for asking! I'm ready to help you explore your documents. What would you like to know?"

        # Bye
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell']
        if question_clean in goodbyes or any(question_clean.startswith(b) for b in goodbyes):
            return "Goodbye! Feel free to come back if you need help with your documents."

        return None

    def _extract_key_terms(self, question: str) -> List[str]:
        """
        Extract key terms from the question for enhanced search.

        Args:
            question: The user's question

        Returns:
            List of key terms
        """
        import re

        # Remove common question words but keep action verbs
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
                         'do', 'does', 'did', 'have', 'has', 'had', 'information', 'about', 'related',
                         'the', 'a', 'an', 'to', 'you', 'your', 'please', 'tell', 'me', 'describe'}

        # Extract potential key terms (capitalized words, numbers, special patterns)
        words = question.split()
        key_terms = []

        for word in words:
            clean_word = word.strip('.,?!;:')
            # Prioritize all-caps words (likely names/acronyms) and longer words
            if clean_word.isupper() and len(clean_word) > 1:
                # All caps word like "BOB" or "KUMAR"
                key_terms.insert(0, clean_word)  # Add to front (higher priority)
            elif clean_word.lower() not in question_words and len(clean_word) > 2:
                key_terms.append(clean_word)

        return key_terms

    def _preprocess_query(self, question: str) -> str:
        """
        Preprocess the query to improve retrieval with synonym expansion.

        Args:
            question: Original question

        Returns:
            Enhanced query string with synonyms
        """
        import re

        # Mahabharata-specific name variations (expand query with all variations)
        name_expansions = {
            'balarama': 'Balarama Baladeva Balabhadra Halayudha',
            'baladeva': 'Baladeva Balarama Balabhadra',
            'krishna': 'Krishna Vasudeva Keshava Madhava Govinda',
            'arjuna': 'Arjuna Partha Kiriti Dhananjaya',
            'abhimanyu': 'Abhimanyu son of Arjuna Subhadra',
            'draupadi': 'Draupadi Panchali Krishnaa Yajnaseni',
            'yudhishthira': 'Yudhishthira Dharmaraja Ajatashatru',
            'bhima': 'Bhima Bhimasena Vrikodara',
        }

        # Check for known names and expand
        question_lower = question.lower()
        for name, expansion in name_expansions.items():
            if name in question_lower:
                # Add all variations to improve retrieval
                question = f"{question} {expansion}"
                print(f"🔍 Query expansion: Added synonyms for '{name}' → {expansion}")
                break  # Only expand first match to avoid query bloat

        # Detect G.O. number queries - HIGHEST PRIORITY
        # Handles: "G.O.Rt.No. 1447", "G O Rt No 1447", "GO RT NO 1447"
        go_match = re.search(r'G\.?\s*O\.?\s*R[tT]\.?\s*N[oO]\.?\s*(\d+)', question, re.IGNORECASE)
        if go_match:
            go_number = go_match.group(1)
            # Create multiple variations to maximize retrieval
            enhanced = f"G.O.Rt.No. {go_number} G.O.RT.No. {go_number} GO Rt No {go_number} government order {go_number}"
            return enhanced

        # Detect if asking about orders/actions by someone
        # Patterns: "orders X passed", "what did X do", "actions by X"
        action_patterns = [
            r'orders?\s+(\w+[\s\w\.]*?)\s+(has\s+)?passed',
            r'what\s+(did|has)\s+(\w+[\s\w\.]*?)\s+(do|done)',
            r'actions?\s+(by|of)\s+(\w+[\s\w\.]*)',
            r'describe.*?orders?\s+(\w+[\s\w\.]*)'
        ]

        for pattern in action_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                # Extract the name and create an enhanced query
                # Focus on finding documents with that name AND order-related terms
                name_parts = question.split()
                # Find capitalized words (likely the name)
                caps_words = [w.strip('.,?!;:') for w in name_parts if w.strip('.,?!;:').isupper() or (len(w) > 2 and w[0].isupper())]
                if caps_words:
                    # Create enhanced query: "NAME order issued passed government"
                    enhanced = f"{' '.join(caps_words)} order issued passed government secretary"
                    return enhanced

        return question

    def _fetch_complete_go_document(self, go_number: str) -> List[Document]:
        """
        Fetch ALL chunks of a specific G.O. document to ensure complete information.

        Args:
            go_number: The G.O. number (e.g., "189", "1447")

        Returns:
            List of all document chunks for that G.O.
        """
        import re

        # Search for all variations of the G.O. number
        go_patterns = [
            f"G.O.Rt.No. {go_number}",
            f"G.O.RT.No. {go_number}",
            f"GO Rt No {go_number}",
            f"G.O.Rt.No.{go_number}",
            f"government order {go_number}"
        ]

        # Fetch more documents to ensure we get all chunks
        all_docs = self.vector_store.similarity_search(
            " ".join(go_patterns),
            k=50  # Fetch many to ensure complete G.O. coverage
        )

        # Filter to only chunks that actually contain this G.O. number
        go_docs = []
        for doc in all_docs:
            content = doc.page_content
            # Check if this chunk actually contains the G.O. number
            if re.search(rf'G\.?\s*O\.?\s*R[tT]\.?\s*N[oO]\.?\s*{go_number}\b', content, re.IGNORECASE):
                go_docs.append(doc)

        return go_docs

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer based on the documents.
        Uses multi-stage retrieval for better accuracy.

        Args:
            question: Question to ask

        Returns:
            Dictionary containing the answer and source documents
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "source_documents": []
            }

        # Check if it's a conversational message
        conversational_response = self._is_conversational(question)
        if conversational_response:
            return {
                "answer": conversational_response,
                "source_documents": []
            }

        try:
            import re

            # Check if asking about a specific G.O. number
            go_match = re.search(r'G\.?\s*O\.?\s*R[tT]\.?\s*N[oO]\.?\s*(\d+)', question, re.IGNORECASE)

            if go_match:
                # Specific G.O. query - fetch ALL chunks of that G.O.
                go_number = go_match.group(1)
                print(f"Detected specific G.O. query for G.O. {go_number} - fetching complete document...")

                complete_go_docs = self._fetch_complete_go_document(go_number)

                if complete_go_docs:
                    # Create context from ALL chunks of this G.O.
                    context = "\n\n---\n\n".join([doc.page_content for doc in complete_go_docs])

                    # Use LLM directly with complete context
                    from langchain.schema import HumanMessage

                    full_prompt = self.prompt_template.format(
                        context=context,
                        question=question
                    )

                    response = self.llm.invoke([HumanMessage(content=full_prompt)])

                    return {
                        "answer": response.content,
                        "source_documents": complete_go_docs
                    }

            # Regular query (not specific G.O.)
            # Preprocess query for better retrieval
            enhanced_query = self._preprocess_query(question)

            # Primary retrieval (use enhanced query if different from original)
            query_to_use = enhanced_query if enhanced_query != question else question
            result = self.qa_chain.invoke({"query": query_to_use})
            answer = result["result"]
            source_docs = result.get("source_documents", [])

            # If answer indicates information not found, try fallback strategies
            not_found_indicators = [
                "don't have that information",
                "don't have the information",
                "not available in the provided documents",
                "cannot find",
                "no information",
                "not mentioned",
                "does not contain"
            ]

            if any(indicator in answer.lower() for indicator in not_found_indicators):
                # Fallback Strategy 1: Try with enhanced query if we haven't already
                if query_to_use == question and enhanced_query != question:
                    try:
                        fallback_result = self.qa_chain.invoke({"query": enhanced_query})
                        fallback_answer = fallback_result["result"]
                        if not any(indicator in fallback_answer.lower() for indicator in not_found_indicators):
                            return {
                                "answer": fallback_answer,
                                "source_documents": fallback_result.get("source_documents", [])
                            }
                    except:
                        pass

                # Fallback Strategy 2: Try with key terms
                key_terms = self._extract_key_terms(question)
                if key_terms:
                    # Try searching with individual key terms
                    fallback_question = " ".join(key_terms[:4])  # Use top 4 key terms
                    try:
                        fallback_result = self.qa_chain.invoke({"query": fallback_question})
                        fallback_answer = fallback_result["result"]

                        # If fallback found something useful, use it
                        if not any(indicator in fallback_answer.lower() for indicator in not_found_indicators):
                            return {
                                "answer": fallback_answer,
                                "source_documents": fallback_result.get("source_documents", [])
                            }
                    except:
                        pass  # Continue with original answer

            return {
                "answer": answer,
                "source_documents": source_docs
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": []
            }

    def update_retriever_k(self, k: int):
        """
        Update the number of documents to retrieve.

        Args:
            k: New number of documents to retrieve
        """
        self.k = k
        self._initialize_chain()

    def get_relevant_documents(self, question: str, k: int = None):
        """
        Get relevant documents for a question without generating an answer.

        Args:
            question: Question to search for
            k: Number of documents to retrieve (optional)

        Returns:
            List of relevant documents
        """
        k = k or self.k
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return docs
