"""
Document processor module for handling document uploads and text extraction.
Supports both text-based and scanned (OCR) PDFs.
"""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document

# OCR-specific imports (optional, will check if available)
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    import io

    # Configure Tesseract path for Windows
    if os.name == 'nt':  # Windows
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class DocumentProcessor:
    """Handles document loading and processing for RAG application with OCR support."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300, use_ocr: bool = True):
        """
        Initialize the document processor with optimized SENTENCE-BASED chunking.

        Args:
            chunk_size: Size of text chunks for splitting (1500 for better context, like Kernel Memory)
            chunk_overlap: Overlap between chunks (300 for continuity, similar to reference app)
            use_ocr: Whether to use OCR ONLY for actual image-based PDFs (default: TRUE, auto-detects)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr and OCR_AVAILABLE

        print(f"📋 Document Processor initialized:")
        print(f"   • Chunking strategy: SENTENCE-BASED (semantic boundaries)")
        print(f"   • Chunk size: {chunk_size} chars")
        print(f"   • Chunk overlap: {chunk_overlap} chars")
        print(f"   • OCR mode: SMART (only for image-based PDFs, not text PDFs)")
        
        if use_ocr and not OCR_AVAILABLE:
            print("⚠️ OCR libraries not installed. Install with: pip install pymupdf pytesseract pillow")
            print("⚠️ Also install Tesseract: https://github.com/tesseract-ocr/tesseract")
            print("⚠️ Falling back to text-based PDF extraction only.")
        
        # SENTENCE-BASED text splitter with prioritized semantic boundaries
        # Prioritizes natural language boundaries for better retrieval accuracy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # 1. Section breaks (highest priority)
                "\n\n",    # 2. Paragraph breaks
                "\n",      # 3. Line breaks
                ". ",      # 4. SENTENCE END (primary semantic boundary)
                "! ",      # 5. Exclamation sentences
                "? ",      # 6. Question sentences
                "; ",      # 7. Semicolon clauses
                ", ",      # 8. Comma clauses
                " ",       # 9. Word boundaries
                ""         # 10. Character split (last resort)
            ],
            keep_separator=True,  # Preserves punctuation for readability
            is_separator_regex=False
        )

        print(f"   • Separator priority: Section → Paragraph → Line → SENTENCE → Words")

    def _is_pdf_scanned(self, file_path: str, sample_pages: int = 2) -> bool:
        """
        Detect if a PDF is scanned by checking if text extraction yields minimal text.

        Args:
            file_path: Path to PDF file
            sample_pages: Number of pages to sample for detection

        Returns:
            True if PDF appears to be scanned, False otherwise
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Check first few pages
            pages_to_check = min(sample_pages, len(documents))
            total_text = ""
            
            for i in range(pages_to_check):
                total_text += documents[i].page_content
            
            # If very little text extracted, it's likely scanned
            # INCREASED THRESHOLD: less than 200 characters per page suggests scanned PDF
            # Most text-based G.O. documents have 500-2000+ chars per page
            threshold = 200 * pages_to_check
            text_length = len(total_text.strip())
            is_scanned = text_length < threshold

            if is_scanned:
                print(f"📷 Detected scanned PDF: {os.path.basename(file_path)} ({text_length} chars < {threshold} threshold)")
            else:
                print(f"📄 Detected text-based PDF: {os.path.basename(file_path)} ({text_length} chars)")
            
            return is_scanned
            
        except Exception as e:
            print(f"⚠️ Error detecting PDF type: {str(e)}")
            return False

    def _extract_text_with_ocr(self, file_path: str) -> List[Document]:
        """
        Extract text from scanned PDF using OCR with PyMuPDF.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects with extracted text
        """
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR libraries not installed. Cannot process scanned PDFs.")

        print(f"🔍 Running OCR on: {os.path.basename(file_path)}...")

        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(file_path)
            total_pages = len(pdf_document)

            documents = []
            for page_num in range(total_pages):
                # Get page
                page = pdf_document[page_num]

                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # Extract text using Tesseract OCR
                text = pytesseract.image_to_string(image, lang='eng')

                # Create Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': file_path,
                        'page': page_num + 1,
                        'extraction_method': 'ocr'
                    }
                )
                documents.append(doc)

                print(f"  ✓ Processed page {page_num + 1}/{total_pages}")

            pdf_document.close()
            print(f"✅ OCR completed: {total_pages} pages processed")
            return documents

        except Exception as e:
            print(f"❌ OCR failed: {str(e)}")
            raise

    def load_document(self, file_path: str):
        """
        Load a document based on its file extension.
        Automatically detects and handles scanned PDFs with OCR.

        Args:
            file_path: Path to the document file

        Returns:
            List of document chunks
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            # Check if PDF is scanned and use OCR if needed
            if self.use_ocr and self._is_pdf_scanned(file_path):
                try:
                    documents = self._extract_text_with_ocr(file_path)
                except Exception as e:
                    print(f"⚠️  OCR failed for scanned PDF: {os.path.basename(file_path)}")
                    print(f"    Error: {str(e)}")
                    print(f"    Falling back to standard PDF extraction (may have limited text)")
                    # Fallback to standard extraction (will have minimal text for scanned PDFs)
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
            else:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
            documents = loader.load()
            
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return documents

    def process_documents(self, file_paths: List[str]):
        """
        Process multiple documents and split them into chunks.
        Automatically handles both text-based and scanned PDFs.

        Args:
            file_paths: List of paths to document files

        Returns:
            List of processed document chunks with metadata
        """
        all_documents = []
        successful = 0
        failed = 0
        failed_files = []

        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"\n📄 Processing {i}/{len(file_paths)}: {os.path.basename(file_path)}")
                documents = self.load_document(file_path)

                # Check if documents actually have content
                if not documents or all(len(doc.page_content.strip()) < 50 for doc in documents):
                    print(f"   ⚠️  WARNING: Document appears empty or has minimal content")
                    print(f"   Content length: {sum(len(doc.page_content) for doc in documents)} chars")
                    failed += 1
                    failed_files.append(os.path.basename(file_path))
                    continue

                # Add source filename to metadata for better tracking
                filename = os.path.basename(file_path)
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['source_file'] = filename
                    doc.metadata['file_path'] = file_path

                all_documents.extend(documents)
                successful += 1
                print(f"   ✅ Success: {len(documents)} pages, {sum(len(d.page_content) for d in documents)} chars")

            except Exception as e:
                print(f"   ❌ Error processing {os.path.basename(file_path)}: {str(e)}")
                failed += 1
                failed_files.append(os.path.basename(file_path))
                continue

        # Split documents into chunks while preserving metadata
        chunks = self.text_splitter.split_documents(all_documents)

        # DEDUPLICATION: Remove duplicate chunks (like reference app does)
        seen_content = set()
        unique_chunks = []
        duplicate_count = 0

        for chunk in chunks:
            # Create a hash of the content (first 100 chars as identifier)
            content_id = chunk.page_content[:100] if len(chunk.page_content) > 100 else chunk.page_content
            content_hash = hash(content_id)

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
            else:
                duplicate_count += 1

        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"  • Total files attempted: {len(file_paths)}")
        print(f"  • Successfully processed: {successful}")
        print(f"  • Failed: {failed}")
        print(f"  • Total pages extracted: {len(all_documents)}")
        print(f"  • Total chunks created: {len(chunks)}")
        print(f"  • Duplicate chunks removed: {duplicate_count}")
        print(f"  • Unique chunks: {len(unique_chunks)}")
        print(f"  • Average chunks per document: {len(unique_chunks) / successful if successful else 0:.1f}")

        if failed_files:
            print(f"\n❌ Failed files ({len(failed_files)}):")
            for ff in failed_files[:10]:  # Show first 10
                print(f"   - {ff}")
            if len(failed_files) > 10:
                print(f"   ... and {len(failed_files) - 10} more")

        print(f"{'='*60}\n")

        return unique_chunks  # Return deduplicated chunks

    def process_text(self, text: str):
        """
        Process raw text and split it into chunks.

        Args:
            text: Raw text string

        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks