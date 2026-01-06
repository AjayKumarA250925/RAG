"""
Demo Readiness Verification Script
Checks if the RAG system is properly configured for production demo.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

def check_environment_variables():
    """Check if all required environment variables are set."""
    print("\n" + "="*60)
    print("🔍 CHECKING ENVIRONMENT VARIABLES")
    print("="*60)

    load_dotenv()

    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT")
    }

    all_set = True
    for var_name, var_value in required_vars.items():
        if var_value:
            masked_value = var_value[:10] + "..." if len(var_value) > 10 else var_value
            print(f"  ✅ {var_name}: {masked_value}")
        else:
            print(f"  ❌ {var_name}: NOT SET")
            all_set = False

    return all_set, required_vars

def check_pinecone_status(api_key, index_name):
    """Check Pinecone index status and statistics."""
    print("\n" + "="*60)
    print("🔍 CHECKING PINECONE STATUS")
    print("="*60)

    try:
        pc = Pinecone(api_key=api_key)

        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"  ❌ Index '{index_name}' does not exist!")
            print(f"  📝 Available indexes: {existing_indexes}")
            return False

        print(f"  ✅ Index '{index_name}' exists")

        # Get index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        total_vectors = stats.get('total_vector_count', 0)
        dimension = stats.get('dimension', 0)
        namespaces = stats.get('namespaces', {})

        print(f"\n  📊 Index Statistics:")
        print(f"     • Total Vectors: {total_vectors:,}")
        print(f"     • Vector Dimension: {dimension}")
        print(f"     • Namespaces: {len(namespaces)}")

        # Estimate number of documents (assuming ~15-20 chunks per document)
        estimated_docs = total_vectors // 18  # Average chunks per doc

        if total_vectors == 0:
            print(f"\n  ⚠️  WARNING: Index is empty!")
            print(f"     You need to upload documents before the demo.")
            return False
        elif total_vectors < 500:
            print(f"\n  ⚠️  WARNING: Only {total_vectors} vectors found")
            print(f"     Estimated documents: ~{estimated_docs}")
            print(f"     Expected for 219 docs: ~3,000-4,000 vectors")
            print(f"     Consider uploading more documents.")
        else:
            print(f"\n  ✅ Good vector count: {total_vectors:,}")
            print(f"     Estimated documents: ~{estimated_docs}")

        return True

    except Exception as e:
        print(f"\n  ❌ Error connecting to Pinecone: {str(e)}")
        return False

def check_dependencies():
    """Check if critical Python packages are installed."""
    print("\n" + "="*60)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*60)

    required_packages = {
        "streamlit": "Streamlit",
        "langchain": "LangChain",
        "pinecone": "Pinecone",
        "openai": "OpenAI",
        "rank_bm25": "BM25 (for hybrid search)",
        "pytesseract": "Tesseract OCR (optional)",
        "pdf2image": "PDF to Image (optional)",
    }

    all_installed = True
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {description}")
        except ImportError:
            if package in ["pytesseract", "pdf2image"]:
                print(f"  ⚠️  {description} - Optional (needed for scanned PDFs)")
            else:
                print(f"  ❌ {description} - REQUIRED")
                all_installed = False

    return all_installed

def check_tesseract():
    """Check if Tesseract OCR is installed (for scanned PDFs)."""
    print("\n" + "="*60)
    print("🔍 CHECKING TESSERACT OCR")
    print("="*60)

    try:
        import pytesseract
        from PIL import Image
        import os

        # Configure Tesseract path for Windows (same as document_processor.py)
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"  ✅ Tesseract OCR installed: v{version}")
        print(f"  ✅ Location: {pytesseract.pytesseract.tesseract_cmd}")
        return True
    except ImportError:
        print(f"  ⚠️  pytesseract package not installed")
        print(f"     Install with: pip install pytesseract")
        return False
    except Exception as e:
        print(f"  ⚠️  Tesseract OCR not found on system")
        print(f"     Download from: https://github.com/tesseract-ocr/tesseract")
        print(f"     This is OPTIONAL - only needed for scanned PDFs")
        return False

def verify_configuration_files():
    """Verify configuration files exist."""
    print("\n" + "="*60)
    print("🔍 CHECKING CONFIGURATION FILES")
    print("="*60)

    required_files = {
        "app.py": "Main application",
        "rag_chain.py": "RAG chain logic",
        "hybrid_retriever.py": "Hybrid retrieval",
        "document_processor.py": "Document processing",
        "vector_store.py": "Vector store manager",
        "requirements.txt": "Dependencies",
        ".env": "Environment variables"
    }

    all_exist = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"  ✅ {filename} ({description})")
        else:
            print(f"  ❌ {filename} ({description}) - MISSING!")
            all_exist = False

    return all_exist

def main():
    """Main verification function."""
    print("="*60)
    print("🎯 RAG DEMO READINESS VERIFICATION")
    print("="*60)

    # Run all checks
    checks_passed = 0
    total_checks = 5

    # 1. Environment variables
    env_ok, env_vars = check_environment_variables()
    if env_ok:
        checks_passed += 1

    # 2. Configuration files
    files_ok = verify_configuration_files()
    if files_ok:
        checks_passed += 1

    # 3. Dependencies
    deps_ok = check_dependencies()
    if deps_ok:
        checks_passed += 1

    # 4. Tesseract (optional)
    tesseract_ok = check_tesseract()
    if tesseract_ok:
        checks_passed += 1

    # 5. Pinecone status
    if env_ok and env_vars.get("PINECONE_API_KEY") and env_vars.get("PINECONE_INDEX_NAME"):
        pinecone_ok = check_pinecone_status(
            env_vars["PINECONE_API_KEY"],
            env_vars["PINECONE_INDEX_NAME"]
        )
        if pinecone_ok:
            checks_passed += 1
    else:
        print("\n⚠️  Skipping Pinecone check (missing API key or index name)")

    # Final summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    print(f"  Checks Passed: {checks_passed}/{total_checks}")

    if checks_passed == total_checks:
        print("\n  🎉 ✅ SYSTEM READY FOR DEMO!")
        print("\n  Next steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Upload your 219 documents")
        print("  3. Test sample queries")
        print("  4. You're ready to go!")
    elif checks_passed >= 3:
        print("\n  ⚠️  SYSTEM MOSTLY READY (minor issues)")
        print("\n  Review warnings above and fix if needed.")
        print("  You can proceed with demo for basic functionality.")
    else:
        print("\n  ❌ SYSTEM NOT READY")
        print("\n  Critical issues found. Fix errors above before demo.")

    print("="*60)

if __name__ == "__main__":
    main()
