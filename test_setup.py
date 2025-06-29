#!/usr/bin/env python3
"""
Setup Validation Script for RAG Ingestion Pipeline

This script validates that your environment is correctly configured
and all dependencies are properly installed.

Run this before using the ingestion pipeline to catch setup issues early.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.8+)")
        return False


def check_imports():
    """Check if all required packages can be imported"""
    print("\n📦 Checking package imports...")
    
    required_packages = [
        ("streamlit", "Streamlit web framework"),
        ("langchain", "LangChain core"),
        ("langchain_openai", "LangChain OpenAI integration"),
        ("langchain_google_genai", "LangChain Google GenAI integration"),
        ("langchain_pinecone", "LangChain Pinecone integration"),
        ("pinecone", "Pinecone vector database"),
        ("openai", "OpenAI API client"),
        ("pandas", "Data manipulation"),
        ("plotly", "Interactive visualizations"),
        ("PyPDF2", "PDF processing"),
        ("docx", "DOCX processing"),
        ("tiktoken", "Token counting")
    ]
    
    failed_imports = []
    
    for package, description in required_packages:
        try:
            if package == "docx":
                import docx
            else:
                __import__(package)
            print(f"   ✅ {package:<20} - {description}")
        except ImportError as e:
            print(f"   ❌ {package:<20} - {description} (MISSING)")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Missing packages: {', '.join(failed_imports)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    return True


def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n🔐 Checking environment variables...")
    
    required_vars = [
        ("OPENAI_API_KEY", "OpenAI API access", True),
        ("PINECONE_API_KEY", "Pinecone vector database access", True),
        ("GEMINI_API_KEY", "Google GenAI access", False),
    ]
    
    optional_vars = [
        ("EMBEDDING_PROVIDER", "Embedding service selection"),
        ("LLM_PROVIDER", "LLM service selection"),
        ("PINECONE_ENVIRONMENT", "Pinecone region"),
        ("CHUNK_SIZE", "Document chunk size"),
        ("CHUNK_OVERLAP", "Chunk overlap size"),
        ("CHUNKING_STRATEGY", "Chunking strategy")
    ]
    
    missing_required = []
    
    # Check required variables
    for var, description, required in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the API key for security
            if "API_KEY" in var:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"   ✅ {var:<20} - {description} ({masked_value})")
            else:
                print(f"   ✅ {var:<20} - {description} ({value})")
        elif required:
            print(f"   ❌ {var:<20} - {description} (MISSING)")
            missing_required.append(var)
        else:
            print(f"   ⚠️  {var:<20} - {description} (Optional, not set)")
    
    # Check optional variables
    print("\n   Optional Configuration:")
    for var, description in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var:<20} - {description} ({value})")
        else:
            print(f"   ⚪ {var:<20} - {description} (Using default)")
    
    if missing_required:
        print(f"\n❌ Missing required variables: {', '.join(missing_required)}")
        print("💡 Create a .env file with your API keys")
        return False
    
    return True


def check_file_structure():
    """Check if required files and directories exist"""
    print("\n📁 Checking project structure...")
    
    required_items = [
        ("src/", "directory", "Source code directory"),
        ("src/ingestion/", "directory", "Ingestion module"),
        ("src/retrieval/", "directory", "Retrieval module"),
        ("ui/", "directory", "UI components"),
        ("config.py", "file", "Configuration file"),
        ("streamlit_app.py", "file", "Streamlit application"),
        ("requirements.txt", "file", "Dependencies list"),
    ]
    
    missing_items = []
    
    for item, item_type, description in required_items:
        path = Path(item)
        
        if item_type == "directory":
            exists = path.is_dir()
        else:
            exists = path.is_file()
        
        if exists:
            print(f"   ✅ {item:<20} - {description}")
        else:
            print(f"   ❌ {item:<20} - {description} (MISSING)")
            missing_items.append(item)
    
    if missing_items:
        print(f"\n❌ Missing items: {', '.join(missing_items)}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic pipeline functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test configuration loading
        print("   🔧 Testing configuration...")
        from config import EMBEDDING_PROVIDER, LLM_PROVIDER
        print(f"      ✅ Embedding Provider: {EMBEDDING_PROVIDER}")
        print(f"      ✅ LLM Provider: {LLM_PROVIDER}")
        
        # Test embedder creation
        print("   🧮 Testing embedder creation...")
        from src.retrieval.embedder import get_embedder
        embedder = get_embedder()
        print(f"      ✅ Embedder created: {type(embedder).__name__}")
        
        # Test vector store creation
        print("   🗄️  Testing vector store creation...")
        from src.ingestion.vector_store import get_vector_store
        vector_store = get_vector_store()
        print(f"      ✅ Vector store created: {type(vector_store).__name__}")
        
        # Test chunker creation
        print("   ✂️  Testing chunker creation...")
        from src.ingestion.chunking import get_chunker
        chunker = get_chunker()
        print(f"      ✅ Chunker created: {type(chunker).__name__}")
        
        # Test document parser
        print("   📄 Testing document parser...")
        from src.ingestion.document_parser import DocumentParserFactory
        supported_types = DocumentParserFactory.supported_extensions()
        print(f"      ✅ Supported file types: {', '.join(supported_types)}")
        
        print("   ✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {str(e)}")
        return False


def main():
    """Run all validation checks"""
    
    print("🔍 RAG Ingestion Pipeline - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Environment Variables", check_environment_variables),
        ("File Structure", check_file_structure),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        try:
            if check_function():
                passed_checks += 1
        except Exception as e:
            print(f"\n❌ Error during {check_name} check: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! Your setup is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run streamlit_app.py")
        print("   2. Or run: python example_ingestion.py")
        print("   3. Upload some test documents and try the pipeline!")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before proceeding.")
        print("\n💡 Common solutions:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Create .env file with API keys")
        print("   - Check that all project files are present")
    
    return passed_checks == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 