#!/usr/bin/env python3
"""
RAG Ingestion Pipeline - Main Application Entry Point

This project provides a complete RAG (Retrieval-Augmented Generation) ingestion pipeline
with both programmatic and visual interfaces.

Usage Options:
1. Streamlit Web Interface (Recommended for testing):
   streamlit run streamlit_app.py

2. Command Line Examples:
   python example_ingestion.py

3. Programmatic Usage:
   from src.ingestion import ingest_single_document
   result = ingest_single_document("document.pdf")
"""

import os
import sys
from pathlib import Path

def main():
    """Main application entry point"""
    
    print("🚀 RAG Ingestion Pipeline")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found!")
        print("Create a .env file with your API keys:")
        print()
        print("OPENAI_API_KEY=sk-your-openai-api-key-here")
        print("PINECONE_API_KEY=your-pinecone-api-key-here")
        print("GEMINI_API_KEY=your-gemini-api-key-here  # optional")
        print()
        return
    
    print("📋 Available Options:")
    print()
    print("1. 🌐 Streamlit Web Interface (Recommended)")
    print("   streamlit run streamlit_app.py")
    print()
    print("2. 💻 Command Line Examples")
    print("   python example_ingestion.py")
    print()
    print("3. 📖 View Documentation")
    print("   - INGESTION_README.md - Core pipeline documentation")
    print("   - STREAMLIT_README.md - UI documentation")
    print()
    
    # Interactive choice
    try:
        choice = input("Choose an option (1-3) or press Enter for Streamlit: ").strip()
        
        if choice == "1" or choice == "":
            print("\n🌐 Starting Streamlit app...")
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
        elif choice == "2":
            print("\n💻 Running command line examples...")
            import subprocess
            subprocess.run([sys.executable, "example_ingestion.py"])
        
        elif choice == "3":
            print("\n📖 Documentation files:")
            docs = ["INGESTION_README.md", "STREAMLIT_README.md"]
            for doc in docs:
                if Path(doc).exists():
                    print(f"   ✅ {doc}")
                else:
                    print(f"   ❌ {doc} (not found)")
        
        else:
            print("Invalid choice. Please run one of the commands above manually.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please run the commands manually:")
        print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
