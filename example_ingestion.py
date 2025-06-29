#!/usr/bin/env python3
"""
Example script demonstrating the RAG ingestion pipeline

This script shows how to:
1. Ingest individual documents
2. Ingest documents from a directory
3. Handle different file types
4. Work with uploaded files (bytes)

Make sure to set up your .env file with the required API keys before running.
"""

import os
from pathlib import Path
from src.ingestion import (
    create_ingestion_pipeline,
    ingest_single_document,
    ingest_directory
)

def example_single_file_ingestion():
    """Example: Ingest a single document"""
    print("=" * 60)
    print("🔥 EXAMPLE 1: Single Document Ingestion")
    print("=" * 60)
    
    # Create an ingestion pipeline
    pipeline = create_ingestion_pipeline(index_name="example-docs")
    
    # Example with a text file (you can replace with your own file)
    example_file = "sample_document.txt"
    
    # Create a sample document if it doesn't exist
    if not os.path.exists(example_file):
        with open(example_file, 'w') as f:
            f.write("""
# Sample Document for RAG Testing

## Introduction
This is a sample document to demonstrate the RAG ingestion pipeline.
It contains multiple sections to test chunking strategies.

## Machine Learning Basics
Machine learning is a subset of artificial intelligence that enables computers 
to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data  
3. Reinforcement Learning: Learning through trial and error

## Natural Language Processing
NLP is a branch of AI that helps computers understand and process human language.
Common NLP tasks include text classification, sentiment analysis, and question answering.

## Conclusion
This document serves as a test case for document ingestion and retrieval systems.
            """.strip())
        print(f"📝 Created sample file: {example_file}")
    
    # Ingest the document
    result = pipeline.ingest_document(
        file_path_or_bytes=example_file,
        additional_metadata={
            "source": "example_script",
            "category": "sample",
            "priority": "high"
        }
    )
    
    print(f"\n📊 Ingestion Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return pipeline

def example_bytes_ingestion():
    """Example: Ingest from bytes (simulating file upload)"""
    print("\n" + "=" * 60)
    print("🔥 EXAMPLE 2: Bytes/Upload Ingestion")
    print("=" * 60)
    
    # Simulate uploaded file content
    document_content = """
RAG Systems: Retrieval-Augmented Generation

Introduction:
RAG systems combine retrieval and generation to provide accurate, 
context-aware responses by fetching relevant information from a knowledge base.

Key Components:
- Document Ingestion: Parse and chunk documents
- Vector Storage: Store embeddings in a vector database
- Retrieval: Find relevant chunks for a query
- Generation: Use LLM to generate response with context

Benefits:
- Up-to-date information without retraining models
- Reduced hallucination through grounding
- Transparent source attribution
- Cost-effective compared to fine-tuning

This demonstrates how to handle file uploads in a Streamlit app.
    """.strip()
    
    # Convert to bytes (simulating file upload)
    document_bytes = document_content.encode('utf-8')
    
    # Ingest using convenience function
    result = ingest_single_document(
        file_path_or_bytes=document_bytes,
        file_extension=".txt",
        index_name="example-docs",
        additional_metadata={
            "upload_method": "streamlit",
            "document_type": "guide"
        }
    )
    
    print(f"\n📊 Ingestion Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

def example_directory_ingestion():
    """Example: Ingest multiple documents from a directory"""
    print("\n" + "=" * 60)
    print("🔥 EXAMPLE 3: Directory Batch Ingestion")
    print("=" * 60)
    
    # Create a sample directory with multiple files
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Create multiple sample files
    documents = {
        "python_basics.txt": """
Python Programming Fundamentals

Variables and Data Types:
Python supports various data types including integers, floats, strings, lists, and dictionaries.
Variables are created by assignment and don't need explicit declaration.

Control Structures:
- if/elif/else statements for conditional logic
- for and while loops for iteration
- try/except blocks for error handling

Functions:
Functions are defined using the 'def' keyword and can accept parameters and return values.
They promote code reusability and organization.
        """,
        
        "data_science.md": """
# Data Science with Python

## Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning

## Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Building
6. Evaluation
        """,
        
        "web_development.txt": """
Web Development Concepts

Frontend Technologies:
- HTML: Structure and content
- CSS: Styling and layout
- JavaScript: Interactivity and dynamic behavior

Backend Technologies:
- Python: Flask, Django, FastAPI
- Database: SQL and NoSQL options
- APIs: RESTful services and GraphQL

Modern frameworks like React, Vue, and Angular have revolutionized frontend development.
        """
    }
    
    # Write sample files
    for filename, content in documents.items():
        file_path = sample_dir / filename
        with open(file_path, 'w') as f:
            f.write(content.strip())
    
    print(f"📁 Created {len(documents)} sample documents in {sample_dir}")
    
    # Ingest the entire directory
    results = ingest_directory(
        directory_path=sample_dir,
        index_name="example-docs",
        recursive=True,
        additional_metadata={
            "batch_source": "example_directory",
            "content_category": "programming"
        }
    )
    
    print(f"\n📊 Batch Ingestion Summary:")
    print(f"  Total files processed: {len(results)}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        filename = result.get('filename', 'unknown')
        chunks = result.get('total_chunks', 0)
        time_taken = result.get('processing_time_seconds', 0)
        print(f"  {status} {filename}: {chunks} chunks in {time_taken:.2f}s")

def example_pipeline_stats():
    """Example: Get pipeline statistics"""
    print("\n" + "=" * 60)
    print("🔥 EXAMPLE 4: Pipeline Statistics")
    print("=" * 60)
    
    pipeline = create_ingestion_pipeline(index_name="example-docs")
    stats = pipeline.get_ingestion_stats()
    
    print("📈 Current Pipeline Configuration:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def cleanup_example_files():
    """Clean up example files created during demonstration"""
    print("\n" + "=" * 60)
    print("🧹 CLEANUP")
    print("=" * 60)
    
    # Remove sample files
    files_to_remove = ["sample_document.txt"]
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Removed {file}")
    
    # Remove sample directory
    import shutil
    sample_dir = Path("sample_documents")
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
        print(f"🗑️  Removed directory {sample_dir}")

if __name__ == "__main__":
    print("🚀 RAG Ingestion Pipeline Examples")
    print("Make sure your .env file is configured with API keys!")
    
    try:
        # Run examples
        pipeline = example_single_file_ingestion()
        example_bytes_ingestion()
        example_directory_ingestion()
        example_pipeline_stats()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("  1. Check your Pinecone index for the ingested documents")
        print("  2. Try querying the documents using the retrieval components")
        print("  3. Integrate with your Streamlit frontend")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("Make sure your environment variables are set correctly in .env")
    
    finally:
        # Clean up example files
        cleanup_response = input("\n🧹 Clean up example files? (y/n): ").lower().strip()
        if cleanup_response == 'y':
            cleanup_example_files()
        else:
            print("📁 Example files kept for your reference") 