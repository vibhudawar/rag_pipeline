#!/usr/bin/env python3
"""
Test RAG Pipeline

This script tests the complete RAG pipeline with sample data.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.rag_pipeline import create_rag_pipeline, simple_rag_query
from src.ingestion import ingest_single_document
import tempfile


def create_sample_documents():
    """Create sample documents for testing"""
    
    sample_docs = [
        {
            'filename': 'machine_learning_guide.txt',
            'content': """
            Machine Learning Guide
            
            Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. 
            
            Key types of machine learning:
            1. Supervised Learning - Learning with labeled data
            2. Unsupervised Learning - Finding patterns in unlabeled data  
            3. Reinforcement Learning - Learning through interaction and feedback
            
            Popular algorithms include:
            - Linear Regression
            - Decision Trees
            - Neural Networks
            - Support Vector Machines
            
            Applications:
            - Image recognition
            - Natural language processing
            - Recommendation systems
            - Autonomous vehicles
            """
        },
        {
            'filename': 'ai_ethics.txt',
            'content': """
            AI Ethics and Responsible AI
            
            As artificial intelligence becomes more prevalent, ensuring ethical AI development is crucial.
            
            Key ethical considerations:
            1. Bias and Fairness - Ensuring AI systems don't discriminate
            2. Transparency - Making AI decisions explainable
            3. Privacy - Protecting user data and personal information
            4. Accountability - Establishing responsibility for AI decisions
            
            Best practices:
            - Diverse development teams
            - Regular bias testing
            - Clear documentation
            - User consent and control
            - Continuous monitoring
            
            Organizations should implement ethical AI frameworks to guide development and deployment.
            """
        }
    ]
    
    return sample_docs


def test_ingestion(sample_docs, index_name="test-rag-pipeline"):
    """Test document ingestion"""
    
    print("🔄 Testing document ingestion...")
    
    results = []
    
    for doc_info in sample_docs:
        try:
            # Ingest document
            result = ingest_single_document(
                file_path_or_bytes=doc_info['content'].encode(),
                file_extension='.txt',
                index_name=index_name,
                additional_metadata={
                    'test_document': True,
                    'category': 'sample'
                }
            )
            
            if result['success']:
                print(f"✅ Ingested: {doc_info['filename']}")
                print(f"   Chunks: {result['total_chunks']}")
            else:
                print(f"❌ Failed: {doc_info['filename']} - {result.get('error', 'Unknown error')}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Exception ingesting {doc_info['filename']}: {str(e)}")
            results.append({'success': False, 'error': str(e)})
    
    successful_ingestions = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 Ingestion complete: {successful_ingestions}/{len(sample_docs)} successful")
    
    return successful_ingestions > 0


def test_rag_pipeline(index_name="test-rag-pipeline"):
    """Test the complete RAG pipeline"""
    
    print("\n🤖 Testing RAG pipeline...")
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "What are the key types of machine learning?",
        "What are some important AI ethics considerations?",
        "How can we ensure fairness in AI systems?"
    ]
    
    try:
        # Create pipeline
        pipeline = create_rag_pipeline(
            index_name=index_name,
            include_web_search=False,  # Disable web search for testing
            reranker_provider="score",  # Use simple score-based reranker
            llm_provider="auto"
        )
        
        print(f"✅ Pipeline created successfully")
        
        # Test single queries
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Question: {question}")
            
            try:
                response = pipeline.query(question, return_sources=True)
                
                if response.get('success', False):
                    print(f"✅ Query successful")
                    print(f"   Model: {response.get('model', 'Unknown')}")
                    print(f"   Processing time: {response.get('processing_time', 0):.2f}s")
                    print(f"   Sources: {len(response.get('sources', []))}")
                    
                    # Show first 100 chars of answer
                    answer = response.get('answer', '')
                    preview = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"   Answer preview: {preview}")
                    
                else:
                    print(f"❌ Query failed: {response.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Exception during query: {str(e)}")
        
        # Test batch query
        print(f"\n--- Batch Query Test ---")
        try:
            batch_results = pipeline.batch_query(test_questions[:2], return_sources=False)
            
            successful_batch = sum(1 for r in batch_results if r.get('success', False))
            print(f"✅ Batch query complete: {successful_batch}/{len(test_questions[:2])} successful")
            
        except Exception as e:
            print(f"❌ Batch query failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {str(e)}")
        return False


def test_simple_interface(index_name="test-rag-pipeline"):
    """Test the simple interface"""
    
    print("\n🔍 Testing simple interface...")
    
    try:
        response = simple_rag_query(
            "What are the main applications of machine learning?",
            index_name=index_name
        )
        
        if response.get('success', False):
            print("✅ Simple interface works")
            return True
        else:
            print(f"❌ Simple interface failed: {response.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Simple interface exception: {str(e)}")
        return False


def cleanup_test_data(index_name="test-rag-pipeline"):
    """Clean up test data"""
    
    print(f"\n🧹 Cleaning up test data...")
    
    try:
        from src.ingestion.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        
        # Check if index exists and delete if needed
        indexes = vector_store.list_indexes()
        if index_name in indexes:
            # Note: Pinecone doesn't allow immediate deletion, so we'll just note it
            print(f"⚠️ Test index '{index_name}' exists - consider manual cleanup")
        else:
            print(f"✅ No cleanup needed for index '{index_name}'")
            
    except Exception as e:
        print(f"⚠️ Cleanup check failed: {str(e)}")


def main():
    """Main test function"""
    
    print("🧪 RAG Pipeline Test Suite")
    print("=" * 40)
    
    # Check environment
    print("🔍 Checking environment...")
    
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("💡 Please set these in your .env file before running tests")
        return False
    
    print("✅ Environment check passed")
    
    # Test index name
    index_name = "test-rag-pipeline"
    
    # Create sample documents
    sample_docs = create_sample_documents()
    print(f"📝 Created {len(sample_docs)} sample documents")
    
    # Test ingestion
    if not test_ingestion(sample_docs, index_name):
        print("❌ Ingestion tests failed, stopping here")
        return False
    
    # Wait a moment for indexing
    print("⏳ Waiting for indexing to complete...")
    import time
    time.sleep(5)
    
    # Test RAG pipeline
    if not test_rag_pipeline(index_name):
        print("❌ RAG pipeline tests failed")
        return False
    
    # Test simple interface
    if not test_simple_interface(index_name):
        print("❌ Simple interface tests failed")
        return False
    
    # Cleanup
    cleanup_test_data(index_name)
    
    print("\n🎉 All tests completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Try running the Streamlit app: streamlit run streamlit_app.py")
    print("   2. Upload your own documents")
    print("   3. Ask questions in the RAG Query tab")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 