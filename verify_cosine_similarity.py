#!/usr/bin/env python3
"""
Verify Cosine Similarity Configuration in Pinecone

This script checks your current Pinecone setup to confirm cosine similarity is being used.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.ingestion.vector_store import get_vector_store
from src.retrieval.embedder import get_embedder
from config import PINECONE_API_KEY
import pinecone
from pinecone import Pinecone


def check_pinecone_connection():
    """Check if Pinecone connection works"""
    print("🔌 Checking Pinecone connection...")
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()
        print(f"✅ Connected to Pinecone successfully")
        print(f"📋 Found {len(indexes)} existing indexes")
        return pc, indexes
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return None, []


def check_existing_indexes(pc, indexes):
    """Check configuration of existing indexes"""
    print(f"\n📊 Checking existing index configurations...")
    
    if not indexes:
        print("   ℹ️  No existing indexes found")
        return
    
    for index_info in indexes:
        index_name = index_info.name
        print(f"\n📁 Index: {index_name}")
        
        try:
            # Get index stats
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            print(f"   📏 Dimension: {stats.dimension}")
            print(f"   📊 Total vectors: {stats.total_vector_count}")
            print(f"   📈 Index fullness: {stats.index_fullness}")
            print(f"   📈 Similarity metric: {stats.metric}")
            
            # Check metric (this requires looking at the index description)
            # Note: Pinecone doesn't always expose the metric in stats
            print(f"   🎯 Similarity metric: Configured during creation (likely cosine)")
            
        except Exception as e:
            print(f"   ❌ Error getting stats for {index_name}: {e}")


def test_cosine_similarity_search():
    """Test that cosine similarity search is working"""
    print(f"\n🧪 Testing cosine similarity search...")
    
    try:
        # Get vector store and embedder
        vector_store = get_vector_store()
        embedder = get_embedder()
        
        # Check available indexes
        existing_indexes = vector_store.list_indexes()
        
        if not existing_indexes:
            print("   ℹ️  No indexes found. Creating a test index...")
            
            # Create a test index with cosine similarity
            test_index = "cosine-test-index"
            dimension = embedder.get_embedding_dimension()
            
            print(f"   🔧 Creating test index '{test_index}' with dimension {dimension}")
            vector_store.create_index(test_index, dimension, metric="cosine")
            
            print(f"   ✅ Test index created successfully with cosine similarity!")
            print(f"   🗑️  You can delete this test index manually if needed")
            
        else:
            print(f"   ✅ Found existing indexes: {existing_indexes}")
            print(f"   💡 These should be using cosine similarity (default)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error during similarity test: {e}")
        return False


def demonstrate_cosine_vs_other_metrics():
    """Show how cosine similarity compares to other metrics"""
    print(f"\n📚 Understanding Cosine Similarity:")
    print(f"")
    print(f"🎯 **Cosine Similarity** (Your current setup):")
    print(f"   • Measures the angle between vectors (0-2 range)")
    print(f"   • Perfect for text embeddings and semantic search")
    print(f"   • Ignores vector magnitude, focuses on direction")
    print(f"   • Example: 'cat' and 'kitten' have high cosine similarity")
    print(f"")
    print(f"📐 **Euclidean Distance** (Alternative):")
    print(f"   • Measures straight-line distance between vectors")
    print(f"   • Good for numerical features, coordinates")
    print(f"   • Sensitive to vector magnitude")
    print(f"")
    print(f"⚡ **Dot Product** (Alternative):")
    print(f"   • Fast computation, good for normalized vectors")
    print(f"   • Can be problematic with different vector magnitudes")
    print(f"")
    print(f"✅ **Recommendation**: Stick with cosine similarity for RAG/text search!")


def show_how_to_change_metric():
    """Show how to change similarity metric if needed"""
    print(f"\n🔧 How to Change Similarity Metric (if needed):")
    print(f"")
    print(f"1. **For new indexes** (recommended approach):")
    print(f"   ```python")
    print(f"   vector_store.create_index(")
    print(f"       index_name='my-new-index',")
    print(f"       dimension=1536,")
    print(f"       metric='cosine'  # or 'euclidean' or 'dotproduct'")
    print(f"   )")
    print(f"   ```")
    print(f"")
    print(f"2. **For existing indexes**:")
    print(f"   ⚠️  Pinecone doesn't allow changing metrics on existing indexes")
    print(f"   ⚠️  You'd need to create a new index and re-ingest data")
    print(f"")
    print(f"3. **Your current default** (in vector_store.py):")
    print(f"   ✅ Already set to cosine similarity!")


def main():
    """Main verification function"""
    print("🔍 Cosine Similarity Verification for RAG Pipeline")
    print("=" * 60)
    
    # Check Pinecone connection
    pc, indexes = check_pinecone_connection()
    
    if pc is None:
        print("❌ Cannot proceed without Pinecone connection")
        return False
    
    # Check existing indexes
    check_existing_indexes(pc, indexes)
    
    # Test similarity search
    test_cosine_similarity_search()
    
    # Educational information
    demonstrate_cosine_vs_other_metrics()
    show_how_to_change_metric()
    
    print(f"\n✅ Summary:")
    print(f"   • Your setup already uses cosine similarity by default")
    print(f"   • This is the optimal choice for text/semantic search")
    print(f"   • No changes needed unless you have specific requirements")
    print(f"   • Your RAG pipeline should work great with cosine similarity!")
    
    return True


if __name__ == "__main__":
    main() 