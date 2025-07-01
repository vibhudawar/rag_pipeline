"""
Complete RAG Pipeline

This module provides a comprehensive RAG (Retrieval-Augmented Generation) pipeline
that combines vector search, web search, reranking, and LLM generation.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import time

# Import our components
from src.retrieval.embedder import get_embedder
from src.retrieval.web_search import search_web
from src.ingestion.vector_store import get_vector_store
from src.reranking.reranker import get_reranker
from src.generation.llm import get_llm_generator


class RAGPipeline:
    """Complete RAG pipeline with hybrid retrieval, reranking, and generation"""
    
    def __init__(
        self,
        index_name: str,
        embedding_provider: str = "auto",
        reranker_provider: str = "auto",
        llm_provider: str = "auto",
        include_web_search: bool = True,
        vector_top_k: int = 10,
        web_search_results: int = 3,
        final_top_k: int = 5,
        **kwargs
    ):
        """
        Initialize RAG pipeline
        
        Args:
            index_name: Name of the vector index to search
            embedding_provider: Embedding provider (openai, gemini, auto)
            reranker_provider: Reranker provider (cohere, huggingface, score, auto)
            llm_provider: LLM provider (openai, gemini, auto)
            include_web_search: Whether to include web search in retrieval
            vector_top_k: Number of documents to retrieve from vector store
            web_search_results: Number of web search results to include
            final_top_k: Final number of documents after reranking
        """
        self.index_name = index_name
        self.include_web_search = include_web_search
        self.vector_top_k = vector_top_k
        self.web_search_results = web_search_results
        self.final_top_k = final_top_k
        
        # Initialize components
        self.embedder = get_embedder(embedding_provider)
        self.vector_store = get_vector_store()
        self.reranker = get_reranker(reranker_provider)
        self.llm_generator = get_llm_generator(llm_provider, **kwargs)
        
        print(f"🚀 RAG Pipeline initialized:")
        print(f"   📚 Index: {index_name}")
        print(f"   🔍 Embedder: {embedding_provider}")
        print(f"   🎯 Reranker: {reranker_provider}")
        print(f"   🤖 LLM: {llm_provider}")
        print(f"   🌐 Web search: {include_web_search}")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid approach (vector + web search)
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents
        """
        all_documents = []
        
        # 1. Vector search
        try:
            vector_docs = self.vector_store.similarity_search(
                index_name=self.index_name,
                query=query,
                embedder=self.embedder,
                top_k=self.vector_top_k
            )

            print(f"🔍 Vector docs: {vector_docs}")
            
            # Add source metadata
            for doc in vector_docs:
                doc.metadata['retrieval_source'] = 'vector_store'
                doc.metadata['retrieval_query'] = query
            
            all_documents.extend(vector_docs)
            
            print(f"📄 Retrieved {len(vector_docs)} documents from vector store")
            
        except Exception as e:
            print(f"⚠️ Vector search failed: {str(e)}")
        
        # 2. Web search (if enabled)
        if self.include_web_search:
            try:
                web_docs = search_web(query, num_results=self.web_search_results)
                
                # Add source metadata
                for doc in web_docs:
                    doc.metadata['retrieval_source'] = 'web_search'
                    doc.metadata['retrieval_query'] = query
                
                all_documents.extend(web_docs)
                
                print(f"🌐 Retrieved {len(web_docs)} documents from web search")
                
            except Exception as e:
                print(f"⚠️ Web search failed: {str(e)}")
        
        return all_documents
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents for better relevance
        
        Args:
            query: Original search query
            documents: Documents to rerank
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        try:
            reranked_docs = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=self.final_top_k
            )
            
            print(f"🎯 Reranked to top {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            print(f"⚠️ Reranking failed: {str(e)}, using original order")
            return documents[:self.final_top_k]
    
    def generate_answer(self, query: str, context_documents: List[Document]) -> Dict[str, Any]:
        """
        Generate answer using LLM with context documents
        
        Args:
            query: User question
            context_documents: Retrieved and reranked documents
            
        Returns:
            Generated response with metadata
        """
        try:
            result = self.llm_generator.generate(
                query=query,
                context_documents=context_documents
            )
            
            print(f"🤖 Generated response using {result.get('model', 'unknown')} model")
            return result
            
        except Exception as e:
            print(f"⚠️ Generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'context_documents': len(context_documents)
            }
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query pipeline
        
        Args:
            question: User question
            return_sources: Whether to include source documents in response
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        start_time = time.time()
        
        print(f"\n🔍 Processing query: '{question}'")
        
        # 1. Retrieve documents
        print("\n📚 Step 1: Retrieving documents...")
        documents = self.retrieve_documents(question)
        
        if not documents:
            return {
                'success': False,
                'error': 'No documents retrieved',
                'question': question,
                'processing_time': time.time() - start_time
            }
        
        # 2. Rerank documents
        print("\n🎯 Step 2: Reranking documents...")
        reranked_docs = self.rerank_documents(question, documents)
        
        # 3. Generate answer
        print("\n🤖 Step 3: Generating answer...")
        generation_result = self.generate_answer(question, reranked_docs)
        
        # 4. Compile final response
        response = {
            'success': generation_result.get('success', False),
            'question': question,
            'answer': generation_result.get('response', ''),
            'model': generation_result.get('model', 'unknown'),
            'processing_time': time.time() - start_time,
            'retrieval_stats': {
                'total_retrieved': len(documents),
                'vector_search': len([d for d in documents if d.metadata.get('retrieval_source') == 'vector_store']),
                'web_search': len([d for d in documents if d.metadata.get('retrieval_source') == 'web_search']),
                'final_context': len(reranked_docs)
            }
        }
        
        # Add error if generation failed
        if not generation_result.get('success', False):
            response['error'] = generation_result.get('error', 'Unknown error')
        
        # Add sources if requested
        if return_sources and reranked_docs:
            response['sources'] = []
            
            for i, doc in enumerate(reranked_docs):
                source_info = {
                    'rank': i + 1,
                    'title': doc.metadata.get('title', f'Document {i+1}'),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'retrieval_source': doc.metadata.get('retrieval_source', 'unknown'),
                    'url': doc.metadata.get('url', ''),
                    'rerank_score': doc.metadata.get('rerank_score'),
                    'similarity_score': doc.metadata.get('score')
                }
                response['sources'].append(source_info)
        
        print(f"\n✅ Query completed in {response['processing_time']:.2f} seconds")
        
        return response
    
    def batch_query(self, questions: List[str], return_sources: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            questions: List of questions to process
            return_sources: Whether to include sources in responses
            
        Returns:
            List of responses
        """
        print(f"\n📊 Processing batch of {len(questions)} questions...")
        
        results = []
        start_time = time.time()
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            result = self.query(question, return_sources=return_sources)
            result['batch_position'] = i
            results.append(result)
        
        total_time = time.time() - start_time
        print(f"\n📊 Batch completed in {total_time:.2f} seconds")
        print(f"   ⏱️ Average time per query: {total_time/len(questions):.2f} seconds")
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            'index_name': self.index_name,
            'include_web_search': self.include_web_search,
            'vector_top_k': self.vector_top_k,
            'web_search_results': self.web_search_results,
            'final_top_k': self.final_top_k,
            'embedder_type': type(self.embedder).__name__,
            'reranker_type': type(self.reranker).__name__,
            'llm_generator_type': type(self.llm_generator).__name__
        }


def create_rag_pipeline(
    index_name: str,
    embedding_provider: str = "auto",
    reranker_provider: str = "auto", 
    llm_provider: str = "auto",
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline
    
    Args:
        index_name: Name of the vector index
        embedding_provider: Embedding provider
        reranker_provider: Reranker provider
        llm_provider: LLM provider
        **kwargs: Additional arguments for pipeline configuration
        
    Returns:
        Configured RAG pipeline
    """
    return RAGPipeline(
        index_name=index_name,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        llm_provider=llm_provider,
        **kwargs
    )


# Convenience functions
def simple_rag_query(question: str, index_name: str = "rag-docs") -> Dict[str, Any]:
    """Simple RAG query with default settings"""
    pipeline = create_rag_pipeline(index_name)
    return pipeline.query(question)


def rag_query_with_options(
    question: str,
    index_name: str = "rag-docs",
    include_web_search: bool = True,
    reranker_provider: str = "auto",
    llm_provider: str = "auto"
) -> Dict[str, Any]:
    """RAG query with custom options"""
    pipeline = create_rag_pipeline(
        index_name=index_name,
        reranker_provider=reranker_provider,
        llm_provider=llm_provider,
        include_web_search=include_web_search
    )
    return pipeline.query(question) 