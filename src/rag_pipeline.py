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
from src.retrieval.web_search import get_web_searcher
from src.ingestion.vector_store import get_vector_store
from src.reranking.reranker import get_reranker
from src.generation.llm import get_llm_generator
from src.generation.conversational_rag import create_conversational_rag


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
        # New conversational parameters
        enable_conversation: bool = False,
        max_conversation_messages: int = 10,
        enable_summary: bool = True,
        summary_threshold: int = 8,
        **kwargs
    ):
        """
        Initialize RAG pipeline
        
        Args:
            index_name: Name of the vector store index
            embedding_provider: Embedding provider ("openai", "gemini", "auto")
            reranker_provider: Reranker provider ("cohere", "huggingface", "auto")
            llm_provider: LLM provider ("openai", "gemini", "auto")
            include_web_search: Whether to include web search results
            vector_top_k: Number of top results from vector search
            web_search_results: Number of web search results to include
            final_top_k: Final number of results after reranking
            enable_conversation: Whether to enable conversational memory
            max_conversation_messages: Maximum messages to keep in memory
            enable_summary: Whether to use summary memory for long conversations
            summary_threshold: Number of messages before summarizing
        """
        self.index_name = index_name
        self.include_web_search = include_web_search
        self.vector_top_k = vector_top_k
        self.web_search_results = web_search_results
        self.final_top_k = final_top_k
        self.enable_conversation = enable_conversation
        
        # Initialize components
        self.vector_store = get_vector_store()
        self.embedder = get_embedder(embedding_provider)
        self.reranker = get_reranker(reranker_provider)
        
        # Initialize generation components
        if enable_conversation:
            # Use conversational RAG
            self.conversational_rag = create_conversational_rag(
                llm_provider=llm_provider,
                max_messages=max_conversation_messages,
                enable_summary=enable_summary,
                summary_threshold=summary_threshold,
                **kwargs
            )
            self.llm_generator = None
        else:
            # Use standard LLM generator
            self.llm_generator = get_llm_generator(llm_provider, **kwargs)
            self.conversational_rag = None
        
        # Initialize web search if enabled
        if include_web_search:
            self.web_searcher = get_web_searcher()
        else:
            self.web_searcher = None
        
        print(f"✅ RAG Pipeline initialized:")
        print(f"   📊 Index: {index_name}")
        print(f"   🔍 Vector search: top-{vector_top_k}")
        print(f"   🌐 Web search: {'enabled' if include_web_search else 'disabled'}")
        print(f"   🎯 Final results: top-{final_top_k}")
        print(f"   💬 Conversation: {'enabled' if enable_conversation else 'disabled'}")
        
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents from multiple sources
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents
        """
        all_documents = []
        
        # 1. Vector search
        try:
            print(f"   📊 Searching vector store (top-{self.vector_top_k})...")
            vector_results = self.vector_store.similarity_search(
                index_name=self.index_name,
                query=query,
                embedder=self.embedder,
                top_k=self.vector_top_k
            )
            
            # Mark documents as from vector store
            for doc in vector_results:
                doc.metadata['retrieval_source'] = 'vector_store'
            
            all_documents.extend(vector_results)
            print(f"   ✅ Vector search: {len(vector_results)} documents")
            
        except Exception as e:
            print(f"   ⚠️ Vector search failed: {str(e)}")
        
        # 2. Web search (if enabled)
        if self.include_web_search and self.web_searcher:
            try:
                print(f"   🌐 Searching web (top-{self.web_search_results})...")
                web_results = self.web_searcher.search(
                    query=query,
                    num_results=self.web_search_results
                )
                
                # Mark documents as from web search
                for doc in web_results:
                    doc.metadata['retrieval_source'] = 'web_search'
                
                all_documents.extend(web_results)
                print(f"   ✅ Web search: {len(web_results)} documents")
                
            except Exception as e:
                print(f"   ⚠️ Web search failed: {str(e)}")
        
        return all_documents
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents for better relevance
        
        Args:
            query: Search query
            documents: Documents to rerank
            
        Returns:
            Reranked documents
        """
        if not documents:
            return documents
        
        try:
            print(f"   🎯 Reranking {len(documents)} documents...")
            reranked_docs = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=self.final_top_k
            )
            
            print(f"   ✅ Reranked to top-{len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            print(f"   ⚠️ Reranking failed: {str(e)}")
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
    
    def chat(self, message: str, thread_id: str = "default", **kwargs) -> Dict[str, Any]:
        """
        Chat with the conversational RAG system
        
        Args:
            message: User message
            thread_id: Conversation thread ID
            **kwargs: Additional arguments
            
        Returns:
            Response with conversation context
        """
        if not self.enable_conversation:
            raise ValueError("Conversation mode is not enabled. Set enable_conversation=True")
        
        start_time = time.time()
        
        print(f"\n💬 Processing conversational query: '{message}'")
        
        # 1. Retrieve documents
        print("\n📚 Step 1: Retrieving documents...")
        documents = self.retrieve_documents(message)
        
        # 2. Rerank documents
        print("\n🎯 Step 2: Reranking documents...")
        reranked_docs = self.rerank_documents(message, documents)
        
        # 3. Generate conversational response
        print("\n🤖 Step 3: Generating conversational response...")
        result = self.conversational_rag.chat(
            message=message,
            context_documents=reranked_docs,
            thread_id=thread_id,
            **kwargs
        )
        
        # 4. Add retrieval stats
        if result.get('success', False):
            result['retrieval_stats'] = {
                'total_retrieved': len(documents),
                'vector_search': len([d for d in documents if d.metadata.get('retrieval_source') == 'vector_store']),
                'web_search': len([d for d in documents if d.metadata.get('retrieval_source') == 'web_search']),
                'final_context': len(reranked_docs)
            }
        
        print(f"💬 Conversational response completed in {time.time() - start_time:.2f} seconds")
        
        return result
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query pipeline (non-conversational)
        
        Args:
            question: User question
            return_sources: Whether to include source documents in response
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        if self.enable_conversation:
            print("⚠️ Pipeline is in conversational mode. Use chat() method instead.")
            return self.chat(question)
        
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
        
        # Add sources if requested
        if return_sources and generation_result.get('success', False):
            response['sources'] = generation_result.get('sources', [])
        
        # Add error information if generation failed
        if not generation_result.get('success', False):
            response['error'] = generation_result.get('error', 'Unknown error')
        
        print(f"🔍 Query completed in {response['processing_time']:.2f} seconds")
        
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
            if self.enable_conversation:
                result = self.chat(question, thread_id=f"batch_{i}")
            else:
                result = self.query(question, return_sources=return_sources)
            result['batch_position'] = i
            results.append(result)
        
        total_time = time.time() - start_time
        print(f"\n📊 Batch completed in {total_time:.2f} seconds")
        print(f"   ⏱️ Average time per query: {total_time/len(questions):.2f} seconds")
        
        return results
    
    def get_conversation_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        if not self.enable_conversation:
            raise ValueError("Conversation mode is not enabled")
        
        return self.conversational_rag._get_conversation_history(thread_id)
    
    def clear_conversation(self, thread_id: str = "default") -> bool:
        """Clear conversation history for a thread"""
        if not self.enable_conversation:
            raise ValueError("Conversation mode is not enabled")
        
        return self.conversational_rag.clear_conversation(thread_id)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            'index_name': self.index_name,
            'include_web_search': self.include_web_search,
            'vector_top_k': self.vector_top_k,
            'web_search_results': self.web_search_results,
            'final_top_k': self.final_top_k,
            'enable_conversation': self.enable_conversation,
            'embedder_type': type(self.embedder).__name__,
            'reranker_type': type(self.reranker).__name__,
            'llm_generator_type': type(self.llm_generator).__name__ if self.llm_generator else 'ConversationalRAG'
        }


def create_rag_pipeline(
    index_name: str = "rag-docs",
    embedding_provider: str = "auto",
    reranker_provider: str = "auto",
    llm_provider: str = "auto",
    include_web_search: bool = True,
    vector_top_k: int = 10,
    web_search_results: int = 3,
    final_top_k: int = 5,
    enable_conversation: bool = False,
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline
    
    Args:
        index_name: Name of the vector store index
        embedding_provider: Embedding provider
        reranker_provider: Reranker provider
        llm_provider: LLM provider
        include_web_search: Whether to include web search
        vector_top_k: Number of vector search results
        web_search_results: Number of web search results
        final_top_k: Final number of results after reranking
        enable_conversation: Whether to enable conversational memory
        **kwargs: Additional arguments
        
    Returns:
        Configured RAG pipeline
    """
    return RAGPipeline(
        index_name=index_name,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        llm_provider=llm_provider,
        include_web_search=include_web_search,
        vector_top_k=vector_top_k,
        web_search_results=web_search_results,
        final_top_k=final_top_k,
        enable_conversation=enable_conversation,
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


def create_conversational_rag_pipeline(
    index_name: str = "rag-docs",
    llm_provider: str = "auto",
    max_messages: int = 10,
    enable_summary: bool = True,
    summary_threshold: int = 8,
    **kwargs
) -> RAGPipeline:
    """Create a conversational RAG pipeline with memory"""
    return create_rag_pipeline(
        index_name=index_name,
        llm_provider=llm_provider,
        enable_conversation=True,
        max_conversation_messages=max_messages,
        enable_summary=enable_summary,
        summary_threshold=summary_threshold,
        **kwargs
    ) 