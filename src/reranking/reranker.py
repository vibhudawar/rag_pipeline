from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
import os


class Reranker(ABC):
    """Abstract base class for document rerankers"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on relevance to query"""
        pass


class CohereReranker(Reranker):
    """Cohere reranker implementation using LangChain"""
    
    def __init__(self, model: str = "rerank-english-v3.0"):
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        self.reranker = CohereRerank(
                cohere_api_key=cohere_api_key,
                model=model,
                top_k=10  # We'll limit this in the rerank method
            )
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using Cohere API"""
        try:
            if not documents:
                return []
            
            reranked_docs = self.reranker.compress_documents(documents, query)
            for i, doc in enumerate(reranked_docs[:top_k]):
                doc.metadata['rerank_position'] = i + 1
                doc.metadata['rerank_model'] = 'cohere-langchain'
                
            return reranked_docs[:top_k]
            
        except Exception as e:
            # If reranking fails, return original documents
            print(f"⚠️ Cohere reranking failed: {str(e)}, returning original order")
            return documents[:top_k]


class HuggingFaceReranker(Reranker):
    """HuggingFace cross-encoder reranker implementation"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
        except ImportError:
            raise ImportError("sentence-transformers is required for HuggingFace reranker. Install with: pip install sentence-transformers")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using HuggingFace cross-encoder"""
        try:
            if not documents:
                return []
            
            # Create query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Score all pairs
            scores = self.model.predict(pairs)
            
            # Sort documents by score (descending)
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k documents
            reranked_docs = [doc for doc, score in scored_docs[:top_k]]
            
            # Add reranking scores to metadata
            for i, (doc, score) in enumerate(scored_docs[:top_k]):
                doc.metadata['rerank_score'] = float(score)
                doc.metadata['rerank_position'] = i + 1
                doc.metadata['rerank_model'] = self.model_name
            
            return reranked_docs
            
        except Exception as e:
            print(f"⚠️ HuggingFace reranking failed: {str(e)}, returning original order")
            return documents[:top_k]


class NoOpReranker(Reranker):
    """No-operation reranker that returns documents in original order"""
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Return documents in original order"""
        return documents[:top_k]


class ScoreBasedReranker(Reranker):
    """Simple reranker that sorts by existing similarity scores"""
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents by existing similarity scores"""
        try:
            # Sort by score if available in metadata
            scored_docs = []
            for doc in documents:
                score = doc.metadata.get('score', doc.metadata.get('similarity_score', 0.0))
                scored_docs.append((doc, score))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k documents
            return [doc for doc, score in scored_docs[:top_k]]
            
        except Exception as e:
            print(f"⚠️ Score-based reranking failed: {str(e)}, returning original order")
            return documents[:top_k]


def get_reranker(provider: str = "auto") -> Reranker:
    """Factory function to get the appropriate reranker"""
    
    if provider == "cohere":
        try:
            return CohereReranker()
        except (ValueError, ImportError) as e:
            print(f"⚠️ Cohere reranker not available: {str(e)}")
    
    elif provider == "huggingface":
        try:
            return HuggingFaceReranker()
        except (ImportError, Exception) as e:
            print(f"⚠️ HuggingFace reranker not available: {str(e)}")
    
    elif provider == "score":
        return ScoreBasedReranker()
    
    elif provider == "none":
        return NoOpReranker()
    
    elif provider == "auto":
        # Try providers in order of preference
        
        # 1. Try Cohere first (best quality)
        try:
            return CohereReranker()
        except (ValueError, ImportError):
            pass
        
        # 2. Try HuggingFace (good quality, local)
        try:
            return HuggingFaceReranker()
        except (ImportError, Exception):
            pass
        
        # 3. Fall back to score-based
        print("ℹ️ Using score-based reranker")
        return ScoreBasedReranker()
    
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")


def rerank_documents(query: str, documents: List[Document], top_k: int = 5, provider: str = "auto") -> List[Document]:
    """Convenience function to rerank documents"""
    reranker = get_reranker(provider)
    return reranker.rerank(query, documents, top_k)
