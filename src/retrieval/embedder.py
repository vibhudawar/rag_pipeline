from abc import ABC, abstractmethod
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY


class Embedder(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        pass


class OpenAIEmbedder(Embedder):
    """OpenAI embedding implementation using LangChain"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=OPENAI_API_KEY
        )
        self.model = model
        # Set dimension based on model
        self._dimension = 1536 if model == "text-embedding-3-small" else 3072
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LangChain OpenAI wrapper"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        return self._dimension


class GeminiEmbedder(Embedder):
    """Gemini embedding implementation using LangChain"""
    
    def __init__(self, model: str = "models/text-embedding-004"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=GEMINI_API_KEY
        )
        self.model = model
        self._dimension = 768  # Gemini embedding dimension
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LangChain Gemini wrapper"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Failed to generate Gemini embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        return self._dimension


def get_embedder() -> Embedder:
    """Factory function to get the appropriate embedder based on config"""
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbedder()
    elif EMBEDDING_PROVIDER == "gemini":
        return GeminiEmbedder()
    else:
        raise ValueError(f"Invalid embedding provider: {EMBEDDING_PROVIDER}")