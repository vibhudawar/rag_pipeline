from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import tiktoken


class Chunker(ABC):
    """Abstract base class for text chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        pass


class RecursiveChunker(Chunker):
    """Recursive character text splitter chunker"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text using recursive character splitting"""
        if metadata is None:
            metadata = {}
        
        chunks = self.splitter.split_text(text)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            })
            
            chunked_documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return chunked_documents


class TokenChunker(Chunker):
    """Token-based text chunker using tiktoken"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, 
                 encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text using token-based splitting"""
        if metadata is None:
            metadata = {}
        
        chunks = self.splitter.split_text(text)
        
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk),
                'token_count': len(self.encoding.encode(chunk)),
                'total_chunks': len(chunks)
            })
            
            chunked_documents.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return chunked_documents


def get_chunker(strategy: str = "recursive", **kwargs) -> Chunker:
    """Factory function to get appropriate chunker"""
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    elif strategy == "token":
        return TokenChunker(**kwargs)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")