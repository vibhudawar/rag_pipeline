from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_community.utilities import SerpAPIWrapper, BraveSearchWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
import os
from config import SERPAPI_KEY, LLM_PROVIDER


class WebSearcher(ABC):
    """Abstract base class for web search providers"""
    
    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web and return results"""
        pass


class SerpAPISearcher(WebSearcher):
    """SerpAPI web search implementation using LangChain"""
    
    def __init__(self):
        if not SERPAPI_KEY:
            raise ValueError("SERPAPI_KEY not found in environment variables")
        self.search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using SerpAPI"""
        try:
            # SerpAPI returns a string, we need to parse it
            raw_result = self.search_tool.run(query)
            
            # For now, create a single result from the raw response
            # In production, you might want to parse this more carefully
            results = [{
                'title': 'Web Search Result',
                'snippet': raw_result[:500] + "..." if len(raw_result) > 500 else raw_result,
                'url': 'https://search-result.com',
                'source': 'serpapi'
            }]
            
            return results[:num_results]
            
        except Exception as e:
            raise RuntimeError(f"SerpAPI search failed: {str(e)}")


class DuckDuckGoSearcher(WebSearcher):
    """DuckDuckGo web search implementation using LangChain"""
    
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            raw_result = self.search_tool.run(query)
            
            # DuckDuckGo returns a string, create a structured result
            results = [{
                'title': f'DuckDuckGo Search: {query}',
                'snippet': raw_result[:500] + "..." if len(raw_result) > 500 else raw_result,
                'url': 'https://duckduckgo.com',
                'source': 'duckduckgo'
            }]
            
            return results[:num_results]
            
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo search failed: {str(e)}")


class MockWebSearcher(WebSearcher):
    """Mock web searcher for testing when no API keys are available"""
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Return mock search results"""
        mock_results = [
            {
                'title': f'Mock Result 1 for "{query}"',
                'snippet': f'This is a mock search result for the query "{query}". In a real implementation, this would contain actual web search results.',
                'url': 'https://example.com/result1',
                'source': 'mock'
            },
            {
                'title': f'Mock Result 2 for "{query}"',
                'snippet': f'Another mock result demonstrating web search functionality for "{query}". Real results would come from search APIs.',
                'url': 'https://example.com/result2',
                'source': 'mock'
            }
        ]
        
        return mock_results[:num_results]


def get_web_searcher() -> WebSearcher:
    """Factory function to get the appropriate web searcher"""
    
    # Try SerpAPI first if available
    if SERPAPI_KEY:
        try:
            return SerpAPISearcher()
        except (ValueError, ImportError) as e:
            print(f"⚠️ SerpAPI not available: {str(e)}")
    
    # Fall back to DuckDuckGo (free, no API key required)
    try:
        return DuckDuckGoSearcher()
    except ImportError as e:
        print(f"⚠️ DuckDuckGo search not available: {str(e)}")
    
    # Ultimate fallback to mock searcher
    print("⚠️ No web search API available, using mock searcher")
    return MockWebSearcher()


def search_web(query: str, num_results: int = 5) -> List[Document]:
    """Convenience function to search web and return LangChain documents"""
    searcher = get_web_searcher()
    results = searcher.search(query, num_results)
    
    documents = []
    for result in results:
        doc = Document(
            page_content=result['snippet'],
            metadata={
                'title': result['title'],
                'url': result['url'],
                'source': result['source'],
                'search_query': query
            }
        )
        documents.append(doc)
    
    return documents
