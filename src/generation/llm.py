from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from config import OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER


class LLMGenerator(ABC):
    """Abstract base class for LLM generators"""
    
    @abstractmethod
    def generate(self, query: str, context_documents: List[Document], **kwargs) -> Dict[str, Any]:
        """Generate response based on query and context documents"""
        pass


class OpenAIGenerator(LLMGenerator):
    """OpenAI LLM generator using LangChain"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        self.model_name = model
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context. 
            
Instructions:
1. Use ONLY the information provided in the context to answer the question
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite the sources by mentioning the document titles when possible
4. Be concise but comprehensive in your response
5. If conflicting information exists in the context, acknowledge it

Context:
{context}

Question: {question}

Please provide a helpful and accurate answer based on the context above."""),
            ("human", "{question}")
        ])
        
        # Create the chain
        self.chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate(self, query: str, context_documents: List[Document], **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        try:
            # Format context
            context = self._format_context(context_documents)
            
            # Generate response
            response = self.chain.invoke({
                "context": context,
                "question": query
            })
            
            return {
                'success': True,
                'response': response,
                'model': self.model_name,
                'context_documents': len(context_documents),
                'sources': self._extract_sources(context_documents)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': self.model_name,
                'context_documents': len(context_documents)
            }
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format context documents for the prompt"""
        if not documents:
            return "No context documents provided."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', f'Document {i}')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            formatted_doc = f"Source {i}: {title} ({source})\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for doc in documents:
            source_info = {
                'title': doc.metadata.get('title', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'url': doc.metadata.get('url', ''),
                'chunk_id': doc.metadata.get('chunk_id', '')
            }
            sources.append(source_info)
        return sources


class GeminiGenerator(LLMGenerator):
    """Google Gemini LLM generator using LangChain"""
    
    def __init__(self, model: str = "gemini-pro", temperature: float = 0.0):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY
        )
        self.model_name = model
        
        # Create prompt template  
        self.prompt = PromptTemplate(
            template="""You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
1. Use ONLY the information provided in the context to answer the question
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite the sources by mentioning the document titles when possible
4. Be concise but comprehensive in your response
5. If conflicting information exists in the context, acknowledge it

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def generate(self, query: str, context_documents: List[Document], **kwargs) -> Dict[str, Any]:
        """Generate response using Gemini"""
        try:
            # Format context
            context = self._format_context(context_documents)
            
            # Generate response
            response = self.chain.invoke({
                "context": context,
                "question": query
            })
            
            return {
                'success': True,
                'response': response,
                'model': self.model_name,
                'context_documents': len(context_documents),
                'sources': self._extract_sources(context_documents)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': self.model_name,
                'context_documents': len(context_documents)
            }
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format context documents for the prompt"""
        if not documents:
            return "No context documents provided."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', f'Document {i}')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            formatted_doc = f"Source {i}: {title} ({source})\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for doc in documents:
            source_info = {
                'title': doc.metadata.get('title', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'url': doc.metadata.get('url', ''),
                'chunk_id': doc.metadata.get('chunk_id', '')
            }
            sources.append(source_info)
        return sources


class MockGenerator(LLMGenerator):
    """Mock generator for testing when no API keys are available"""
    
    def generate(self, query: str, context_documents: List[Document], **kwargs) -> Dict[str, Any]:
        """Generate mock response"""
        # Create a simple mock response based on context
        context_info = []
        for i, doc in enumerate(context_documents[:3], 1):
            title = doc.metadata.get('title', f'Document {i}')
            preview = doc.page_content[:100] + "..."
            context_info.append(f"- {title}: {preview}")
        
        mock_response = f"""Based on the provided context documents, I can see information about:

{chr(10).join(context_info)}

This is a mock response for the query: "{query}"

In a real implementation, this would be generated by an actual language model using the context documents to provide accurate and relevant answers.

Note: This is a demonstration response since no LLM API key is configured."""
        
        return {
            'success': True,
            'response': mock_response,
            'model': 'mock-generator',
            'context_documents': len(context_documents),
            'sources': self._extract_sources(context_documents)
        }
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        for doc in documents:
            source_info = {
                'title': doc.metadata.get('title', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'url': doc.metadata.get('url', ''),
                'chunk_id': doc.metadata.get('chunk_id', '')
            }
            sources.append(source_info)
        return sources


def get_llm_generator(provider: str = None, model: str = None, **kwargs) -> LLMGenerator:
    """Factory function to get the appropriate LLM generator"""
    
    if provider is None:
        provider = LLM_PROVIDER or "auto"
    
    if provider == "openai":
        try:
            model = model or "gpt-3.5-turbo"
            return OpenAIGenerator(model=model, **kwargs)
        except ValueError as e:
            print(f"⚠️ OpenAI generator not available: {str(e)}")
    
    elif provider == "gemini":
        try:
            model = model or "gemini-pro"
            return GeminiGenerator(model=model, **kwargs)
        except ValueError as e:
            print(f"⚠️ Gemini generator not available: {str(e)}")
    
    elif provider == "mock":
        return MockGenerator()
    
    elif provider == "auto":
        # Try providers in order of preference
        
        # 1. Try OpenAI first
        try:
            return OpenAIGenerator(**kwargs)
        except ValueError:
            pass
        
        # 2. Try Gemini
        try:
            return GeminiGenerator(**kwargs)
        except ValueError:
            pass
        
        # 3. Fall back to mock
        print("⚠️ No LLM API keys available, using mock generator")
        return MockGenerator()
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def generate_response(query: str, context_documents: List[Document], provider: str = None, **kwargs) -> Dict[str, Any]:
    """Convenience function to generate response"""
    generator = get_llm_generator(provider, **kwargs)
    return generator.generate(query, context_documents, **kwargs) 