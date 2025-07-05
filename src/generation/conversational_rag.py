from typing import List, Dict, Any, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from config import OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER
import time


class ConversationalRAG:
    """
    Conversational RAG system with memory using LangGraph
    
    Features:
    - Maintains conversation history
    - Automatic message trimming to manage context window
    - Summary memory for long conversations
    - Multi-turn conversations with context
    """
    
    def __init__(
        self,
        llm_provider: str = "auto",
        model: str = None,
        temperature: float = 0.0,
        max_messages: int = 10,
        enable_summary: bool = True,
        summary_threshold: int = 8
    ):
        """
        Initialize conversational RAG
        
        Args:
            llm_provider: LLM provider ("openai", "gemini", "auto")
            model: Specific model name
            temperature: Generation temperature
            max_messages: Maximum messages to keep in context
            enable_summary: Whether to use summary memory
            summary_threshold: Number of messages before summarizing
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_messages = max_messages
        self.enable_summary = enable_summary
        self.summary_threshold = summary_threshold
        
        # Initialize memory first (needed for graph compilation)
        self.memory = MemorySaver()
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
        
        # Create the conversational graph (uses self.memory)
        self.app = self._create_conversational_graph()
        
        print(f"✅ Conversational RAG initialized with {self.llm_provider} provider")
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider"""
        if self.llm_provider == "auto":
            # Try providers in order
            try:
                if OPENAI_API_KEY:
                    return ChatOpenAI(
                        model=self.model or "gpt-4o-mini",
                        temperature=self.temperature,
                        openai_api_key=OPENAI_API_KEY
                    )
            except:
                pass
            
            try:
                if GEMINI_API_KEY:
                    return ChatGoogleGenerativeAI(
                        model=self.model or "gemini-2.5-flash",
                        temperature=self.temperature,
                        google_api_key=GEMINI_API_KEY
                    )
            except:
                pass
            
            raise ValueError("No LLM provider available")
        
        elif self.llm_provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found")
            return ChatOpenAI(
                model=self.model or "gpt-4o-mini",
                temperature=self.temperature,
                openai_api_key=OPENAI_API_KEY
            )
        
        elif self.llm_provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found")
            return ChatGoogleGenerativeAI(
                model=self.model or "gemini-2.5-flash",
                temperature=self.temperature,
                google_api_key=GEMINI_API_KEY
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def _create_conversational_graph(self):
        """Create the conversational graph with memory management"""
        workflow = StateGraph(state_schema=MessagesState)
        
        # Define the function that calls the model
        def call_model(state: MessagesState):
            """Handle model calls with memory management"""
            try:
                # Get all messages except the last one (current user input)
                message_history = state["messages"][:-1]
                last_human_message = state["messages"][-1]
                
                # Create system prompt
                system_prompt = self._create_system_prompt()
                system_message = SystemMessage(content=system_prompt)
                
                # Handle memory management
                if self.enable_summary and len(message_history) >= self.summary_threshold:
                    # Use summary memory for long conversations
                    return self._handle_summary_memory(
                        system_message, message_history, last_human_message
                    )
                else:
                    # Use trimming for shorter conversations
                    return self._handle_trimmed_memory(
                        system_message, state["messages"]
                    )
            
            except Exception as e:
                # Return error message
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                return {"messages": [AIMessage(content=error_msg)]}
        
        # Add node and edge
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)
    
    def _create_system_prompt(self):
        """Create the system prompt for conversational RAG"""
        return """You are a helpful AI assistant that answers questions based on provided context documents and conversation history.

Instructions:
1. Use the provided context documents to answer questions when available
2. Reference previous conversation history when relevant
3. If no context is provided, use your general knowledge but mention this
4. Be conversational and remember what was discussed earlier
5. If you don't know something, admit it rather than guessing
6. Cite sources when using context documents
7. Ask clarifying questions when needed

The conversation history and context will help you provide more relevant and personalized responses."""
    
    def _handle_trimmed_memory(self, system_message: SystemMessage, messages: List[BaseMessage]):
        """Handle memory with message trimming"""
        # Create trimmer - keep last N messages
        trimmer = trim_messages(
            strategy="last",
            max_tokens=self.max_messages,
            token_counter=len
        )
        
        # Trim messages
        trimmed_messages = trimmer.invoke(messages)
        
        # Invoke model
        response = self.llm.invoke([system_message] + trimmed_messages)
        
        return {"messages": [response]}
    
    def _handle_summary_memory(self, system_message: SystemMessage, message_history: List[BaseMessage], last_human_message: BaseMessage):
        """Handle memory with summarization"""
        from langchain_core.messages import RemoveMessage
        
        # Create summary of conversation history
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can, including:\n"
            "- Main topics discussed\n"
            "- Key information provided\n"
            "- User preferences or context\n"
            "- Any ongoing tasks or questions\n"
            "Preserve important context for future reference."
        )
        
        summary_message = self.llm.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )
        
        # Delete old messages
        delete_messages = [RemoveMessage(id=m.id) for m in message_history]
        
        # Create new human message
        human_message = HumanMessage(content=last_human_message.content)
        
        # Generate response with summary
        response = self.llm.invoke([system_message, summary_message, human_message])
        
        return {"messages": [summary_message, human_message, response] + delete_messages}
    
    def chat(
        self,
        message: str,
        context_documents: List[Document] = None,
        thread_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message to the conversational RAG system
        
        Args:
            message: User message
            context_documents: Optional context documents for RAG
            thread_id: Conversation thread ID
            **kwargs: Additional arguments
            
        Returns:
            Response with conversation history
        """
        try:
            start_time = time.time()
            
            # Format context if provided
            context_info = ""
            if context_documents:
                context_info = self._format_context_for_conversation(context_documents)
                # Include context in the message
                enhanced_message = f"Context:\n{context_info}\n\nQuestion: {message}"
            else:
                enhanced_message = message
            
            # Create human message
            human_message = HumanMessage(content=enhanced_message)
            
            # Invoke the conversational app
            response = self.app.invoke(
                {"messages": [human_message]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Extract the latest AI response
            ai_response = response["messages"][-1].content
            
            # Get conversation history
            conversation_history = self._get_conversation_history(thread_id)
            
            return {
                'success': True,
                'response': ai_response,
                'thread_id': thread_id,
                'conversation_history': conversation_history,
                'context_documents': len(context_documents) if context_documents else 0,
                'sources': self._extract_sources(context_documents) if context_documents else [],
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'thread_id': thread_id,
                'processing_time': time.time() - start_time
            }
    
    def _format_context_for_conversation(self, documents: List[Document]) -> str:
        """Format context documents for conversational use"""
        if not documents:
            return ""
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', f'Document {i}')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            formatted_doc = f"[Source {i}] {title} ({source}):\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents with comprehensive metadata"""
        if not documents:
            return []
        
        sources = []
        for i, doc in enumerate(documents, 1):
            # Extract comprehensive metadata
            source_info = {
                'rank': i,
                'title': doc.metadata.get('title', doc.metadata.get('filename', f'Document {i}')),
                'source': doc.metadata.get('source', doc.metadata.get('file_path', 'Unknown')),
                'url': doc.metadata.get('url', ''),
                'chunk_id': doc.metadata.get('chunk_id', ''),
                'retrieval_source': doc.metadata.get('retrieval_source', 'unknown'),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                # Include scoring information if available
                'similarity_score': doc.metadata.get('score', doc.metadata.get('similarity_score')),
                'rerank_score': doc.metadata.get('rerank_score'),
                # Include additional metadata
                'category': doc.metadata.get('category'),
                'department': doc.metadata.get('department'),
                'document_type': doc.metadata.get('document_type', doc.metadata.get('file_type')),
                'upload_date': doc.metadata.get('upload_timestamp', doc.metadata.get('created_date')),
                'chunk_index': doc.metadata.get('chunk_index', doc.metadata.get('chunk_number')),
                'total_chunks': doc.metadata.get('total_chunks'),
                'priority': doc.metadata.get('priority'),
                'version': doc.metadata.get('version'),
            }
            
            # Clean up None values
            source_info = {k: v for k, v in source_info.items() if v is not None}
            
            sources.append(source_info)
        
        return sources
    
    def _get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        try:
            # Get the current state
            current_state = self.app.get_state({"configurable": {"thread_id": thread_id}})
            
            # Extract messages and format for display
            messages = current_state.values.get("messages", [])
            
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({
                        'role': 'user',
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', None)
                    })
                elif isinstance(msg, AIMessage):
                    history.append({
                        'role': 'assistant',
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', None)
                    })
            
            return history
            
        except Exception as e:
            print(f"⚠️ Error getting conversation history: {str(e)}")
            return []
    
    def clear_conversation(self, thread_id: str) -> bool:
        """Clear conversation history for a thread"""
        try:
            # Create a new empty state
            self.app.update_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": []}
            )
            return True
        except Exception as e:
            print(f"⚠️ Error clearing conversation: {str(e)}")
            return False
    
    def get_active_threads(self) -> List[str]:
        """Get list of active conversation threads"""
        try:
            # This is a simplified version - in a real app you'd track threads
            return ["default"]
        except Exception as e:
            print(f"⚠️ Error getting active threads: {str(e)}")
            return []


def create_conversational_rag(
    llm_provider: str = "auto",
    model: str = None,
    temperature: float = 0.0,
    max_messages: int = 10,
    enable_summary: bool = True,
    summary_threshold: int = 8
) -> ConversationalRAG:
    """
    Factory function to create a conversational RAG system
    
    Args:
        llm_provider: LLM provider ("openai", "gemini", "auto")
        model: Specific model name
        temperature: Generation temperature
        max_messages: Maximum messages to keep in context
        enable_summary: Whether to use summary memory
        summary_threshold: Number of messages before summarizing
        
    Returns:
        ConversationalRAG instance
    """
    return ConversationalRAG(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        max_messages=max_messages,
        enable_summary=enable_summary,
        summary_threshold=summary_threshold
    ) 