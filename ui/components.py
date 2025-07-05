import streamlit as st
import time
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


def display_ingestion_result(result: Dict[str, Any], show_metadata: bool = False):
    """Display the results of document ingestion in a nice format"""
    
    if result['success']:
        st.success("✅ Document ingested successfully!")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filename", result.get('filename', 'Unknown'))
        
        with col2:
            st.metric("File Type", result.get('file_type', 'Unknown').upper())
        
        with col3:
            st.metric("Total Chunks", result.get('total_chunks', 0))
        
        with col4:
            st.metric("Processing Time", f"{result.get('processing_time_seconds', 0):.2f}s")
        
        # Show metadata if requested
        if show_metadata and 'metadata' in result:
            with st.expander("📋 Document Metadata"):
                metadata = result['metadata']
                for key, value in metadata.items():
                    st.text(f"{key}: {value}")
    
    else:
        st.error(f"❌ Failed to ingest document: {result.get('error', 'Unknown error')}")
        if 'processing_time_seconds' in result:
            st.info(f"⏱️ Processing time: {result['processing_time_seconds']:.2f}s")


def display_batch_results(results: List[Dict[str, Any]]):
    """Display batch processing results with summary statistics"""
    
    if not results:
        st.warning("No results to display")
        return
    
    # Calculate summary statistics
    total_files = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total_files - successful
    total_chunks = sum(r.get('total_chunks', 0) for r in results if r.get('success', False))
    total_time = sum(r.get('processing_time_seconds', 0) for r in results)
    
    # Display summary metrics
    st.subheader("📊 Batch Processing Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Files", total_files)
    
    with col2:
        st.metric("Successful", successful, delta=f"{(successful/total_files*100):.1f}%")
    
    with col3:
        st.metric("Failed", failed, delta=f"{(failed/total_files*100):.1f}%" if failed > 0 else None)
    
    with col4:
        st.metric("Total Chunks", total_chunks)
    
    with col5:
        st.metric("Total Time", f"{total_time:.2f}s")
    
    # Create a detailed results table
    st.subheader("📋 Detailed Results")
    
    # Prepare data for the table
    table_data = []
    for result in results:
        table_data.append({
            'Status': '✅' if result.get('success', False) else '❌',
            'Filename': result.get('filename', 'Unknown'),
            'File Type': result.get('file_type', 'Unknown').upper(),
            'Chunks': result.get('total_chunks', 0) if result.get('success', False) else 0,
            'Time (s)': f"{result.get('processing_time_seconds', 0):.2f}",
            'Error': result.get('error', '') if not result.get('success', False) else ''
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    # Create a visualization if there are multiple files
    if total_files > 1:
        st.subheader("📈 Processing Visualization")
        
        # Success/Failure pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            if successful > 0 or failed > 0:
                fig_pie = px.pie(
                    values=[successful, failed],
                    names=['Successful', 'Failed'],
                    title="Success Rate",
                    color_discrete_map={'Successful': '#00ff00', 'Failed': '#ff0000'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Chunks per file bar chart
            if successful > 0:
                successful_results = [r for r in results if r.get('success', False)]
                if len(successful_results) > 1:
                    fig_bar = px.bar(
                        x=[r.get('filename', 'Unknown')[:20] for r in successful_results],
                        y=[r.get('total_chunks', 0) for r in successful_results],
                        title="Chunks per File",
                        labels={'x': 'Filename', 'y': 'Number of Chunks'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)


def show_pipeline_config(stats: Dict[str, Any]):
    """Display current pipeline configuration"""
    
    st.subheader("⚙️ Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Index Name:** {stats.get('index_name', 'Unknown')}")
        st.info(f"**Vector Store:** {stats.get('vector_store_type', 'Unknown')}")
        st.info(f"**Embedder:** {stats.get('embedder_type', 'Unknown')}")
    
    with col2:
        st.info(f"**Chunking Strategy:** {stats.get('chunking_strategy', 'Unknown')}")
        st.info(f"**Chunk Size:** {stats.get('chunk_size', 'Unknown')}")
        st.info(f"**Chunk Overlap:** {stats.get('chunk_overlap', 'Unknown')}")


def create_progress_bar(message: str = "Processing..."):
    """Create a progress bar for long operations"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(message)
    return progress_bar, status_text


def update_progress_bar(progress_bar, status_text, progress: float, message: str):
    """Update progress bar and status message"""
    progress_bar.progress(progress)
    status_text.text(message)


def show_supported_file_types():
    """Display supported file types"""
    st.sidebar.markdown("### 📁 Supported File Types")
    st.sidebar.markdown("""
    - **PDF** (.pdf)
    - **Word Documents** (.docx)
    - **Text Files** (.txt)
    - **Markdown** (.md)
    """)


def initialize_sidebar_config():
    """Initialize sidebar configuration once and store in session state"""
    
    # Only initialize if not already in session state
    if 'sidebar_initialized' not in st.session_state:
        st.session_state.sidebar_initialized = True
        
        # Show supported file types
        show_supported_file_types()
        
        st.sidebar.markdown("### ⚙️ Configuration")
        
        # Initialize default values if not in session state
        if 'config_index_name' not in st.session_state:
            st.session_state.config_index_name = "rag-documents"
        if 'config_chunking_strategy' not in st.session_state:
            st.session_state.config_chunking_strategy = "recursive"
        if 'config_chunk_size' not in st.session_state:
            st.session_state.config_chunk_size = 1000
        if 'config_chunk_overlap' not in st.session_state:
            st.session_state.config_chunk_overlap = 200


def get_sidebar_config():
    """Get configuration from sidebar widgets with unique keys"""
    
    # Initialize sidebar if needed
    initialize_sidebar_config()
    
    # Create sidebar widgets with unique keys
    index_name = st.sidebar.text_input(
        "Index Name",
        value=st.session_state.config_index_name,
        help="Name of the Pinecone index to use",
        key="sidebar_index_name"
    )
    
    chunking_strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        options=["recursive", "token"],
        index=0 if st.session_state.config_chunking_strategy == "recursive" else 1,
        help="How to split documents into chunks",
        key="sidebar_chunking_strategy"
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=st.session_state.config_chunk_size,
        step=100,
        help="Maximum characters per chunk",
        key="sidebar_chunk_size"
    )
    
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=st.session_state.config_chunk_overlap,
        step=50,
        help="Character overlap between chunks",
        key="sidebar_chunk_overlap"
    )
    
    # Update session state with current values
    st.session_state.config_index_name = index_name
    st.session_state.config_chunking_strategy = chunking_strategy
    st.session_state.config_chunk_size = chunk_size
    st.session_state.config_chunk_overlap = chunk_overlap
    
    return {
        'index_name': index_name,
        'chunking_strategy': chunking_strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }


def create_config_sidebar():
    """Legacy function for backward compatibility - now uses session state approach"""
    return get_sidebar_config()


def display_search_results(results: List[Dict[str, Any]], query: str):
    """Display search results in a nice format"""
    
    if not results:
        st.warning(f"No results found for query: '{query}'")
        return
    
    st.success(f"Found {len(results)} results for: '{query}'")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"📄 Result {i} (Score: {result.get('score', 0):.3f})"):
            # Display the text content
            st.markdown("**Content:**")
            st.text_area(
                label="",
                value=result.get('text', 'No content available'),
                height=150,
                disabled=True,
                key=f"search_result_{i}_{hash(query)}"  # Unique key with query hash
            )
            
            # Display metadata
            if 'metadata' in result and result['metadata']:
                st.markdown("**Metadata:**")
                metadata = result['metadata']
                
                # Create columns for metadata
                cols = st.columns(min(len(metadata), 3))
                for idx, (key, value) in enumerate(metadata.items()):
                    with cols[idx % 3]:
                        st.text(f"{key}: {value}")


def show_error_message(error: Exception, context: str = ""):
    """Display error messages in a user-friendly way"""
    
    error_msg = str(error)
    
    # Common error patterns and solutions
    solutions = {
        "OPENAI_API_KEY": "Please set your OpenAI API key in the .env file",
        "PINECONE_API_KEY": "Please set your Pinecone API key in the .env file",
        "Failed to create Pinecone index": "Check your Pinecone API key and account limits",
        "Failed to generate": "Check your API keys and internet connection",
        "Unsupported file type": "This file type is not supported. Try PDF, DOCX, TXT, or MD files"
    }
    
    st.error(f"❌ {context}: {error_msg}")
    
    # Show relevant solution if available
    for pattern, solution in solutions.items():
        if pattern in error_msg:
            st.info(f"💡 **Suggested Solution:** {solution}")
            break
    
    # Show generic help
    with st.expander("🔧 Troubleshooting Tips"):
        st.markdown("""
        1. **Check your .env file** - Make sure all required API keys are set
        2. **Verify file format** - Only PDF, DOCX, TXT, and MD files are supported
        3. **Check internet connection** - API calls require internet access
        4. **Review API limits** - You might have hit rate limits
        5. **Check file size** - Very large files might timeout
        """)


def create_file_uploader(accept_multiple_files: bool = False, key: str = None):
    """Create a file uploader with supported file types"""
    
    return st.file_uploader(
        "Choose file(s)" if accept_multiple_files else "Choose a file",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=accept_multiple_files,
        key=key,
        help="Supported formats: PDF, DOCX, TXT, MD"
    )


def display_rag_response(response: Dict[str, Any]):
    """Display complete RAG response with answer and sources"""
    
    if not response.get('success', False):
        st.error(f"❌ Query failed: {response.get('error', 'Unknown error')}")
        return
    
    # Main answer
    st.subheader("🤖 Generated Answer")
    
    # Answer content
    answer = response.get('answer', '')
    if answer:
        st.markdown(answer)
    else:
        st.warning("No answer generated")
        return
    
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model = response.get('model', 'Unknown')
        st.metric("Model Used", model)
    
    with col2:
        processing_time = response.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col3:
        retrieval_stats = response.get('retrieval_stats', {})
        total_retrieved = retrieval_stats.get('total_retrieved', 0)
        st.metric("Documents Retrieved", total_retrieved)
    
    with col4:
        final_context = retrieval_stats.get('final_context', 0)
        st.metric("Context Documents", final_context)
    
    # Retrieval breakdown
    if retrieval_stats:
        st.subheader("📊 Retrieval Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vector_count = retrieval_stats.get('vector_search', 0)
            st.metric("Vector Search", vector_count)
        
        with col2:
            web_count = retrieval_stats.get('web_search', 0)
            st.metric("Web Search", web_count)
        
        with col3:
            if vector_count + web_count > 0:
                ratio = web_count / (vector_count + web_count) * 100
                st.metric("Web Search %", f"{ratio:.1f}%")
    
    # Sources
    sources = response.get('sources', [])
    if sources:
        st.subheader("📚 Source Documents")
        
        # Add styling for better scrollable display
        st.markdown("""
        <style>
        .rag-source-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Add scrollable container for sources
        st.markdown('<div class="rag-source-container">', unsafe_allow_html=True)
        
        for source in sources:
            with st.expander(f"#{source['rank']} {source['title']} ({source['retrieval_source']})"):
                
                # Source metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text(f"Source: {source['source']}")
                    if source.get('url'):
                        st.link_button("🔗 View Source", source['url'], key=f"rag_source_link_{source['rank']}")
                
                with col2:
                    if source.get('rerank_score') is not None:
                        st.metric("Rerank Score", f"{source['rerank_score']:.4f}")
                        
                    if source.get('similarity_score') is not None:
                        st.metric("Similarity", f"{source['similarity_score']:.4f}")
                
                # Content preview in scrollable text area
                st.markdown("**Content Preview:**")
                st.text_area(
                    label="",
                    value=source['content_preview'],
                    height=120,
                    disabled=True,
                    key=f"rag_source_content_{source['rank']}_{hash(source['content_preview'])}"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)


def display_rag_batch_results(results: List[Dict[str, Any]]):
    """Display results from batch RAG queries"""
    
    if not results:
        st.warning("No results to display")
        return
    
    # Summary statistics
    st.subheader("📊 Batch Query Summary")
    
    successful_queries = sum(1 for r in results if r.get('success', False))
    total_time = sum(r.get('processing_time', 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(results))
    
    with col2:
        st.metric("Successful", successful_queries)
    
    with col3:
        st.metric("Total Time", f"{total_time:.2f}s")
    
    with col4:
        st.metric("Avg Time/Query", f"{avg_time:.2f}s")
    
    # Individual results
    st.subheader("📋 Individual Results")
    
    for i, result in enumerate(results):
        question = result.get('question', f'Query {i+1}')
        success = result.get('success', False)
        status_icon = "✅" if success else "❌"
        
        with st.expander(f"{status_icon} {question}"):
            if success:
                # Show answer preview
                answer = result.get('answer', '')[:200]
                if len(result.get('answer', '')) > 200:
                    answer += "..."
                
                st.markdown(f"**Answer:** {answer}")
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    model = result.get('model', 'Unknown')
                    st.text(f"Model: {model}")
                
                with col2:
                    processing_time = result.get('processing_time', 0)
                    st.text(f"Time: {processing_time:.2f}s")
                
                with col3:
                    sources_count = len(result.get('sources', []))
                    st.text(f"Sources: {sources_count}")
                
            else:
                error = result.get('error', 'Unknown error')
                st.error(f"Error: {error}")


def create_rag_query_interface():
    """Create RAG query interface components"""
    
    st.subheader("🤖 RAG Query Interface")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., 'What are the key benefits of machine learning?'",
            help="Ask questions about your ingested documents",
            key="rag_query_input",
            height=100
        )
    
    with col2:
        st.markdown("**Options:**")
        
        # New conversational option
        enable_conversation = st.checkbox(
            "Enable conversation memory",
            value=False,
            help="Remember conversation history and context",
            key="rag_enable_conversation"
        )
        
        include_web_search = st.checkbox(
            "Include web search",
            value=True,
            help="Add web search results to enhance context",
            key="rag_include_web"
        )
        
        show_sources = st.checkbox(
            "Show sources",
            value=True,
            help="Display source documents used for the answer",
            key="rag_show_sources"
        )
        
        reranker_provider = st.selectbox(
            "Reranker",
            ["auto", "cohere", "huggingface", "score", "none"],
            help="Choose reranking method for better results",
            key="rag_reranker"
        )
        
        llm_provider = st.selectbox(
            "LLM Provider",
            ["auto", "openai", "gemini", "mock"],
            help="Choose language model for generation",
            key="rag_llm_provider"
        )
        
        # Conversation settings (shown when conversation is enabled)
        if enable_conversation:
            st.markdown("**Conversation Settings:**")
            
            max_messages = st.slider(
                "Max messages to remember",
                min_value=4,
                max_value=20,
                value=10,
                help="Maximum number of messages to keep in memory",
                key="rag_max_messages"
            )
            
            enable_summary = st.checkbox(
                "Enable summary memory",
                value=True,
                help="Use summary for long conversations",
                key="rag_enable_summary"
            )
            
            summary_threshold = st.slider(
                "Summary threshold",
                min_value=4,
                max_value=16,
                value=8,
                help="Number of messages before summarizing",
                key="rag_summary_threshold"
            )
        else:
            max_messages = 10
            enable_summary = True
            summary_threshold = 8
    
    return {
        'query': query,
        'enable_conversation': enable_conversation,
        'include_web_search': include_web_search,
        'show_sources': show_sources,
        'reranker_provider': reranker_provider,
        'llm_provider': llm_provider,
        'max_messages': max_messages,
        'enable_summary': enable_summary,
        'summary_threshold': summary_threshold
    }


def create_rag_batch_interface():
    """Create batch RAG query interface"""
    
    st.subheader("📊 Batch RAG Queries")
    
    # Questions input
    questions_text = st.text_area(
        "Enter questions (one per line):",
        placeholder="What is machine learning?\nHow does AI work?\nExplain neural networks?",
        help="Enter multiple questions, one per line",
        key="rag_batch_questions",
        height=150
    )
    
    # Parse questions
    questions = []
    if questions_text:
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
    # Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_web_search = st.checkbox(
            "Include web search",
            value=True,
            key="rag_batch_web"
        )
    
    with col2:
        show_sources = st.checkbox(
            "Show sources",
            value=False,
            help="Sources can make results very long",
            key="rag_batch_sources"
        )
    
    with col3:
        reranker_provider = st.selectbox(
            "Reranker",
            ["auto", "cohere", "huggingface", "score", "none"],
            key="rag_batch_reranker"
        )
    
    if questions:
        st.info(f"📊 Ready to process {len(questions)} questions")
    
    return {
        'questions': questions,
        'include_web_search': include_web_search,
        'show_sources': show_sources,
        'reranker_provider': reranker_provider
    }


def show_rag_pipeline_status(pipeline_info: Dict[str, Any]):
    """Show RAG pipeline configuration and status"""
    
    st.subheader("🔧 RAG Pipeline Configuration")
    
    # Pipeline overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Settings:**")
        st.text(f"Index: {pipeline_info.get('index_name', 'Unknown')}")
        st.text(f"Web Search: {'✅' if pipeline_info.get('include_web_search', False) else '❌'}")
        st.text(f"Vector Top-K: {pipeline_info.get('vector_top_k', 'Unknown')}")
        st.text(f"Final Top-K: {pipeline_info.get('final_top_k', 'Unknown')}")
    
    with col2:
        st.markdown("**Component Types:**")
        st.text(f"Embedder: {pipeline_info.get('embedder_type', 'Unknown')}")
        st.text(f"Reranker: {pipeline_info.get('reranker_type', 'Unknown')}")
        st.text(f"Generator: {pipeline_info.get('llm_generator_type', 'Unknown')}")
    
    # Performance settings
    if pipeline_info.get('web_search_results'):
        st.markdown("**Retrieval Settings:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vector Results", pipeline_info.get('vector_top_k', 0))
        
        with col2:
            st.metric("Web Results", pipeline_info.get('web_search_results', 0))
        
        with col3:
            st.metric("Final Results", pipeline_info.get('final_top_k', 0))


def create_conversational_chat_interface():
    """Create dedicated conversational chat interface"""
    
    st.subheader("💬 Conversational RAG Chat")
    
    # Initialize session state for conversation
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = "default"
    
    # Chat configuration
    with st.expander("🔧 Chat Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            thread_id = st.text_input(
                "Thread ID",
                value=st.session_state.current_thread_id,
                help="Unique identifier for this conversation",
                key="chat_thread_id"
            )
            
            max_messages = st.slider(
                "Max messages to remember",
                min_value=4,
                max_value=20,
                value=10,
                help="Maximum number of messages to keep in memory",
                key="chat_max_messages"
            )
            
            enable_summary = st.checkbox(
                "Enable summary memory",
                value=True,
                help="Use summary for long conversations",
                key="chat_enable_summary"
            )
        
        with col2:
            include_web_search = st.checkbox(
                "Include web search",
                value=True,
                help="Add web search results to enhance context",
                key="chat_include_web"
            )
            
            llm_provider = st.selectbox(
                "LLM Provider",
                ["auto", "openai", "gemini", "mock"],
                help="Choose language model for generation",
                key="chat_llm_provider"
            )
            
            summary_threshold = st.slider(
                "Summary threshold",
                min_value=4,
                max_value=16,
                value=8,
                help="Number of messages before summarizing",
                key="chat_summary_threshold"
            )
    
    # Update thread ID in session state
    if thread_id != st.session_state.current_thread_id:
        st.session_state.current_thread_id = thread_id
        st.session_state.conversation_history = []
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for i, message in enumerate(st.session_state.conversation_history):
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        
        # Add a divider
        st.markdown("---")
    
    # Chat input
    message = st.text_area(
        "Enter your message:",
        placeholder="Ask a question or continue the conversation...",
        help="Your message will be processed with conversation context",
        key="chat_message_input",
        height=100
    )
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        send_button = st.button("💬 Send", type="primary", key="chat_send_btn")
    
    with col2:
        clear_button = st.button("🗑️ Clear Chat", key="chat_clear_btn")
    
    with col3:
        show_sources = st.checkbox(
            "Show sources",
            value=True,
            help="Display source documents",
            key="chat_show_sources"
        )
    
    return {
        'message': message,
        'send_button': send_button,
        'clear_button': clear_button,
        'thread_id': thread_id,
        'include_web_search': include_web_search,
        'show_sources': show_sources,
        'llm_provider': llm_provider,
        'max_messages': max_messages,
        'enable_summary': enable_summary,
        'summary_threshold': summary_threshold
    }


def display_conversational_response(response: Dict[str, Any], show_sources: bool = True):
    """Display conversational RAG response with chat context"""
    
    if response.get('success', False):
        # Display the response
        st.markdown("### 🤖 Assistant Response")
        st.markdown(response['response'])
        
        # Display conversation metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{response.get('processing_time', 0):.2f}s")
        
        with col2:
            st.metric("Context Documents", response.get('context_documents', 0))
        
        with col3:
            st.metric("Thread ID", response.get('thread_id', 'N/A'))
        
        # Show retrieval stats if available
        if 'retrieval_stats' in response:
            with st.expander("📊 Retrieval Statistics"):
                stats = response['retrieval_stats']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Retrieved", stats.get('total_retrieved', 0))
                
                with col2:
                    st.metric("Vector Search", stats.get('vector_search', 0))
                
                with col3:
                    st.metric("Web Search", stats.get('web_search', 0))
                
                with col4:
                    st.metric("Final Context", stats.get('final_context', 0))
        
        # Show sources if requested
        if show_sources and response.get('sources'):
            with st.expander("📚 Source Documents"):
                for i, source in enumerate(response['sources'], 1):
                    # Header with rank and title
                    rank = source.get('rank', i)
                    title = source.get('title', 'Unknown')
                    retrieval_source = source.get('retrieval_source', 'unknown')
                    
                    st.markdown(f"### 📄 Source {rank}: {title}")
                    st.markdown(f"**Source Type:** {retrieval_source.replace('_', ' ').title()}")
                    
                    # Metadata in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if source.get('source') and source['source'] != 'Unknown':
                            st.markdown(f"**Origin:** {source['source']}")
                        
                        if source.get('document_type'):
                            st.markdown(f"**Type:** {source['document_type']}")
                        
                        if source.get('category'):
                            st.markdown(f"**Category:** {source['category']}")
                    
                    with col2:
                        if source.get('similarity_score') is not None:
                            st.metric("Similarity Score", f"{source['similarity_score']:.4f}")
                        
                        if source.get('rerank_score') is not None:
                            st.metric("Rerank Score", f"{source['rerank_score']:.4f}")
                    
                    with col3:
                        if source.get('department'):
                            st.markdown(f"**Department:** {source['department']}")
                        
                        if source.get('upload_date'):
                            st.markdown(f"**Upload Date:** {source['upload_date']}")
                        
                        if source.get('version'):
                            st.markdown(f"**Version:** {source['version']}")
                    
                    # Content preview in a scrollable text area
                    if source.get('content_preview'):
                        st.markdown("**Content Preview:**")
                        st.text_area(
                            label="",
                            value=source['content_preview'],
                            height=120,
                            disabled=True,
                            key=f"conv_source_content_{rank}_{hash(str(source))}"
                        )
                    
                    # Additional metadata and links
                    if source.get('url'):
                        st.link_button("🔗 View Original Source", source['url'], key=f"conv_source_link_{rank}")
                    
                    # Chunk information
                    chunk_info = []
                    if source.get('chunk_index') is not None:
                        chunk_info.append(f"Chunk {source['chunk_index']}")
                    if source.get('total_chunks'):
                        chunk_info.append(f"of {source['total_chunks']}")
                    if source.get('chunk_id'):
                        chunk_info.append(f"(ID: {source['chunk_id']})")
                    
                    if chunk_info:
                        st.caption(" ".join(chunk_info))
                    
                    # Add separator between sources (except for the last one)
                    if i < len(response['sources']):
                        st.markdown("---")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Show conversation history if available
        if 'conversation_history' in response:
            history = response['conversation_history']
            if history:
                with st.expander("💬 Conversation Context"):
                    for msg in history[-5:]:  # Show last 5 messages
                        if msg['role'] == 'user':
                            st.markdown(f"**You:** {msg['content'][:100]}...")
                        else:
                            st.markdown(f"**Assistant:** {msg['content'][:100]}...")
    
    else:
        # Display error
        st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
        
        if 'processing_time' in response:
            st.metric("Processing Time", f"{response['processing_time']:.2f}s")
