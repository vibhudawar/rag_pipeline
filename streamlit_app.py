#!/usr/bin/env python3
"""
Streamlit App for RAG Pipeline Testing

This app provides a user-friendly interface to:
- Upload and ingest single documents
- Batch process multiple files
- Test document retrieval with search
- Configure pipeline settings
- View ingestion statistics

Make sure your .env file is configured with the required API keys.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time
import hashlib
from typing import Dict, Any, List

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our components and pipeline
from ui.components import (
    display_ingestion_result,
    display_batch_results,
    show_pipeline_config,
    create_progress_bar,
    update_progress_bar,
    get_sidebar_config,
    display_search_results,
    show_error_message,
    create_file_uploader,
    display_rag_response,
    display_rag_batch_results,
    create_rag_query_interface,
    create_rag_batch_interface,
    show_rag_pipeline_status
)

from src.ingestion import (
    create_ingestion_pipeline,
    ingest_single_document
)

# Import RAG pipeline
from src.rag_pipeline import create_rag_pipeline

# Configure page
st.set_page_config(
    page_title="RAG Ingestion Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_file_hash(uploaded_file):
    """Generate a unique hash for an uploaded file"""
    if uploaded_file is None:
        return None
    
    # Create hash from file name, size, and content
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    
    hash_content = f"{uploaded_file.name}_{uploaded_file.size}_{len(file_content)}"
    return hashlib.md5(hash_content.encode()).hexdigest()


def is_file_already_processed(file_hash: str, config: Dict[str, Any]) -> bool:
    """Check if a file has already been processed"""
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    
    # Check if this file hash + config combination has been processed
    key = f"{file_hash}_{config['index_name']}"
    return key in st.session_state.processed_files


def mark_file_as_processed(file_hash: str, config: Dict[str, Any], result: Dict[str, Any]):
    """Mark a file as processed and store the result"""
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    
    key = f"{file_hash}_{config['index_name']}"
    st.session_state.processed_files[key] = {
        'result': result,
        'timestamp': time.time(),
        'config': config.copy()
    }


def get_processed_file_result(file_hash: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get the result of a previously processed file"""
    if 'processed_files' not in st.session_state:
        return None
    
    key = f"{file_hash}_{config['index_name']}"
    return st.session_state.processed_files.get(key, {}).get('result')


def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['GEMINI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        st.info("💡 Please create a `.env` file with your API keys before using this app.")
        
        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            Create a `.env` file in your project root with:
            ```
            OPENAI_API_KEY=sk-your-openai-api-key-here
            PINECONE_API_KEY=your-pinecone-api-key-here
            GEMINI_API_KEY=your-gemini-api-key-here  # optional
            
            # Configuration (optional)
            EMBEDDING_PROVIDER=openai
            LLM_PROVIDER=openai
            PINECONE_ENVIRONMENT=us-east-1
            CHUNK_SIZE=1000
            CHUNK_OVERLAP=200
            CHUNKING_STRATEGY=recursive
            ```
            """)
        return False
    
    return True


def single_document_upload(config: Dict[str, Any]):
    """Single document upload and ingestion interface"""
    st.header("📄 Single Document Upload")
    st.markdown("Upload a single document to ingest into your RAG system.")
    
    # File uploader
    uploaded_file = create_file_uploader(accept_multiple_files=False, key="single_upload")
    
    # Additional metadata input
    with st.expander("📋 Additional Metadata (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.text_input("Category", help="Document category or type", key="single_category")
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=1, key="single_priority")
        
        with col2:
            department = st.text_input("Department", help="Originating department", key="single_department")
            version = st.text_input("Version", help="Document version", key="single_version")
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        show_metadata = st.checkbox("Show detailed metadata", value=False, key="single_show_metadata")
    
    with col2:
        auto_process = st.checkbox("Auto-process on upload", value=True, key="single_auto_process")
    
    # Process file if uploaded
    if uploaded_file is not None:
        # Generate file hash for tracking
        file_hash = get_file_hash(uploaded_file)
        
        # Create additional metadata
        additional_metadata = {
            "uploaded_via": "streamlit",
            "upload_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded_file.name
        }
        
        if category:
            additional_metadata["category"] = category
        if department:
            additional_metadata["department"] = department
        if version:
            additional_metadata["version"] = version
        if priority:
            additional_metadata["priority"] = priority
        
        # Show file info
        st.subheader("📁 File Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filename", uploaded_file.name)
        
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        
        with col3:
            file_extension = f".{uploaded_file.name.split('.')[-1]}"
            st.metric("File Type", file_extension.upper())
        
        # Check if file has already been processed
        already_processed = is_file_already_processed(file_hash, config)
        
        if already_processed:
            st.info("🔄 This file has already been processed with the current configuration.")
            
            # Show existing result
            existing_result = get_processed_file_result(file_hash, config)
            if existing_result:
                st.subheader("📊 Previous Processing Result")
                display_ingestion_result(existing_result, show_metadata=show_metadata)
                
                # Option to reprocess
                if st.button("🔄 Reprocess File", key="single_reprocess_btn"):
                    # Remove from processed files to allow reprocessing
                    key = f"{file_hash}_{config['index_name']}"
                    if key in st.session_state.processed_files:
                        del st.session_state.processed_files[key]
                    st.rerun()
        else:
            # Process button or auto-process
            should_process = auto_process
            if not auto_process:
                should_process = st.button("🚀 Process Document", type="primary", key="single_process_btn")
            
            if should_process:
                try:
                    # Show progress
                    progress_bar, status_text = create_progress_bar("Ingesting document...")
                    
                    update_progress_bar(progress_bar, status_text, 0.2, "Reading file...")
                    
                    # Get file content
                    file_content = uploaded_file.read()
                    
                    update_progress_bar(progress_bar, status_text, 0.4, "Processing document...")
                    
                    # Ingest the document
                    result = ingest_single_document(
                        file_path_or_bytes=file_content,
                        file_extension=file_extension,
                        index_name=config['index_name'],
                        additional_metadata=additional_metadata
                    )
                    
                    update_progress_bar(progress_bar, status_text, 1.0, "Complete!")
                    time.sleep(0.5)  # Brief pause to show completion
                    
                    # Clear progress bar
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mark file as processed
                    mark_file_as_processed(file_hash, config, result)
                    
                    # Display results
                    display_ingestion_result(result, show_metadata=show_metadata)
                    
                    # Store result in session state for potential search testing
                    if result['success']:
                        if 'ingested_files' not in st.session_state:
                            st.session_state.ingested_files = []
                        st.session_state.ingested_files.append({
                            'filename': result['filename'],
                            'index_name': config['index_name'],
                            'chunks': result['total_chunks']
                        })
                    
                except Exception as e:
                    show_error_message(e, "Document ingestion failed")


def get_batch_hash(uploaded_files):
    """Generate a unique hash for a batch of files"""
    if not uploaded_files:
        return None
    
    # Create hash from all file names and sizes
    batch_info = []
    for file in uploaded_files:
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        batch_info.append(f"{file.name}_{file.size}_{len(file_content)}")
    
    batch_string = "_".join(sorted(batch_info))
    return hashlib.md5(batch_string.encode()).hexdigest()


def batch_document_upload(config: Dict[str, Any]):
    """Batch document upload interface"""
    st.header("📚 Batch Document Upload")
    st.markdown("Upload multiple documents for batch processing.")
    
    # File uploader for multiple files
    uploaded_files = create_file_uploader(accept_multiple_files=True, key="batch_upload")
    
    # Batch processing options
    col1, col2 = st.columns(2)
    
    with col1:
        batch_category = st.text_input("Batch Category", help="Category for all documents in this batch", key="batch_category")
        batch_department = st.text_input("Batch Department", help="Department for all documents", key="batch_department")
    
    with col2:
        batch_priority = st.selectbox("Batch Priority", ["low", "medium", "high"], index=1, key="batch_priority")
        stop_on_error = st.checkbox("Stop on first error", value=False, key="batch_stop_on_error")
    
    if uploaded_files:
        st.subheader(f"📁 Selected Files ({len(uploaded_files)} files)")
        
        # Generate batch hash
        batch_hash = get_batch_hash(uploaded_files)
        
        # Check if batch has already been processed
        batch_already_processed = is_file_already_processed(f"batch_{batch_hash}", config)
        
        # Show file list
        total_size = 0
        for file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(file.name)
            with col2:
                st.text(f"{file.size / 1024:.2f} KB")
            with col3:
                st.text(f".{file.name.split('.')[-1].upper()}")
            total_size += file.size
        
        st.metric("Total Size", f"{total_size / 1024:.2f} KB")
        
        if batch_already_processed:
            st.info("🔄 This batch has already been processed with the current configuration.")
            
            # Show existing result
            existing_result = get_processed_file_result(f"batch_{batch_hash}", config)
            if existing_result and 'batch_results' in existing_result:
                st.subheader("📊 Previous Batch Processing Results")
                display_batch_results(existing_result['batch_results'])
                
                # Option to reprocess
                if st.button("🔄 Reprocess Batch", key="batch_reprocess_btn"):
                    # Remove from processed files to allow reprocessing
                    key = f"batch_{batch_hash}_{config['index_name']}"
                    if key in st.session_state.processed_files:
                        del st.session_state.processed_files[key]
                    st.rerun()
        else:
            # Process button
            if st.button("🚀 Process All Documents", type="primary", key="batch_process_btn"):
                try:
                    # Create batch metadata
                    batch_metadata = {
                        "uploaded_via": "streamlit_batch",
                        "batch_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "batch_size": len(uploaded_files)
                    }
                    
                    if batch_category:
                        batch_metadata["category"] = batch_category
                    if batch_department:
                        batch_metadata["department"] = batch_department
                    if batch_priority:
                        batch_metadata["priority"] = batch_priority
                    
                    # Process files
                    results = []
                    
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {file.name} ({i + 1}/{len(uploaded_files)})...")
                        
                        try:
                            # Get file content and extension
                            file_content = file.read()
                            file_extension = f".{file.name.split('.')[-1]}"
                            
                            # Add file-specific metadata
                            file_metadata = batch_metadata.copy()
                            file_metadata["batch_position"] = i + 1
                            file_metadata["original_filename"] = file.name
                            
                            # Ingest the document
                            result = ingest_single_document(
                                file_path_or_bytes=file_content,
                                file_extension=file_extension,
                                index_name=config['index_name'],
                                additional_metadata=file_metadata
                            )
                            
                            results.append(result)
                            
                            # Update results display
                            with results_container.container():
                                st.subheader("📊 Processing Results")
                                display_batch_results(results)
                            
                            # Stop on error if requested
                            if not result['success'] and stop_on_error:
                                st.warning("⚠️ Stopping batch processing due to error")
                                break
                            
                        except Exception as e:
                            error_result = {
                                'success': False,
                                'error': str(e),
                                'filename': file.name,
                                'processing_time_seconds': 0
                            }
                            results.append(error_result)
                            
                            if stop_on_error:
                                st.warning("⚠️ Stopping batch processing due to error")
                                show_error_message(e, f"Processing {file.name}")
                                break
                    
                    # Final results
                    progress_bar.progress(1.0)
                    status_text.text("✅ Batch processing complete!")
                    
                    # Clear progress after a moment
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mark batch as processed
                    batch_result = {
                        'batch_results': results,
                        'total_files': len(uploaded_files),
                        'successful_files': sum(1 for r in results if r.get('success', False))
                    }
                    mark_file_as_processed(f"batch_{batch_hash}", config, batch_result)
                    
                    # Store successful results in session state
                    successful_results = [r for r in results if r.get('success', False)]
                    if successful_results:
                        if 'ingested_files' not in st.session_state:
                            st.session_state.ingested_files = []
                        
                        for result in successful_results:
                            st.session_state.ingested_files.append({
                                'filename': result['filename'],
                                'index_name': config['index_name'],
                                'chunks': result['total_chunks']
                            })
                    
                except Exception as e:
                    show_error_message(e, "Batch processing failed")


def search_and_test(config: Dict[str, Any]):
    """Search interface to test ingested documents"""
    st.header("🔍 Search & Test")
    st.markdown("Test retrieval from your ingested documents.")
    
    # Show ingested files info
    if 'ingested_files' in st.session_state and st.session_state.ingested_files:
        st.subheader("📚 Recently Ingested Documents")
        
        files_df = []
        for file_info in st.session_state.ingested_files[-10:]:  # Show last 10
            files_df.append({
                'Filename': file_info['filename'],
                'Index': file_info['index_name'],
                'Chunks': file_info['chunks']
            })
        
        if files_df:
            import pandas as pd
            df = pd.DataFrame(files_df)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No documents have been ingested yet. Upload some documents first!")
    
    # Search interface
    st.subheader("🔍 Search Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'machine learning algorithms'",
            help="Search for relevant content in your ingested documents",
            key="search_query"
        )
    
    with col2:
        top_k = st.selectbox("Number of results", [3, 5, 10, 15], index=1, key="search_top_k")
    
    # Search button
    if st.button("🔍 Search", type="primary", key="search_btn") and query:
        try:
            # Show search progress
            with st.spinner("Searching documents..."):
                # Import here to avoid circular imports
                from src.ingestion.vector_store import get_vector_store
                from src.retrieval.embedder import get_embedder
                
                # Get vector store and embedder
                vector_store = get_vector_store()
                print(f"🔍 Vector store1: {vector_store.__dict__}")
                embedder = get_embedder()
                
                # Perform similarity search
                results = vector_store.similarity_search(
                    index_name=config['index_name'],
                    query=query,
                    embedder=embedder,
                    top_k=top_k
                )
                
                # Convert Document objects to dictionary format for display
                formatted_results = []
                for doc in results:
                    formatted_results.append({
                        'text': doc.page_content,
                        'metadata': doc.metadata,
                        'score': doc.metadata.get('score', 0.0)
                    })
                
                # Display results
                st.subheader("📄 Search Results")
                display_search_results(formatted_results, query)
                
        except Exception as e:
            show_error_message(e, "Search failed")
    
    elif query and not st.button("🔍 Search", type="primary", key="search_btn_inactive"):
        st.info("Click 'Search' to find relevant documents")


def configuration_and_stats(config: Dict[str, Any]):
    """Configuration and statistics interface"""
    st.header("⚙️ Configuration & Statistics")
    st.markdown("View and manage your RAG pipeline configuration.")
    
    try:
        # Create pipeline to get stats
        pipeline = create_ingestion_pipeline(index_name=config['index_name'])
        stats = pipeline.get_ingestion_stats()
        
        # Show current configuration
        show_pipeline_config(stats)
        
        # Vector store information
        st.subheader("🗄️ Vector Store Information")
        
        try:
            indexes = pipeline.vector_store.list_indexes()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Available Indexes", len(indexes))
            
            with col2:
                current_index = config['index_name']
                index_exists = current_index in indexes
                st.metric("Current Index Status", "✅ Exists" if index_exists else "❌ Not Found")
            
            # List all indexes
            if indexes:
                st.subheader("📋 Available Indexes")
                for idx in indexes:
                    status = "🎯 Current" if idx == current_index else "📁"
                    st.text(f"{status} {idx}")
            else:
                st.info("No indexes found. Upload some documents to create an index.")
        
        except Exception as e:
            st.warning(f"Could not retrieve vector store information: {str(e)}")
        
        # Environment information
        st.subheader("🌍 Environment Information")
        
        env_info = {
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "Not set"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "Not set"),
            "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", "Not set"),
            "CHUNK_SIZE": os.getenv("CHUNK_SIZE", "Not set"),
            "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", "Not set"),
            "CHUNKING_STRATEGY": os.getenv("CHUNKING_STRATEGY", "Not set")
        }
        
        col1, col2 = st.columns(2)
        items = list(env_info.items())
        mid = len(items) // 2
        
        with col1:
            for key, value in items[:mid]:
                st.text(f"{key}: {value}")
        
        with col2:
            for key, value in items[mid:]:
                st.text(f"{key}: {value}")
        
        # Processed files information
        if st.checkbox("Show processed files", key="show_processed_files"):
            st.subheader("📂 Processed Files Cache")
            
            if 'processed_files' in st.session_state and st.session_state.processed_files:
                st.metric("Cached processed files", len(st.session_state.processed_files))
                
                if st.button("🗑️ Clear processed files cache", key="clear_processed_cache"):
                    st.session_state.processed_files = {}
                    st.success("Processed files cache cleared!")
                    st.rerun()
                
                # Show processed files
                with st.expander("View processed files details"):
                    for key, data in st.session_state.processed_files.items():
                        result = data['result']
                        timestamp = time.ctime(data['timestamp'])
                        if 'batch_results' in result:
                            st.text(f"📚 Batch: {key} - {timestamp}")
                            st.text(f"   Files: {result['total_files']}, Successful: {result['successful_files']}")
                        else:
                            filename = result.get('filename', 'Unknown')
                            success = "✅" if result.get('success', False) else "❌"
                            st.text(f"📄 {success} {filename} - {timestamp}")
            else:
                st.info("No processed files in cache")
        
        # Session state information
        if st.checkbox("Show session information", key="show_session_info"):
            st.subheader("🔄 Session Information")
            
            if 'ingested_files' in st.session_state:
                st.metric("Documents ingested this session", len(st.session_state.ingested_files))
                
                if st.button("🗑️ Clear session data", key="clear_session_btn"):
                    st.session_state.ingested_files = []
                    st.success("Session data cleared!")
                    st.rerun()
            else:
                st.info("No documents ingested this session")
    
    except Exception as e:
        show_error_message(e, "Failed to load configuration")


def rag_query_interface(config: Dict[str, Any]):
    """RAG query interface for asking questions about ingested documents"""
    st.header("🤖 RAG Query & Chat")
    st.markdown("Ask questions about your ingested documents using the complete RAG pipeline.")
    
    # Check if we have ingested documents
    if 'ingested_files' in st.session_state and st.session_state.ingested_files:
        st.info(f"📚 {len(st.session_state.ingested_files)} documents available for querying")
    else:
        st.warning("⚠️ No documents ingested yet. Please upload some documents first!")
        return
    
    # Create tabs for single and batch queries
    single_tab, batch_tab, pipeline_tab = st.tabs([
        "🔍 Single Query",
        "📊 Batch Queries", 
        "⚙️ Pipeline Config"
    ])
    
    with single_tab:
        single_rag_query(config)
    
    with batch_tab:
        batch_rag_query(config)
    
    with pipeline_tab:
        pipeline_configuration(config)


def single_rag_query(config: Dict[str, Any]):
    """Single RAG query interface"""
    
    # Create query interface
    query_config = create_rag_query_interface()
    
    query = query_config['query']
    include_web_search = query_config['include_web_search']
    show_sources = query_config['show_sources']
    reranker_provider = query_config['reranker_provider']
    llm_provider = query_config['llm_provider']
    
    # Query button
    if st.button("🚀 Ask Question", type="primary", key="single_rag_query_btn") and query:
        try:
            with st.spinner("🔍 Processing your question..."):
                # Create RAG pipeline
                pipeline = create_rag_pipeline(
                    index_name=config['index_name'],
                    include_web_search=include_web_search,
                    reranker_provider=reranker_provider,
                    llm_provider=llm_provider
                )
                
                # Process query
                response = pipeline.query(query, return_sources=show_sources)
                
                # Display results
                display_rag_response(response)
                
        except Exception as e:
            show_error_message(e, "RAG query failed")
    
    elif query and not st.button("🚀 Ask Question", type="primary", key="single_rag_query_btn_inactive"):
        st.info("👆 Click 'Ask Question' to get an answer")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        Try asking questions like:
        
        - What are the main topics covered in the documents?
        - Can you summarize the key findings?
        - What recommendations are mentioned?
        - How does [specific topic] relate to [another topic]?
        - What are the advantages and disadvantages of [topic]?
        - Can you explain [specific concept] mentioned in the documents?
        """)


def batch_rag_query(config: Dict[str, Any]):
    """Batch RAG query interface"""
    
    # Create batch interface
    batch_config = create_rag_batch_interface()
    
    questions = batch_config['questions']
    include_web_search = batch_config['include_web_search']
    show_sources = batch_config['show_sources']
    reranker_provider = batch_config['reranker_provider']
    
    # Process button
    if st.button("🚀 Process All Questions", type="primary", key="batch_rag_query_btn") and questions:
        try:
            with st.spinner(f"🔍 Processing {len(questions)} questions..."):
                # Create RAG pipeline
                pipeline = create_rag_pipeline(
                    index_name=config['index_name'],
                    include_web_search=include_web_search,
                    reranker_provider=reranker_provider
                )
                
                # Process queries
                results = pipeline.batch_query(questions, return_sources=show_sources)
                
                # Display results
                display_rag_batch_results(results)
                
        except Exception as e:
            show_error_message(e, "Batch RAG query failed")
    
    elif questions and not st.button("🚀 Process All Questions", type="primary", key="batch_rag_query_btn_inactive"):
        st.info(f"👆 Click 'Process All Questions' to get answers for {len(questions)} questions")
    
    # Sample batch questions
    if st.button("📝 Load Sample Questions", key="load_sample_questions"):
        sample_questions = [
            "What are the main topics covered in the documents?",
            "Can you summarize the key findings?",
            "What recommendations are mentioned?",
            "What are the most important concepts discussed?"
        ]
        
        # Update the text area (this is a bit hacky but works)
        st.session_state.rag_batch_questions = "\n".join(sample_questions)
        st.rerun()


def pipeline_configuration(config: Dict[str, Any]):
    """Show and configure RAG pipeline settings"""
    
    try:
        # Create a pipeline to get configuration info
        pipeline = create_rag_pipeline(
            index_name=config['index_name'],
            include_web_search=True,  # Default settings for info
            reranker_provider="auto",
            llm_provider="auto"
        )
        
        # Show pipeline status
        pipeline_info = pipeline.get_pipeline_info()
        show_rag_pipeline_status(pipeline_info)
        
        # Advanced settings
        st.subheader("🔧 Advanced Settings")
        
        with st.expander("RAG Pipeline Parameters"):
            st.markdown("""
            **Vector Top-K**: Number of documents retrieved from vector store  
            **Web Search Results**: Number of web search results to include  
            **Final Top-K**: Number of documents after reranking  
            
            **Providers**:
            - **auto**: Automatically choose best available provider
            - **openai**: Use OpenAI models (requires API key)
            - **gemini**: Use Google Gemini models (requires API key)
            - **cohere**: Use Cohere reranker (requires API key)
            - **huggingface**: Use local HuggingFace models
            - **mock**: Use mock providers for testing
            """)
        
        # Environment variables info
        st.subheader("🌍 Environment Configuration")
        
        env_status = {
            "OPENAI_API_KEY": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
            "GEMINI_API_KEY": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
            "COHERE_API_KEY": "✅" if os.getenv("COHERE_API_KEY") else "❌",
            "SERPAPI_KEY": "✅" if os.getenv("SERPAPI_KEY") else "❌",
            "PINECONE_API_KEY": "✅" if os.getenv("PINECONE_API_KEY") else "❌"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**LLM Providers:**")
            st.text(f"OpenAI: {env_status['OPENAI_API_KEY']}")
            st.text(f"Gemini: {env_status['GEMINI_API_KEY']}")
        
        with col2:
            st.markdown("**Other Services:**")
            st.text(f"Cohere: {env_status['COHERE_API_KEY']}")
            st.text(f"SerpAPI: {env_status['SERPAPI_KEY']}")
            st.text(f"Pinecone: {env_status['PINECONE_API_KEY']}")
        
        # Recommendations
        missing_keys = [key for key, status in env_status.items() if status == "❌"]
        if missing_keys:
            st.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
            st.info("💡 Add these to your .env file for full functionality")
        
    except Exception as e:
        show_error_message(e, "Failed to load pipeline configuration")


def main():
    """Main application function"""
    
    # Title and description
    st.title("📚 RAG Pipeline")
    st.markdown("**Upload, process, and test documents for your Retrieval-Augmented Generation system**")
    
    # Check environment first
    if not check_environment():
        return
    
    # Get sidebar configuration once at the top level
    config = get_sidebar_config()
    
    # Create tabs for different functionality
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Single Upload",
        "📚 Batch Upload", 
        "🔍 Search & Test",
        "🤖 RAG Query",
        "⚙️ Config & Stats"
    ])
    
    with tab1:
        single_document_upload(config)
    
    with tab2:
        batch_document_upload(config)
    
    with tab3:
        search_and_test(config)
    
    with tab4:
        rag_query_interface(config)
    
    with tab5:
        configuration_and_stats(config)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Upload documents in the first two tabs, then ask questions in the RAG Query tab!"
    )


if __name__ == "__main__":
    main() 