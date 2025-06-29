# Complete RAG Pipeline Documentation

This document provides comprehensive information about the RAG (Retrieval-Augmented Generation) pipeline with hybrid retrieval, reranking, and generation capabilities.

## 🎯 Features

### Core Pipeline Components
- **📚 Document Ingestion**: Upload and process PDF, DOCX, TXT, MD files
- **🔍 Hybrid Retrieval**: Vector search + Web search
- **🎯 Reranking**: Multiple rerankers for improved relevance
- **🤖 Generation**: LLM-powered answer generation with source citations
- **🖥️ Streamlit UI**: User-friendly interface for all operations

### Supported Providers

#### Embedding Providers
- **OpenAI**: `text-embedding-ada-002` (recommended)
- **Google Gemini**: `models/embedding-001`
- **Auto**: Automatically choose best available

#### LLM Providers  
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`
- **Google Gemini**: `gemini-pro`
- **Mock**: For testing without API keys

#### Reranking Providers
- **Cohere**: High-quality commercial reranker
- **HuggingFace**: Local cross-encoder models
- **Score-based**: Simple similarity score sorting
- **Auto**: Automatically choose best available

#### Web Search Providers
- **SerpAPI**: Google search results (requires API key)
- **DuckDuckGo**: Free search (no API key required)
- **Mock**: For testing

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here

# Optional (for enhanced functionality)
GEMINI_API_KEY=your-gemini-api-key-here
COHERE_API_KEY=your-cohere-api-key-here
SERPAPI_KEY=your-serpapi-key-here

# Configuration
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
RERANKER_PROVIDER=auto
INCLUDE_WEB_SEARCH=true
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test the Pipeline

```bash
python test_rag_pipeline.py
```

### 4. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

## 📋 Usage Guide

### Streamlit Interface

The app has 5 main tabs:

#### 1. 📄 Single Upload
- Upload individual documents
- Add metadata (category, department, version)
- Real-time processing with progress tracking
- Automatic deduplication

#### 2. 📚 Batch Upload  
- Upload multiple files at once
- Batch metadata settings
- Progress tracking for each file
- Error handling options

#### 3. 🔍 Search & Test
- Basic vector similarity search
- View recently ingested documents
- Test retrieval without generation

#### 4. 🤖 RAG Query (NEW!)
- **Single Query**: Ask individual questions
- **Batch Queries**: Process multiple questions
- **Pipeline Config**: View/configure RAG settings

##### Single Query Options:
- Include web search: Enhance with web results
- Show sources: Display source documents used
- Reranker: Choose reranking method
- LLM Provider: Select language model

##### Batch Query Features:
- Process multiple questions at once
- Performance statistics
- Batch result summaries

#### 5. ⚙️ Config & Stats
- View pipeline configuration
- Monitor ingestion statistics
- Manage processed files cache
- Environment status

### Programmatic Usage

#### Simple Query
```python
from src.rag_pipeline import simple_rag_query

response = simple_rag_query(
    "What is machine learning?",
    index_name="my-docs"
)

print(response['answer'])
```

#### Advanced Pipeline
```python
from src.rag_pipeline import create_rag_pipeline

# Create pipeline with custom settings
pipeline = create_rag_pipeline(
    index_name="my-docs",
    embedding_provider="openai",
    reranker_provider="cohere", 
    llm_provider="gpt-4",
    include_web_search=True,
    vector_top_k=15,
    web_search_results=5,
    final_top_k=8
)

# Single query
response = pipeline.query(
    "Explain the benefits of renewable energy",
    return_sources=True
)

# Batch queries
questions = [
    "What is solar energy?",
    "How do wind turbines work?",
    "What are the costs of renewable energy?"
]

batch_results = pipeline.batch_query(questions)
```

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EMBEDDING_PROVIDER` | Embedding model provider | `auto` | `openai`, `gemini`, `auto` |
| `LLM_PROVIDER` | Language model provider | `auto` | `openai`, `gemini`, `auto` |
| `RERANKER_PROVIDER` | Reranking method | `auto` | `cohere`, `huggingface`, `score`, `none`, `auto` |
| `INCLUDE_WEB_SEARCH` | Enable web search | `true` | `true`, `false` |
| `VECTOR_TOP_K` | Vector search results | `10` | Any integer |
| `WEB_SEARCH_RESULTS` | Web search results | `3` | Any integer |
| `FINAL_TOP_K` | Final reranked results | `5` | Any integer |
| `CHUNK_SIZE` | Document chunk size | `1000` | Any integer |
| `CHUNK_OVERLAP` | Chunk overlap | `200` | Any integer |

### Pipeline Parameters

When creating a pipeline programmatically:

```python
pipeline = create_rag_pipeline(
    index_name="my-index",           # Pinecone index name
    embedding_provider="openai",     # Embedding provider
    reranker_provider="cohere",      # Reranker provider
    llm_provider="openai",           # LLM provider
    include_web_search=True,         # Enable web search
    vector_top_k=10,                 # Vector search results
    web_search_results=3,            # Web search results
    final_top_k=5,                   # Final context size
    temperature=0.0                  # LLM temperature
)
```

## 🔧 Component Details

### RAG Pipeline Flow

1. **Query Processing**: User submits a question
2. **Hybrid Retrieval**: 
   - Vector search: Find similar documents in vector store
   - Web search: Get relevant web results (if enabled)
3. **Document Combination**: Merge results from both sources
4. **Reranking**: Improve relevance using specialized reranker
5. **Context Preparation**: Format top documents for LLM
6. **Generation**: Generate answer using LLM with context
7. **Response Formatting**: Return answer with sources and metadata

### Reranking Options

#### Cohere Reranker (Best Quality)
- Requires `COHERE_API_KEY`
- Uses `rerank-english-v3.0` model
- Commercial API with high accuracy

#### HuggingFace Reranker (Local)
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Runs locally, no API required
- Requires `sentence-transformers` package

#### Score-based Reranker (Fallback)
- Uses existing similarity scores
- Fast and always available
- Lower quality than dedicated rerankers

### Web Search Integration

Web search enhances RAG by providing:
- Recent information not in your documents
- Broader context and perspectives
- Current events and updates

Search providers:
- **SerpAPI**: Requires API key, high quality Google results
- **DuckDuckGo**: Free, no API key required
- **Mock**: For testing and development

## 📊 Response Format

RAG query responses include:

```python
{
    'success': True,
    'question': 'What is machine learning?',
    'answer': 'Machine learning is...',
    'model': 'gpt-3.5-turbo',
    'processing_time': 2.34,
    'retrieval_stats': {
        'total_retrieved': 13,
        'vector_search': 10,
        'web_search': 3,
        'final_context': 5
    },
    'sources': [
        {
            'rank': 1,
            'title': 'ML Guide',
            'content_preview': 'Machine learning is...',
            'source': 'document',
            'retrieval_source': 'vector_store',
            'url': '',
            'rerank_score': 0.95,
            'similarity_score': 0.87
        }
    ]
}
```

## 🧪 Testing

### Run Test Suite
```bash
python test_rag_pipeline.py
```

The test suite validates:
- Document ingestion
- Vector search functionality  
- RAG pipeline operations
- Batch processing
- Error handling

### Manual Testing

1. **Upload Test Documents**: Use sample documents in Streamlit
2. **Basic Search**: Test vector similarity search
3. **RAG Queries**: Ask questions and verify answers
4. **Source Verification**: Check that sources are relevant
5. **Provider Testing**: Try different LLM and reranker providers

## 🔍 Troubleshooting

### Common Issues

#### "No documents retrieved"
- Check if documents are properly ingested
- Verify Pinecone index exists and has data
- Try broader or different query terms

#### "Generation failed"
- Verify LLM API keys are set correctly
- Check API quota/limits
- Try different LLM provider

#### "Reranking failed, using original order"
- Check Cohere API key if using Cohere reranker
- Install `sentence-transformers` for HuggingFace reranker
- Pipeline falls back to score-based reranking

#### "Web search failed"
- Check SerpAPI key if using SerpAPI
- Pipeline falls back to DuckDuckGo or vector-only search
- Web search failures don't stop the pipeline

### Performance Optimization

#### For Better Quality:
- Use Cohere reranker with valid API key
- Increase `final_top_k` for more context
- Use GPT-4 for better generation quality
- Enable web search for broader knowledge

#### For Better Speed:
- Reduce `vector_top_k` and `web_search_results`
- Use score-based reranker instead of Cohere/HuggingFace
- Disable web search if not needed
- Use GPT-3.5-turbo instead of GPT-4

#### For Cost Optimization:
- Use Gemini instead of OpenAI (if available)
- Reduce context size (`final_top_k`)
- Use local HuggingFace reranker instead of Cohere
- Cache frequently asked questions

## 🤝 Contributing

The RAG pipeline is designed to be extensible:

- **Add new LLM providers**: Extend `src/generation/llm.py`
- **Add new rerankers**: Extend `src/reranking/reranker.py`  
- **Add new web search**: Extend `src/retrieval/web_search.py`
- **Add new embedders**: Extend `src/retrieval/embedder.py`

Follow the existing abstract base class patterns for consistency.

## 📄 License

This project is part of the RAG ingestion pipeline and follows the same licensing terms.

---

💡 **Pro Tip**: Start with default settings and gradually customize based on your specific use case and quality requirements! 