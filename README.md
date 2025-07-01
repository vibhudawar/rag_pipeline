# 📚 RAG Pipeline

A comprehensive **Retrieval-Augmented Generation (RAG)** system that combines vector search, web search, reranking, and LLM generation to provide intelligent question-answering over your documents.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.46.0-red.svg)

## 🌟 Features

### 🔍 **Hybrid Retrieval**
- **Vector Search**: Semantic similarity search using Pinecone vector database
- **Web Search**: Real-time web results via SerpAPI, DuckDuckGo, or Brave
- **Smart Combination**: Automatically merges and deduplicates results

### 📄 **Document Processing**
- **Multiple Formats**: PDF, DOCX, TXT, Markdown support
- **Intelligent Chunking**: Recursive and token-based chunking strategies
- **Batch Upload**: Process multiple documents simultaneously
- **Metadata Enhancement**: Rich metadata extraction and custom tagging

### 🎯 **Advanced Reranking**
- **Cohere Reranker**: State-of-the-art neural reranking
- **HuggingFace Models**: Local cross-encoder models
- **Score-based**: Fallback similarity score ranking
- **Configurable**: Choose best model for your use case

### 🤖 **Multi-Provider LLM Support**
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Google Gemini**: Gemini Pro, Gemini Pro Vision
- **Auto-Selection**: Intelligent provider fallback

### 🖥️ **User-Friendly Interface**
- **Streamlit Web App**: Beautiful, intuitive interface
- **Real-time Processing**: Live progress tracking
- **Interactive Search**: Test retrieval before querying
- **Batch Operations**: Handle multiple queries at once

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API Keys for your chosen providers (OpenAI, Pinecone, etc.)

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
# Required APIs
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here

# Optional APIs (for enhanced features)
GEMINI_API_KEY=your-gemini-api-key-here
COHERE_API_KEY=your-cohere-api-key-here
SERPAPI_KEY=your-serpapi-key-here

# Configuration (optional)
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
PINECONE_ENVIRONMENT=us-east-1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNKING_STRATEGY=recursive

# RAG Pipeline Settings
INCLUDE_WEB_SEARCH=true
VECTOR_TOP_K=10
WEB_SEARCH_RESULTS=3
FINAL_TOP_K=5
RERANKER_PROVIDER=auto
```

### 3. Launch the App

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to access the web interface.

## 📖 Usage Guide

### Document Upload

1. **Single Upload**: Upload individual documents with custom metadata
2. **Batch Upload**: Process multiple files with shared metadata
3. **Supported Formats**: PDF, DOCX, TXT, MD files
4. **Smart Processing**: Automatic format detection and optimization

### Search & Test

- **Vector Search**: Test semantic similarity search
- **Results Preview**: View retrieved chunks with scores
- **Metadata Display**: Inspect document metadata and sources

### RAG Queries

#### Single Query
```python
from src.rag_pipeline import create_rag_pipeline

# Create pipeline
pipeline = create_rag_pipeline(
    index_name="my-docs",
    include_web_search=True,
    reranker_provider="cohere"
)

# Ask questions
response = pipeline.query("What are the key benefits of machine learning?")
print(response['answer'])
```

#### Batch Processing
```python
questions = [
    "What is machine learning?",
    "How does neural networks work?",
    "What are the applications of AI?"
]

results = pipeline.batch_query(questions)
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
```

### Configuration Options

| Component | Options | Description |
|-----------|---------|-------------|
| **Embeddings** | `openai`, `gemini`, `auto` | Text embedding providers |
| **LLM** | `openai`, `gemini`, `auto` | Language model providers |
| **Reranker** | `cohere`, `huggingface`, `score`, `auto` | Document reranking methods |
| **Chunking** | `recursive`, `token` | Text splitting strategies |
| **Web Search** | `serpapi`, `duckduckgo`, `brave` | Web search providers |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector Store   │    │   Web Search    │
│   Ingestion     │───▶│   (Pinecone)     │◀──▶│   (Multiple)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chunking &    │    │   Embedding      │    │   Retrieval     │
│   Preprocessing │    │   Generation     │    │   Fusion        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │                        │
                                 ▼                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   Reranking      │◀───│   Query         │
                        │   (Cohere/HF)    │    │   Processing    │
                        └──────────────────┘    └─────────────────┘
                                 │                        │
                                 ▼                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   LLM            │◀───│   Context       │
                        │   Generation     │    │   Preparation   │
                        └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
RAG/
├── src/
│   ├── ingestion/           # Document processing
│   │   ├── document_parser.py
│   │   ├── chunking.py
│   │   └── vector_store.py
│   ├── retrieval/           # Search components
│   │   ├── embedder.py
│   │   └── web_search.py
│   ├── reranking/           # Result reranking
│   │   └── reranker.py
│   ├── generation/          # LLM integration
│   │   └── llm.py
│   └── rag_pipeline.py      # Main pipeline
├── ui/
│   └── components.py        # Streamlit components
├── streamlit_app.py         # Web interface
├── config.py               # Configuration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🧪 Testing

Run the test suite to verify your setup:

```bash
python test_rag_pipeline.py
```

This will test:
- Document ingestion
- Vector search functionality
- RAG pipeline operations
- Batch processing
- Error handling

## 🔧 Advanced Configuration

### Custom Chunking

```python
from src.ingestion.chunking import get_chunker

chunker = get_chunker(
    strategy="recursive",
    chunk_size=1500,
    chunk_overlap=300
)
```

### Custom Reranking

```python
from src.reranking.reranker import CohereReranker

reranker = CohereReranker(model="rerank-english-v3.0")
```

### Pipeline Customization

```python
pipeline = create_rag_pipeline(
    index_name="custom-index",
    vector_top_k=15,
    web_search_results=5,
    final_top_k=8,
    include_web_search=True
)
```

## 📊 Monitoring & Analytics

The pipeline provides comprehensive analytics:

- **Processing Times**: Track query response times
- **Retrieval Stats**: Monitor source breakdown (vector vs web)
- **Success Rates**: Track successful vs failed queries
- **Usage Patterns**: Analyze query types and frequencies

## 🚨 Troubleshooting

### Common Issues

#### "No documents retrieved"
- ✅ Verify documents are properly ingested
- ✅ Check Pinecone index exists and has data
- ✅ Try broader search terms

#### "Generation failed"
- ✅ Verify API keys are correct
- ✅ Check API rate limits
- ✅ Try different LLM provider

#### "Reranking failed"
- ✅ Check Cohere API key (if using Cohere)
- ✅ Install `sentence-transformers` for HuggingFace
- ✅ Pipeline automatically falls back to score-based ranking

#### "Web search failed"
- ✅ Verify SerpAPI key (if using SerpAPI)
- ✅ Pipeline falls back to DuckDuckGo search
- ✅ Web search failures don't stop the pipeline

### Performance Optimization

#### For Better Quality:
- Use Cohere reranker with valid API key
- Increase `final_top_k` for more context
- Enable web search for broader knowledge
- Use GPT-4 for better generation quality

#### For Better Speed:
- Reduce `vector_top_k` and `web_search_results`
- Use score-based reranker instead of neural models
- Disable web search if not needed
- Use GPT-3.5-turbo instead of GPT-4

#### For Cost Optimization:
- Use Gemini instead of OpenAI (when available)
- Reduce context size (`final_top_k`)
- Use local HuggingFace reranker instead of Cohere
- Cache frequently asked questions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the excellent AI framework
- [Pinecone](https://www.pinecone.io/) for vector database services
- [Streamlit](https://streamlit.io/) for the beautiful web interface
- [OpenAI](https://openai.com/) and [Google](https://ai.google/) for powerful language models
- [Cohere](https://cohere.ai/) for state-of-the-art reranking

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](../../issues)
3. Create a new issue with detailed description

---

**Built with ❤️ for the AI community**
