# RAG Ingestion Pipeline

This directory contains the core ingestion pipeline for the RAG system, built with LangChain integrations for maximum compatibility and ease of use.

## 🚀 Quick Start

1. **Set up environment variables** (create a `.env` file):
   ```bash
   # Service Selection
   EMBEDDING_PROVIDER=openai
   LLM_PROVIDER=openai
   
   # API Keys
   OPENAI_API_KEY=sk-your-openai-api-key-here
   PINECONE_API_KEY=your-pinecone-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here  # optional
   
   # Configuration
   PINECONE_ENVIRONMENT=us-east-1
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   CHUNKING_STRATEGY=recursive
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the example**:
   ```bash
   python example_ingestion.py
   ```

## 🏗️ Architecture

The ingestion pipeline follows a modular design with these key components:

### 📄 Document Parsing (`src/ingestion/document_parser.py`)
- **Supported formats**: PDF, DOCX, TXT, MD
- **Factory pattern** for easy extension
- **Metadata extraction** (filename, file type, page count, etc.)
- **Handles both file paths and bytes** (for uploads)

### ✂️ Text Chunking (`src/ingestion/chunking.py`)
- **Recursive chunking**: Preserves semantic boundaries
- **Token-based chunking**: Precise token control using tiktoken
- **Configurable**: chunk size, overlap, separators
- **Metadata preservation**: Each chunk retains document metadata

### 🧮 Embeddings (`src/retrieval/embedder.py`)
- **LangChain integration**: Uses `langchain-openai` and `langchain-google-genai`
- **Provider abstraction**: Easy switching between OpenAI and Gemini
- **Batch processing**: Efficient embedding generation
- **Consistent interface**: Same API regardless of provider

### 🗃️ Vector Storage (`src/ingestion/vector_store.py`)
- **LangChain Pinecone integration**: Uses `langchain-pinecone`
- **Hybrid approach**: LangChain methods + raw Pinecone for flexibility
- **Automatic index management**: Creates indexes as needed
- **Metadata filtering**: Supports complex queries

### 🔄 Pipeline Orchestration (`src/ingestion/__init__.py`)
- **End-to-end workflow**: Parse → Chunk → Embed → Store
- **Batch processing**: Handle multiple files efficiently
- **Progress tracking**: Detailed logging and statistics
- **Error handling**: Graceful failure handling

## 📝 Usage Examples

### Single Document Ingestion
```python
from src.ingestion import ingest_single_document

result = ingest_single_document(
    file_path_or_bytes="document.pdf",
    index_name="my-documents",
    additional_metadata={"category": "research"}
)
print(f"Ingested {result['total_chunks']} chunks")
```

### Streamlit File Upload
```python
from src.ingestion import ingest_single_document

# In your Streamlit app
uploaded_file = st.file_uploader("Upload document", type=['pdf', 'txt', 'docx'])
if uploaded_file:
    file_extension = f".{uploaded_file.name.split('.')[-1]}"
    
    result = ingest_single_document(
        file_path_or_bytes=uploaded_file.read(),
        file_extension=file_extension,
        additional_metadata={"uploaded_by": "user", "app": "streamlit"}
    )
    
    if result['success']:
        st.success(f"✅ Ingested {result['total_chunks']} chunks!")
```

### Directory Batch Processing
```python
from src.ingestion import ingest_directory

results = ingest_directory(
    directory_path="./documents",
    recursive=True,
    additional_metadata={"batch_type": "initial_load"}
)

successful = sum(1 for r in results if r['success'])
print(f"Successfully processed {successful}/{len(results)} documents")
```

### Advanced Pipeline Usage
```python
from src.ingestion import create_ingestion_pipeline

# Create pipeline with custom index
pipeline = create_ingestion_pipeline(index_name="custom-index")

# Get pipeline statistics
stats = pipeline.get_ingestion_stats()
print(f"Using {stats['embedder_type']} with {stats['chunking_strategy']} chunking")

# Ingest with custom settings
result = pipeline.ingest_document(
    file_path_or_bytes="document.pdf",
    additional_metadata={
        "priority": "high",
        "department": "research",
        "version": "1.0"
    }
)
```

## ⚙️ Configuration Options

All configuration is managed through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `"openai"` | Embedding service: "openai" or "gemini" |
| `LLM_PROVIDER` | `"openai"` | LLM service: "openai" or "gemini" |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `CHUNKING_STRATEGY` | `"recursive"` | Chunking method: "recursive" or "token" |
| `PINECONE_ENVIRONMENT` | `"us-east-1"` | Pinecone region |

## 🔧 Extending the Pipeline

### Adding New Document Types
```python
# In src/ingestion/document_parser.py
class CustomParser(DocumentParser):
    def parse(self, file_path_or_bytes):
        # Your parsing logic
        return {'text': extracted_text, 'metadata': metadata}

# Register in DocumentParserFactory
DocumentParserFactory._parsers['.custom'] = CustomParser
```

### Adding New Vector Stores
```python
# In src/ingestion/vector_store.py
class ChromaVectorStore(VectorStore):
    # Implement abstract methods
    pass

# Update factory function
def get_vector_store():
    provider = os.getenv("VECTOR_STORE_PROVIDER", "pinecone")
    if provider == "chroma":
        return ChromaVectorStore()
    return PineconeVectorStore()
```

### Custom Chunking Strategies
```python
# In src/ingestion/chunking.py
class SemanticChunker(Chunker):
    def chunk_text(self, text, metadata=None):
        # Your semantic chunking logic
        pass

# Update factory function
def get_chunker(strategy="recursive", **kwargs):
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    # ... existing strategies
```

## 🧪 Testing

Run the example script to test the full pipeline:
```bash
python example_ingestion.py
```

This will:
1. Create sample documents
2. Test single file ingestion
3. Test bytes/upload ingestion
4. Test directory batch processing
5. Show pipeline statistics
6. Clean up test files

## 🚨 Common Issues

### Missing API Keys
```
RuntimeError: Failed to generate OpenAI embeddings
```
**Solution**: Ensure `OPENAI_API_KEY` is set in your `.env` file

### Pinecone Connection Issues
```
RuntimeError: Failed to create Pinecone index
```
**Solution**: 
- Check `PINECONE_API_KEY` 
- Verify `PINECONE_ENVIRONMENT` matches your account
- Ensure you're within Pinecone's rate limits

### Document Parsing Errors
```
ValueError: Unsupported file type: .xyz
```
**Solution**: The pipeline only supports PDF, DOCX, TXT, and MD files. Convert your document or add a custom parser.

## 📈 Performance Tips

1. **Batch processing**: Use directory ingestion for multiple files
2. **Chunk size**: Larger chunks = fewer API calls but less granular retrieval
3. **Overlap**: More overlap = better context but more storage
4. **Provider choice**: Test both OpenAI and Gemini for your use case

## 🔗 Integration Points

This ingestion pipeline is designed to work with:
- **Streamlit**: For file uploads and user interfaces
- **FastAPI**: For API endpoints and background processing
- **Retrieval modules**: The stored vectors work with the retrieval pipeline
- **Generation modules**: Retrieved context feeds into LLM generation

## 🎯 Next Steps

1. **Set up retrieval**: Implement the retrieval components to query your ingested documents
2. **Add web search**: Integrate external search for hybrid retrieval
3. **Implement reranking**: Add reranking for better result quality
4. **Build UI**: Create Streamlit interface for document upload and management 