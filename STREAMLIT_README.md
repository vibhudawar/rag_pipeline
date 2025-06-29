# 📚 RAG Ingestion Pipeline - Streamlit UI

A beautiful, interactive web interface for testing your RAG (Retrieval-Augmented Generation) ingestion pipeline. Upload documents, process them, and test retrieval - all through an intuitive web interface!

## 🚀 Quick Start

### 1. Set up Environment Variables

Create a `.env` file in your project root:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here

# Optional API Keys
GEMINI_API_KEY=your-gemini-api-key-here

# Service Selection (optional)
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai

# Configuration (optional)
PINECONE_ENVIRONMENT=us-east-1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNKING_STRATEGY=recursive
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Features

### 📄 Single Document Upload
- **Drag & drop or browse** to upload PDF, DOCX, TXT, or MD files
- **Real-time processing** with progress indicators
- **Custom metadata** - Add categories, departments, versions, etc.
- **Instant feedback** - See processing results and statistics
- **Auto-processing** - Option to process files immediately on upload

### 📚 Batch Document Upload
- **Multiple file upload** - Process several documents at once
- **Batch metadata** - Apply common metadata to all files
- **Progress tracking** - See real-time status for each file
- **Error handling** - Option to stop on first error or continue
- **Visual results** - Charts and tables showing batch statistics

### 🔍 Search & Test
- **Semantic search** - Test retrieval from your ingested documents
- **Configurable results** - Choose how many results to return
- **Detailed results** - See content, scores, and metadata
- **Session tracking** - View recently ingested documents

### ⚙️ Configuration & Stats
- **Pipeline settings** - Adjust chunk size, overlap, and strategy
- **Index management** - View and switch between Pinecone indexes
- **Environment info** - See current configuration and API settings
- **Session management** - Track and clear session data

## 🖥️ Interface Overview

### Sidebar Configuration
- **Index Name**: Choose which Pinecone index to use
- **Chunking Strategy**: Recursive or token-based splitting
- **Chunk Size**: Control document chunk sizes
- **Chunk Overlap**: Set overlap between chunks
- **File Types**: View supported formats

### Main Tabs

#### 1. 📄 Single Upload Tab
```
┌─────────────────────────────────────┐
│ 📄 Single Document Upload          │
├─────────────────────────────────────┤
│ [File Uploader]                     │
│                                     │
│ 📋 Additional Metadata (Optional)   │
│ ├─ Category: ___________            │
│ ├─ Department: _________            │
│ ├─ Priority: [Dropdown]             │
│ └─ Version: ____________            │
│                                     │
│ ☑ Show detailed metadata           │
│ ☑ Auto-process on upload           │
│                                     │
│ 📁 File Information                 │
│ ┌─────┬──────┬───────────┐         │
│ │Name │ Size │   Type    │         │
│ └─────┴──────┴───────────┘         │
│                                     │
│ [🚀 Process Document]              │
│                                     │
│ ✅ Document ingested successfully!  │
│ ┌──────┬──────┬──────┬──────┐     │
│ │ Name │ Type │Chunks│ Time │     │
│ └──────┴──────┴──────┴──────┘     │
└─────────────────────────────────────┘
```

#### 2. 📚 Batch Upload Tab
```
┌─────────────────────────────────────┐
│ 📚 Batch Document Upload            │
├─────────────────────────────────────┤
│ [Multiple File Uploader]            │
│                                     │
│ Batch Settings:                     │
│ ├─ Category: ___________            │
│ ├─ Department: _________            │
│ ├─ Priority: [Dropdown]             │
│ └─ ☑ Stop on first error           │
│                                     │
│ 📁 Selected Files (3 files)         │
│ ┌─────────────┬──────┬──────┐      │
│ │ Filename    │ Size │ Type │      │
│ ├─────────────┼──────┼──────┤      │
│ │ doc1.pdf    │ 45KB │ PDF  │      │
│ │ doc2.txt    │ 12KB │ TXT  │      │
│ │ doc3.docx   │ 89KB │DOCX  │      │
│ └─────────────┴──────┴──────┘      │
│                                     │
│ [🚀 Process All Documents]         │
│                                     │
│ 📊 Batch Processing Summary         │
│ ┌─────┬────┬─────┬──────┬──────┐   │
│ │Total│ ✅ │  ❌ │Chunks│ Time │   │
│ │  3  │ 2  │  1  │  24  │ 12s │   │
│ └─────┴────┴─────┴──────┴──────┘   │
│                                     │
│ 📈 [Success/Failure Pie Chart]     │
│ 📊 [Chunks per File Bar Chart]     │
└─────────────────────────────────────┘
```

#### 3. 🔍 Search & Test Tab
```
┌─────────────────────────────────────┐
│ 🔍 Search & Test                    │
├─────────────────────────────────────┤
│ 📚 Recently Ingested Documents      │
│ ┌─────────────┬─────────┬──────┐    │
│ │ Filename    │  Index  │Chunks│    │
│ ├─────────────┼─────────┼──────┤    │
│ │ doc1.pdf    │rag-docs │  8   │    │
│ │ doc2.txt    │rag-docs │  4   │    │
│ └─────────────┴─────────┴──────┘    │
│                                     │
│ 🔍 Search Documents                 │
│ Query: [________________] [5 ▼]     │
│                                     │
│ [🔍 Search]                        │
│                                     │
│ 📄 Search Results                   │
│ ┌─────────────────────────────────┐ │
│ │ 📄 Result 1 (Score: 0.856)     │ │
│ │ ├─ Content: [Text preview...]   │ │
│ │ └─ Metadata: [Key-value pairs] │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

#### 4. ⚙️ Config & Stats Tab
```
┌─────────────────────────────────────┐
│ ⚙️ Configuration & Statistics       │
├─────────────────────────────────────┤
│ ⚙️ Current Configuration            │
│ ┌──────────────┬──────────────────┐ │
│ │ Index Name   │ Vector Store     │ │
│ │ rag-docs     │ PineconeVector.. │ │
│ ├──────────────┼──────────────────┤ │
│ │ Embedder     │ Chunking         │ │
│ │ OpenAI..     │ recursive        │ │
│ └──────────────┴──────────────────┘ │
│                                     │
│ 🗄️ Vector Store Information         │
│ ┌──────────────┬──────────────────┐ │
│ │ Available    │ Current Index    │ │
│ │ Indexes: 3   │ Status: ✅ Exists│ │
│ └──────────────┴──────────────────┘ │
│                                     │
│ 📋 Available Indexes                │
│ 🎯 rag-docs (Current)              │
│ 📁 test-index                      │
│ 📁 backup-index                    │
│                                     │
│ 🌍 Environment Information          │
│ EMBEDDING_PROVIDER: openai         │
│ CHUNK_SIZE: 1000                   │
│ [... other environment vars ...]   │
└─────────────────────────────────────┘
```

## 🎨 Features Highlights

### Real-time Progress Tracking
- **Visual progress bars** during document processing
- **Status messages** showing current step
- **Processing times** for performance monitoring

### Rich Visualizations
- **Success/failure pie charts** for batch operations
- **Chunks per file bar charts** for analysis
- **Interactive data tables** with sorting and filtering

### Error Handling & Help
- **Descriptive error messages** with suggested solutions
- **Troubleshooting tips** in expandable sections
- **Environment validation** on app startup

### Session Management
- **Track ingested documents** across the session
- **Cross-tab functionality** - upload in one tab, search in another
- **Session data persistence** until browser refresh

## 🚨 Common Issues & Solutions

### Missing API Keys
```
❌ Missing required environment variables: OPENAI_API_KEY, PINECONE_API_KEY
```
**Solution**: Create a `.env` file with your API keys (see setup section above)

### File Upload Issues
```
❌ Unsupported file type: .xyz
```
**Solution**: Only PDF, DOCX, TXT, and MD files are supported

### Search Not Working
```
❌ Search failed: No results found
```
**Solutions**:
- Make sure documents are uploaded and processed first
- Check that you're using the correct index name
- Verify your embeddings are working correctly

## 🔧 Customization

### Changing Default Settings
Edit the configuration in the sidebar:
- **Index Name**: Change the target Pinecone index
- **Chunk Size**: Adjust for your document types
- **Chunk Overlap**: Tune for better retrieval
- **Chunking Strategy**: Choose recursive or token-based

### Adding Custom Metadata
Use the metadata fields to organize your documents:
- **Category**: Document type or subject
- **Department**: Organizational unit
- **Version**: Document version tracking
- **Priority**: Processing priority

## 🚀 Tips for Best Results

1. **Start Small**: Upload 1-2 test documents first
2. **Check Configuration**: Verify settings in the Config tab
3. **Test Search**: Use the Search tab to validate ingestion
4. **Monitor Performance**: Watch processing times for optimization
5. **Use Metadata**: Add meaningful metadata for better organization

## 🔗 Integration

This Streamlit app is designed to work seamlessly with:
- **FastAPI backends** for production deployment
- **Jupyter notebooks** for experimentation
- **CI/CD pipelines** for automated document processing
- **External data sources** through file uploads

---

**Happy document processing! 🎉**

For more information, see the main [INGESTION_README.md](INGESTION_README.md) 