# 💬 Conversational RAG with Memory

This project now includes **conversational RAG capabilities** with memory management, allowing for multi-turn conversations that maintain context and remember previous interactions.

## 🎯 Features

### Core Conversational Features
- **Multi-turn conversations** with context preservation
- **Memory management** with automatic trimming and summarization
- **Thread-based conversations** for parallel chat sessions
- **Hybrid retrieval** combining vector search and web search
- **Source citation** and conversation history tracking

### Memory Management
- **Automatic message trimming** to manage context window size
- **Summary memory** for long conversations
- **Configurable memory limits** and thresholds
- **Thread isolation** for separate conversation contexts

### Integration
- **Seamless integration** with existing RAG pipeline
- **Streamlit interface** with dedicated chat functionality
- **Backward compatibility** with non-conversational queries

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `langgraph>=0.2.78`
- `langgraph-checkpoint>=2.1.8`
- `langgraph-checkpoint-sqlite>=2.1.8`
- `langgraph-sdk>=0.1.70`

### 2. Basic Usage

```python
from src.generation.conversational_rag import create_conversational_rag
from langchain_core.documents import Document

# Create conversational RAG system
conv_rag = create_conversational_rag(
    llm_provider="auto",  # or "openai", "gemini"
    max_messages=10,      # Memory limit
    enable_summary=True,  # Enable summary for long conversations
    summary_threshold=8   # Summarize after 8 messages
)

# Create context documents
docs = [
    Document(
        page_content="Machine learning is...",
        metadata={"title": "ML Guide", "source": "book"}
    )
]

# Start conversation
response = conv_rag.chat(
    message="What is machine learning?",
    context_documents=docs,
    thread_id="user_123"
)

print(response['response'])

# Continue conversation
response = conv_rag.chat(
    message="Can you tell me more about the types?",
    context_documents=docs,
    thread_id="user_123"  # Same thread maintains context
)
```

### 3. Using with RAG Pipeline

```python
from src.rag_pipeline import create_conversational_rag_pipeline

# Create conversational RAG pipeline
pipeline = create_conversational_rag_pipeline(
    index_name="my-documents",
    llm_provider="openai",
    max_messages=12,
    enable_summary=True,
    summary_threshold=10
)

# Chat with full RAG pipeline
response = pipeline.chat(
    message="What are the key findings in the research?",
    thread_id="research_discussion"
)

# Continue conversation with full context
response = pipeline.chat(
    message="How do these findings compare to previous work?",
    thread_id="research_discussion"
)
```

## 🎛️ Configuration

### Memory Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | int | 10 | Maximum messages to keep in memory |
| `enable_summary` | bool | True | Enable summary memory for long conversations |
| `summary_threshold` | int | 8 | Number of messages before creating summary |

### LLM Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_provider` | str | "auto" | LLM provider ("openai", "gemini", "auto") |
| `model` | str | None | Specific model name |
| `temperature` | float | 0.0 | Generation temperature |

### Example Configuration

```python
# High-memory configuration for detailed conversations
conv_rag = create_conversational_rag(
    llm_provider="openai",
    model="gpt-4o-mini",
    temperature=0.1,
    max_messages=20,
    enable_summary=True,
    summary_threshold=15
)

# Low-memory configuration for simple chats
conv_rag = create_conversational_rag(
    llm_provider="auto",
    max_messages=6,
    enable_summary=False,
    summary_threshold=10
)
```

## 🎨 Streamlit Interface

### New Chat Interface

The Streamlit app now includes a dedicated **"💬 Conversational Chat"** tab with:

- **Real-time conversation** with memory
- **Thread management** for multiple conversations
- **Memory configuration** controls
- **Conversation history** display
- **Source citation** and retrieval stats

### Features

1. **Chat Configuration**
   - Thread ID management
   - Memory settings (max messages, summary threshold)
   - LLM provider selection
   - Web search integration

2. **Conversation Management**
   - Clear conversation history
   - Thread switching
   - Message history display

3. **Enhanced Single Query**
   - Option to enable conversation memory
   - Configurable memory settings
   - Seamless switching between modes

## 📊 Memory Management

### Automatic Trimming

When conversations exceed the `max_messages` limit, the system automatically trims older messages:

```python
# Example: max_messages=6
# Conversation: [msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8]
# After trimming: [msg3, msg4, msg5, msg6, msg7, msg8]  # Keeps last 6
```

### Summary Memory

For long conversations, the system can create summaries to preserve context:

```python
# Original conversation:
# User: "What is ML?"
# Assistant: "Machine learning is..."
# User: "What are the types?"
# Assistant: "There are three main types..."

# Summary created:
# "Summary: User asked about machine learning and its types. 
#  Discussed definition and three main types: supervised, unsupervised, reinforcement."
```

### Thread Management

Different conversation threads maintain separate memory:

```python
# Thread 1: Research discussion
pipeline.chat("What are the findings?", thread_id="research")
pipeline.chat("How significant are they?", thread_id="research")

# Thread 2: Technical implementation
pipeline.chat("How do I implement this?", thread_id="technical")
pipeline.chat("What are the requirements?", thread_id="technical")

# Each thread maintains its own context
```

## 🔧 Advanced Usage

### Custom Memory Strategies

```python
from src.generation.conversational_rag import ConversationalRAG

# Create with custom settings
conv_rag = ConversationalRAG(
    llm_provider="openai",
    max_messages=15,
    enable_summary=True,
    summary_threshold=12
)

# Get conversation history
history = conv_rag._get_conversation_history("thread_id")

# Clear specific conversation
conv_rag.clear_conversation("thread_id")
```

### Integration with Existing Pipeline

```python
from src.rag_pipeline import create_rag_pipeline

# Create pipeline with conversation enabled
pipeline = create_rag_pipeline(
    index_name="documents",
    enable_conversation=True,
    max_conversation_messages=10,
    enable_summary=True,
    summary_threshold=8
)

# Use chat method for conversational queries
response = pipeline.chat("Question?", thread_id="user_session")

# Use query method for one-off queries
response = pipeline.query("Single question?")
```

## 🧪 Testing

### Running Tests

```bash
# Run conversational RAG tests
python test_conversational_rag.py

# Run full RAG pipeline tests
python test_rag_pipeline.py
```

### Test Coverage

The test suite includes:

1. **Basic Conversational RAG** - Multi-turn conversations
2. **Memory Trimming** - Automatic message limiting
3. **Summary Memory** - Long conversation summarization
4. **Pipeline Integration** - Full RAG pipeline with conversation
5. **Thread Management** - Separate conversation contexts

## 📚 Implementation Details

### Architecture

```
User Message
     ↓
RAG Pipeline (retrieve & rerank documents)
     ↓
ConversationalRAG (with memory management)
     ↓
LangGraph State Management
     ↓
Memory Trimming/Summary (if needed)
     ↓
LLM Generation (with context + history)
     ↓
Response + Updated Memory
```

### Memory Flow

1. **Message Input**: User sends message
2. **Context Retrieval**: RAG pipeline retrieves relevant documents
3. **Memory Check**: System checks conversation history length
4. **Memory Management**: 
   - If under threshold: Use trimming
   - If over threshold: Create summary
5. **LLM Generation**: Generate response with context + memory
6. **Memory Update**: Store new message and response

### Key Components

- **`ConversationalRAG`**: Core conversational system with memory
- **`RAGPipeline`**: Extended to support conversation mode
- **`LangGraph`**: State management and memory persistence
- **`MemorySaver`**: In-memory conversation storage
- **`MessagesState`**: Message history management

## ⚠️ Limitations & Considerations

### Current Limitations

1. **Memory Storage**: Currently uses in-memory storage (sessions reset on restart)
2. **Concurrent Users**: Not optimized for high-concurrency scenarios
3. **Token Limits**: Summary creation counts toward token usage
4. **Model Dependency**: Requires LLM API access for conversation

### Performance Considerations

- **Memory Usage**: Longer conversations use more memory
- **Token Costs**: Summary generation adds to token usage
- **Response Time**: Memory processing adds slight latency

### Future Enhancements

- **Persistent Memory**: Database storage for conversation history
- **Advanced Summarization**: More sophisticated summary strategies
- **User Authentication**: Per-user conversation management
- **Analytics**: Conversation quality metrics

## 🎉 Benefits

### For Users
- **Natural Conversations**: Multi-turn dialogues with context
- **Better Understanding**: AI remembers previous discussion
- **Efficient Interaction**: No need to repeat context
- **Personalized Experience**: Thread-based conversations

### For Developers
- **Easy Integration**: Simple API for conversation features
- **Flexible Configuration**: Customizable memory management
- **Backward Compatibility**: Works with existing RAG setup
- **Extensible**: Built on LangGraph for advanced workflows

## 📖 Examples

### Research Assistant

```python
# Research conversation
conv_rag.chat("What does the paper say about climate change?", thread_id="research")
conv_rag.chat("What are the main findings?", thread_id="research")
conv_rag.chat("How do these compare to previous studies?", thread_id="research")
conv_rag.chat("What are the implications?", thread_id="research")
```

### Technical Support

```python
# Technical support conversation
conv_rag.chat("How do I install the software?", thread_id="support_user_123")
conv_rag.chat("I'm getting an error, what should I do?", thread_id="support_user_123")
conv_rag.chat("That didn't work, any other suggestions?", thread_id="support_user_123")
```

### Educational Tutoring

```python
# Educational conversation
conv_rag.chat("Explain machine learning to me", thread_id="student_alice")
conv_rag.chat("Can you give me an example?", thread_id="student_alice")
conv_rag.chat("How is this different from traditional programming?", thread_id="student_alice")
```

---

## 🔗 Related Documentation

- [Main RAG Pipeline Documentation](README_RAG_PIPELINE.md)
- [Streamlit Interface Guide](STREAMLIT_README.md)
- [LangChain Memory Documentation](https://python.langchain.com/docs/how_to/chatbots_memory/)

---

**Happy Chatting! 💬✨** 