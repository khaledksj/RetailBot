# replit.md

## Overview

The Shop Manual Chatbot RAG System is a production-grade Python web application that enables retail shops to upload PDF manuals, index them with high-quality embeddings, and serve an intelligent chatbot that answers questions using Retrieval-Augmented Generation (RAG). The system provides multilingual support (English/Arabic), real-time streaming responses, document deduplication, and citation tracking with page references.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: FastAPI with Uvicorn ASGI server for high-performance async operations
- **API Design**: RESTful endpoints for document ingestion and synchronous chat, with WebSocket support for streaming responses
- **Modular Structure**: Clean separation of concerns with distinct modules for PDF processing, chunking, vector storage, and RAG pipeline

### Frontend Architecture  
- **Technology**: Vanilla HTML/CSS/JavaScript with Bootstrap 5 for responsive design
- **Interface**: Tab-based navigation with separate upload and chat interfaces
- **Internationalization**: Built-in support for English and Arabic with RTL layout support
- **Real-time Communication**: WebSocket integration for streaming chat responses

### Document Processing Pipeline
- **PDF Extraction**: Primary extraction using pypdf with pdfminer.six fallback for robust text extraction
- **Text Chunking**: Recursive document chunking with configurable token limits (1000 tokens) and overlap (150 tokens)
- **Normalization**: Text cleaning and normalization to handle PDF artifacts and formatting issues
- **Deduplication**: Content hash-based duplicate detection to prevent redundant processing

### Vector Storage & Retrieval
- **Storage Backend**: In-memory vector store with pluggable interface for future PostgreSQL + pgvector integration
- **Embeddings**: OpenAI text-embedding-3-large for high-quality semantic representations
- **Search Strategy**: Similarity search with Maximal Marginal Relevance (MMR) reranking to reduce redundancy
- **Metadata Tracking**: Comprehensive chunk metadata including document ID, filename, page numbers, and timestamps

### RAG Implementation
- **Query Processing**: Multi-step pipeline with query embedding, similarity search, and context assembly
- **Response Generation**: GPT-4o-mini with temperature control and streaming support
- **Context Management**: Intelligent context selection with configurable retrieval parameters (top-k=8, final selection=5)
- **Refusal Handling**: Built-in logic to refuse answering questions outside the provided context
- **Citation System**: Automatic source attribution with clickable references including filename and page numbers

### Configuration Management
- **Settings**: Pydantic-based configuration with environment variable support
- **Flexibility**: Configurable models, chunk sizes, retrieval parameters, and rate limits
- **Environment**: .env file support for secure API key management

### Observability & Monitoring
- **Logging**: Structured JSON logging with contextual information for debugging and monitoring
- **Health Checks**: Dedicated endpoints for application and vector store health monitoring
- **Debug Tools**: Debug search endpoint for testing retrieval quality and troubleshooting

### Security & Rate Limiting
- **Input Validation**: Comprehensive request validation with file type and size restrictions
- **Rate Limiting**: Configurable request rate limits to prevent abuse
- **Error Handling**: Graceful error handling with appropriate HTTP status codes and user-friendly messages

## External Dependencies

### AI/ML Services
- **OpenAI API**: GPT-4o-mini for chat completions and text-embedding-3-large for embeddings
- **API Key Management**: Secure storage via environment variables

### Python Libraries
- **Web Framework**: FastAPI for API endpoints, Uvicorn for ASGI serving
- **PDF Processing**: pypdf (primary), pdfminer.six (fallback)
- **ML/AI**: OpenAI Python client, tiktoken for tokenization, numpy for vector operations
- **Configuration**: Pydantic for settings management, python-dotenv for environment loading
- **Development**: pytest for testing, asyncio for async operations

### Frontend Dependencies
- **CSS Framework**: Bootstrap 5 for responsive design and components
- **Icons**: Font Awesome for iconography
- **Browser APIs**: WebSocket API for real-time communication, File API for drag-and-drop uploads

### Future Database Integration
- **Vector Database**: Architecture prepared for PostgreSQL with pgvector extension
- **Alternative Storage**: Chroma support planned for local development
- **Migration Path**: Pluggable vector store interface allows seamless backend switching