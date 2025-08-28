# Shop Manual Chatbot RAG System

A production-grade Python web application that enables retail shops to upload PDF manuals, index them with high-quality embeddings, and serve a chatbot that answers questions using Retrieval-Augmented Generation (RAG).

## Features

- **PDF Upload & Ingestion**: Upload multiple PDF manuals with automatic text extraction and chunking
- **Vector Search**: High-quality embeddings using OpenAI's text-embedding-3-large model
- **RAG Chatbot**: Intelligent responses using GPT-4o-mini with retrieval-augmented generation
- **Citation System**: Clickable source references with filename and page numbers
- **Multilingual Support**: English and Arabic interface with RTL support
- **Streaming Responses**: Real-time chat responses via WebSocket
- **Document Deduplication**: Prevents duplicate uploads with content hashing
- **Responsive Design**: Clean, modern UI that works on desktop and mobile

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Python 3.11+
- **AI/ML**: OpenAI GPT-4o-mini, text-embedding-3-large
- **Vector Storage**: In-memory (configurable for PostgreSQL + pgvector)
- **PDF Processing**: pypdf with pdfminer.six fallback
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Logging**: Structured JSON logging

## Quick Start

1. **Clone and Setup**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd shop-manual-chatbot
   
   # Copy environment configuration
   cp .env.example .env
   ```

2. **Configure Environment**
   Edit `.env` file with your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn openai pypdf pdfminer.six numpy pydantic python-dotenv
   ```

4. **Run the Application**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the Application**
   Open your browser to `http://localhost:8000`

## Usage

### Upload Documents

1. Navigate to the "Upload Documents" tab
2. Drag and drop PDF files or click to browse
3. Optionally enable "Force re-ingest" to update existing documents
4. Click "Upload Documents" to process

### Chat with Documents

1. Switch to the "Chat" tab
2. Type your question about the uploaded manuals
3. Adjust temperature slider for response creativity (0.0-0.5)
4. View cited sources below the response

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `VECTOR_BACKEND` | `memory` | Vector storage backend |
| `MAX_CHUNK_TOKENS` | `1000` | Maximum tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `150` | Overlap between chunks |
| `TOP_K` | `8` | Initial retrieval count |
| `RETRIEVAL_TOP_N` | `5` | Final chunks for context |
| `MAX_INPUT_TOKENS` | `4000` | Maximum query length |
| `DEFAULT_TEMPERATURE` | `0.2` | Default response randomness |

### Vector Backends

Currently supported:
- **memory**: In-memory storage (default, for development)
- **pgvector**: PostgreSQL with pgvector extension (TODO)
- **chroma**: Chroma vector database (TODO)

To switch backends, update the `VECTOR_BACKEND` environment variable.

## API Endpoints

### Document Ingestion
- `POST /api/ingest` - Upload and process PDF documents
- `GET /api/health` - Health check with system status

### Chat
- `POST /api/chat` - Synchronous chat endpoint
- `WebSocket /ws/chat` - Streaming chat responses

### Debug
- `GET /api/debug/search?query=...` - Debug retrieval results

## Architecture

