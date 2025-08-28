"""
PDF ingestion API endpoints for the Shop Manual Chatbot RAG system.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
import hashlib
import asyncio
from datetime import datetime

from app.core.logging import get_logger
from app.core.models import IngestResponse, DocumentInfo
from app.core.pdf import PDFProcessor
from app.core.chunker import DocumentChunker
from app.core.llm import EmbeddingService
from app.core.db import get_vector_store
from app.core.settings import get_settings

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(
    files: List[UploadFile] = File(...),
    force: bool = Form(False)
):
    """
    Ingest one or more PDF files into the vector database.
    
    - **files**: List of PDF files to ingest
    - **force**: If True, re-ingest even if file already exists
    """
    logger.info(f"Ingestion started", extra={
        "file_count": len(files),
        "force_reingest": force
    })
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types and sizes
    for file in files:
        if not file.content_type == "application/pdf":
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a PDF"
            )
    
    vector_store = get_vector_store()
    pdf_processor = PDFProcessor()
    chunker = DocumentChunker()
    embedding_service = EmbeddingService()
    
    results = []
    total_chunks = 0
    
    for file in files:
        try:
            filename = file.filename or "unknown.pdf"
            logger.info(f"Processing file: {filename}")
            
            # Read file content
            content = await file.read()
            
            # Check file size (20MB limit per file)
            if len(content) > 20 * 1024 * 1024:
                results.append(DocumentInfo(
                    filename=filename,
                    status="failed",
                    error="File size exceeds 20MB limit",
                    pages_processed=0,
                    chunks_created=0
                ))
                continue
            
            # Calculate content hash for deduplication
            content_hash = hashlib.sha256(content).hexdigest()
            
            # Check if document already exists
            if not force and await vector_store.document_exists(content_hash):
                results.append(DocumentInfo(
                    filename=filename,
                    status="skipped",
                    error="Document already exists (use force=true to re-ingest)",
                    pages_processed=0,
                    chunks_created=0
                ))
                continue
            
            # Extract text from PDF
            pages_text = await pdf_processor.extract_text(content)
            
            if not pages_text:
                results.append(DocumentInfo(
                    filename=filename,
                    status="failed",
                    error="No text could be extracted from PDF",
                    pages_processed=0,
                    chunks_created=0
                ))
                continue
            
            # Create document chunks
            all_chunks = []
            for page_num, page_text in enumerate(pages_text, 1):
                if page_text.strip():
                    page_chunks = chunker.chunk_text(
                        text=page_text,
                        page_number=page_num,
                        filename=filename
                    )
                    all_chunks.extend(page_chunks)
            
            if not all_chunks:
                results.append(DocumentInfo(
                    filename=filename,
                    status="failed",
                    error="No valid chunks could be created",
                    pages_processed=len(pages_text),
                    chunks_created=0
                ))
                continue
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = await embedding_service.create_embeddings(chunk_texts)
            
            # Store in vector database
            doc_id = await vector_store.store_document(
                filename=filename,
                content_hash=content_hash,
                chunks=all_chunks,
                embeddings=embeddings
            )
            
            total_chunks += len(all_chunks)
            
            results.append(DocumentInfo(
                filename=filename,
                status="success",
                pages_processed=len(pages_text),
                chunks_created=len(all_chunks),
                doc_id=doc_id
            ))
            
            logger.info(f"Successfully processed {filename}", extra={
                "pages": len(pages_text),
                "chunks": len(all_chunks),
                "doc_id": doc_id
            })
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            results.append(DocumentInfo(
                filename=filename,
                status="failed",
                error=str(e),
                pages_processed=0,
                chunks_created=0
            ))
    
    logger.info(f"Ingestion completed", extra={
        "total_files": len(files),
        "successful": len([r for r in results if r.status == "success"]),
        "total_chunks": total_chunks
    })
    
    return IngestResponse(
        success=True,
        message=f"Processed {len(files)} files, created {total_chunks} chunks",
        documents=results
    )
