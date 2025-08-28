"""
Document chunking utilities for the Shop Manual Chatbot RAG system.
"""

import re
import tiktoken
from typing import List, Optional
from uuid import uuid4

from app.core.settings import get_settings
from app.core.logging import get_logger
from app.core.models import Chunk

logger = get_logger(__name__)
settings = get_settings()

class DocumentChunker:
    """Document chunker with semantic and token-based splitting."""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = settings.max_chunk_tokens
        self.overlap_tokens = settings.chunk_overlap_tokens
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Zero-width chars
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Non-ASCII chars (simple approach)
        
        return text.strip()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _split_by_semantic_boundaries(self, text: str) -> List[str]:
        """Split text by semantic boundaries (headings, lists, paragraphs)."""
        chunks = []
        
        # Split by double newlines (paragraph boundaries)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with current paragraph
                if self._count_tokens(paragraph) <= self.max_tokens:
                    current_chunk = paragraph
                else:
                    # Paragraph itself is too long, split it by sentences
                    sentence_chunks = self._split_by_sentences(paragraph)
                    chunks.extend(sentence_chunks[:-1])  # Add all but last
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences when paragraph is too long."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is still too long, use token-based splitting
                if self._count_tokens(sentence) > self.max_tokens:
                    token_chunks = self._split_by_tokens(sentence)
                    chunks.extend(token_chunks[:-1])
                    current_chunk = token_chunks[-1] if token_chunks else ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text by token windows with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            if end >= len(tokens):
                break
            start = end - self.overlap_tokens
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get overlap tokens from previous chunk
            prev_tokens = self.tokenizer.encode(prev_chunk)
            if len(prev_tokens) > self.overlap_tokens:
                overlap_tokens = prev_tokens[-self.overlap_tokens:]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                
                # Prepend overlap to current chunk
                current_chunk = overlap_text + " " + current_chunk
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def chunk_text(
        self,
        text: str,
        page_number: int,
        filename: str,
        doc_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk text from a document page.
        
        Args:
            text: Raw text to chunk
            page_number: Page number in the document
            filename: Source filename
            doc_id: Document ID (optional)
        
        Returns:
            List of Chunk objects
        """
        logger.info(f"Chunking text from {filename} page {page_number}", extra={
            "document_name": filename,
            "page": page_number,
            "text_length": len(text)
        })
        
        # Normalize text
        normalized_text = self._normalize_text(text)
        
        if not normalized_text:
            return []
        
        # Try semantic splitting first
        text_chunks = self._split_by_semantic_boundaries(normalized_text)
        
        # Add overlap between chunks
        if len(text_chunks) > 1:
            text_chunks = self._add_overlap(text_chunks)
        
        # Create Chunk objects
        chunks = []
        for chunk_idx, chunk_text in enumerate(text_chunks):
            token_count = self._count_tokens(chunk_text)
            
            chunk = Chunk(
                chunk_id=str(uuid4()),
                doc_id=doc_id or str(uuid4()),
                filename=filename,
                page=page_number,
                chunk_idx=chunk_idx,
                content=chunk_text,
                content_tokens=token_count,
                embedding=[]  # Will be filled later
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {filename} page {page_number}", extra={
            "chunks_created": len(chunks),
            "total_tokens": sum(chunk.content_tokens for chunk in chunks)
        })
        
        return chunks
