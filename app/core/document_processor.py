"""
Enhanced document processing with Arabic text extraction and OCR fallback.
"""

import io
import os
import tempfile
from typing import List
from .arabic_extractor import extract_text_safely, extraction_health
from app.core.logging import get_logger

logger = get_logger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced document processor with Arabic support and OCR fallback."""
    
    async def extract_text(self, document_content: bytes, file_type: str = "pdf", filename: str = "unknown") -> List[str]:
        """
        Extract text from document content with Arabic support and OCR fallback.
        
        Args:
            document_content: Document file content as bytes
            file_type: Type of document ("pdf" or "docx")
            filename: Original filename for logging
            
        Returns:
            List of text strings, one per page
        """
        if not document_content:
            logger.warning(f"Empty document content for {filename}")
            return []
        
        try:
            # Write content to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as temp_file:
                temp_file.write(document_content)
                temp_path = temp_file.name
            
            try:
                # Use the enhanced Arabic extractor
                text = extract_text_safely(temp_path, expect_arabic=True)
                
                # Log extraction health
                health = extraction_health(text)
                logger.info({
                    "event": "extraction_done",
                    "filename": filename,
                    "metrics": health
                })
                
                # Alert if extraction quality is poor
                if health["arabic_ratio"] < 0.05 and health["length"] > 0:
                    logger.warning({
                        "event": "extraction_suspect",
                        "filename": filename,
                        "health": health["health"]
                    })
                
                # Split text into pages (simple heuristic)
                if text:
                    # For now, treat as single page - can be enhanced to detect page breaks
                    pages = [text]
                    logger.info(f"Successfully extracted {len(text)} characters from {filename}")
                    return pages
                else:
                    logger.error(f"No text extracted from {filename}")
                    return []
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Document processing failed for {filename}: {str(e)}")
            return []