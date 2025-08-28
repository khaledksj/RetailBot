"""
PDF processing utilities for the Shop Manual Chatbot RAG system.
"""

import io
from typing import List, Optional
import pypdf
from pypdf import PdfReader
from pdfminer.high_level import extract_text

from app.core.logging import get_logger

logger = get_logger(__name__)

class PDFProcessor:
    """PDF processor with multiple extraction backends."""
    
    async def extract_text(self, pdf_content: bytes) -> List[str]:
        """
        Extract text from PDF content.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            List of text strings, one per page
        """
        logger.info(f"Extracting text from PDF", extra={
            "pdf_size_bytes": len(pdf_content)
        })
        
        # Try pypdf first
        try:
            pages_text = await self._extract_with_pypdf(pdf_content)
            if pages_text and any(page.strip() for page in pages_text):
                logger.info(f"Successfully extracted text with pypdf", extra={
                    "pages_count": len(pages_text),
                    "total_chars": sum(len(page) for page in pages_text)
                })
                return pages_text
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {str(e)}")
        
        # Fallback to pdfminer
        try:
            pages_text = await self._extract_with_pdfminer(pdf_content)
            if pages_text and any(page.strip() for page in pages_text):
                logger.info(f"Successfully extracted text with pdfminer", extra={
                    "pages_count": len(pages_text),
                    "total_chars": sum(len(page) for page in pages_text)
                })
                return pages_text
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {str(e)}")
        
        logger.error("All PDF extraction methods failed")
        return []
    
    async def _extract_with_pypdf(self, pdf_content: bytes) -> List[str]:
        """Extract text using pypdf library."""
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        
        pages_text = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                pages_text.append(text or "")
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                pages_text.append("")
        
        return pages_text
    
    async def _extract_with_pdfminer(self, pdf_content: bytes) -> List[str]:
        """Extract text using pdfminer library."""
        pdf_file = io.BytesIO(pdf_content)
        
        # Extract all text (pdfminer doesn't provide easy per-page extraction)
        try:
            full_text = extract_text(pdf_file)
        except Exception as e:
            logger.error(f"pdfminer full text extraction failed: {str(e)}")
            return []
        
        # Simple heuristic to split by pages
        # This is not perfect but works for many PDFs
        pages = self._split_text_by_pages(full_text)
        
        return pages
    
    def _split_text_by_pages(self, text: str) -> List[str]:
        """
        Split extracted text into pages using heuristics.
        This is a simple approach and may not work perfectly for all PDFs.
        """
        # Common page break indicators
        page_break_patterns = [
            '\f',  # Form feed character
            '\n\n\n',  # Multiple newlines
        ]
        
        # Try form feed first (most reliable)
        if '\f' in text:
            pages = text.split('\f')
            return [page.strip() for page in pages if page.strip()]
        
        # Fallback: split by multiple newlines and try to identify page boundaries
        paragraphs = text.split('\n\n')
        
        # If we have a reasonable number of paragraphs, group them
        if len(paragraphs) > 5:
            # Simple heuristic: assume roughly equal distribution
            pages_count = max(1, len(paragraphs) // 10)  # Assume ~10 paragraphs per page
            page_size = len(paragraphs) // pages_count
            
            pages = []
            for i in range(0, len(paragraphs), page_size):
                page_paragraphs = paragraphs[i:i + page_size]
                page_text = '\n\n'.join(page_paragraphs)
                if page_text.strip():
                    pages.append(page_text.strip())
            
            return pages
        
        # Last resort: return as single page
        return [text.strip()] if text.strip() else []
    
    def validate_pdf(self, pdf_content: bytes) -> bool:
        """Validate that the content is a valid PDF."""
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            
            # Check if we can read at least the number of pages
            page_count = len(reader.pages)
            return page_count > 0
            
        except Exception as e:
            logger.warning(f"PDF validation failed: {str(e)}")
            return False
