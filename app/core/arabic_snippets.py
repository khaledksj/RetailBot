"""
Arabic-aware snippet generation for RAG responses.
"""

import regex as re
from typing import List, Dict, Any

# Arabic sentence boundaries
BOUNDARIES = r'[\.!\?؟؛،]'

def make_snippet(txt: str, hit_idx: int, radius: int = 240) -> str:
    """
    Create a snippet around a hit index, respecting Arabic punctuation.
    
    Args:
        txt: Full text
        hit_idx: Index of the match/hit
        radius: Characters to include around the hit
        
    Returns:
        Snippet string with proper boundaries
    """
    if not txt or hit_idx < 0:
        return ""
    
    # Calculate initial boundaries
    start = max(0, hit_idx - radius)
    end = min(len(txt), hit_idx + radius)
    
    # Try to extend to natural sentence boundaries
    # Look backwards for sentence start
    left_text = txt[:start][::-1]  # Reverse for regex search
    left_match = re.search(BOUNDARIES, left_text)
    if left_match:
        start = max(0, start - left_match.start())
    
    # Look forwards for sentence end
    right_text = txt[end:]
    right_match = re.search(BOUNDARIES, right_text)
    if right_match:
        end = min(len(txt), end + right_match.end())
    
    snippet = txt[start:end].strip()
    
    # Add ellipsis if we're not at the document boundaries
    if start > 0:
        snippet = "..." + snippet
    if end < len(txt):
        snippet = snippet + "..."
    
    return snippet

def create_source_snippets(chunks: List[Dict[str, Any]], max_snippet_length: int = 300) -> List[Dict[str, Any]]:
    """
    Create snippets for source chunks with Arabic-aware boundary detection.
    
    Args:
        chunks: List of chunk dictionaries with 'content', 'filename', 'page' keys
        max_snippet_length: Maximum length for each snippet
        
    Returns:
        List of source dictionaries with snippets
    """
    sources = []
    
    for chunk in chunks:
        content = chunk.get('content', '')
        filename = chunk.get('filename', 'unknown')
        page = chunk.get('page', 1)
        
        # Create snippet from the beginning of the chunk
        if content:
            snippet = make_snippet(content, 0, max_snippet_length // 2)
            # Ensure snippet isn't too long
            if len(snippet) > max_snippet_length:
                snippet = snippet[:max_snippet_length - 3] + "..."
        else:
            snippet = ""
        
        sources.append({
            "filename": filename,
            "page": page,
            "snippet": snippet
        })
    
    return sources