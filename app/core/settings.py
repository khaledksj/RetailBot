"""
Application settings and configuration for the Shop Manual Chatbot RAG system.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4o-mini"
    
    # Vector Database Configuration
    vector_backend: str = "memory"  # memory, supabase, pgvector, chroma
    supabase_url: Optional[str] = os.getenv("DATABASE_URL")
    db_url: Optional[str] = None
    
    # Chunking Configuration
    max_chunk_tokens: int = 1000
    chunk_overlap_tokens: int = 150
    
    # Retrieval Configuration
    top_k: int = 8
    retrieval_top_n: int = 5
    
    # Chat Configuration
    max_input_tokens: int = 4000
    default_temperature: float = 0.2
    max_temperature: float = 0.5
    
    # File Upload Configuration
    max_file_size_mb: int = 30
    max_total_size_mb: int = 100
    
    # Logging Configuration
    log_level: str = "INFO"
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
