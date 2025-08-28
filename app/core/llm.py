"""
LLM and embedding service interface for the Shop Manual Chatbot RAG system.
"""

from typing import List, AsyncGenerator, Dict, Any
import asyncio
from openai import AsyncOpenAI
import tiktoken
import os

from app.core.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

class EmbeddingService:
    """Service for generating text embeddings using OpenAI."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings", extra={
                "model": self.model,
                "text_count": len(texts),
                "total_chars": sum(len(text) for text in texts)
            })
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=1536  # Specify 1536 dimensions to match database schema
            )
            
            embeddings = [data.embedding for data in response.data]
            
            logger.info(f"Embeddings generated successfully", extra={
                "embedding_count": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0
            })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

class ChatService:
    """Service for chat completion using OpenAI."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.chat_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> str:
        """Generate a chat response."""
        try:
            # Ensure temperature is within bounds
            temperature = max(0.0, min(settings.max_temperature, temperature))
            
            logger.info(f"Generating chat response", extra={
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message_count": len(messages)
            })
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            logger.info(f"Chat response generated", extra={
                "response_length": len(content) if content else 0,
                "finish_reason": response.choices[0].finish_reason
            })
            
            return content or ""
            
        except Exception as e:
            logger.error(f"Failed to generate chat response: {str(e)}")
            raise
    
    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming chat response."""
        try:
            # Ensure temperature is within bounds
            temperature = max(0.0, min(settings.max_temperature, temperature))
            
            logger.info(f"Starting streaming response", extra={
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Failed to stream chat response: {str(e)}")
            raise
