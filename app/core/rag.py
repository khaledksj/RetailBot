"""
RAG (Retrieval-Augmented Generation) pipeline for the Shop Manual Chatbot.
"""

from typing import List, Dict, Any, AsyncGenerator
import numpy as np
from datetime import datetime

from app.core.settings import get_settings
from app.core.logging import get_logger
from app.core.models import ChatResponse, Source, Chunk
from app.core.llm import EmbeddingService, ChatService
from app.core.db import get_vector_store

logger = get_logger(__name__)
settings = get_settings()

class RAGPipeline:
    """RAG pipeline for processing queries and generating responses."""
    
    # RAG prompt template (use verbatim as requested)
    RAG_PROMPT_TEMPLATE = """You are a help bot for a retail shop. Answer ONLY using the provided CONTEXT from the shop's PDF manuals.
- If the answer is not in the context, say: "I couldn't find this in the manuals."
- Be concise, actionable, and quote exact lines when helpful.
- Always return citations as: [filename.pdf – p.PAGE].
CONTEXT:
{context}
USER QUESTION:
{question}"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chat_service = ChatService()
        self.vector_store = get_vector_store()
    
    def _maximal_marginal_relevance(
        self,
        chunks_with_scores: List[tuple[Chunk, float]],
        query_embedding: List[float],
        top_n: int,
        lambda_param: float = 0.5
    ) -> List[tuple[Chunk, float]]:
        """Apply Maximal Marginal Relevance (MMR) to rerank retrieved chunks."""
        if len(chunks_with_scores) <= top_n:
            return chunks_with_scores
        
        query_emb = np.array(query_embedding)
        selected = []
        remaining = list(chunks_with_scores)
        
        # Select first chunk (highest similarity)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_n and remaining:
            best_score = float('-inf')
            best_idx = 0
            
            for i, (chunk, _) in enumerate(remaining):
                # Calculate similarity to query
                chunk_emb = np.array(chunk.embedding)
                query_sim = np.dot(query_emb, chunk_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
                )
                
                # Calculate maximum similarity to already selected chunks
                max_selected_sim = 0
                for selected_chunk, _ in selected:
                    selected_emb = np.array(selected_chunk.embedding)
                    sim = np.dot(chunk_emb, selected_emb) / (
                        np.linalg.norm(chunk_emb) * np.linalg.norm(selected_emb)
                    )
                    max_selected_sim = max(max_selected_sim, sim)
                
                # MMR score
                mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_selected_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _format_context(self, chunks: List[Chunk]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        
        for chunk in chunks:
            context_parts.append(
                f"[{chunk.filename} – p.{chunk.page}]\n{chunk.content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Chunk]) -> List[Source]:
        """Extract source information from chunks."""
        sources = []
        seen_sources = set()
        
        for chunk in chunks:
            source_key = (chunk.filename, chunk.page)
            if source_key not in seen_sources:
                sources.append(Source(
                    filename=chunk.filename,
                    page=chunk.page,
                    snippet=""  # Remove vector chunk content from sources display
                ))
                seen_sources.add(source_key)
        
        return sources
    
    async def process_query(
        self,
        query: str,
        session_id: str = "default",
        temperature: float = 0.2
    ) -> ChatResponse:
        """Process a user query through the RAG pipeline."""
        start_time = datetime.utcnow()
        
        logger.info(f"Processing RAG query", extra={
            "session_id": session_id,
            "query_length": len(query),
            "temperature": temperature
        })
        
        try:
            # Truncate query if too long
            if self.chat_service.count_tokens(query) > settings.max_input_tokens:
                query = self.chat_service.truncate_text(query, settings.max_input_tokens)
                logger.warning(f"Query truncated to fit token limit")
            
            # Generate query embedding
            query_embeddings = await self.embedding_service.create_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Retrieve similar chunks
            chunks_with_scores = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=settings.top_k
            )
            
            # Log similarity scores for debugging
            if chunks_with_scores:
                scores = [score for _, score in chunks_with_scores]
                logger.info(f"Similarity scores", extra={
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "avg_score": sum(scores) / len(scores),
                    "all_scores": scores[:3]  # Log first 3 scores
                })
            
            # Lower threshold for Arabic content - accept anything with score > 0.1 (very permissive)
            valid_chunks = [(chunk, score) for chunk, score in chunks_with_scores if score > 0.1]
            
            if not valid_chunks:
                logger.info(f"No relevant chunks found for query", extra={
                    "total_chunks_found": len(chunks_with_scores),
                    "max_score": max([score for _, score in chunks_with_scores]) if chunks_with_scores else 0
                })
                return ChatResponse(
                    answer="I couldn't find this in the manuals.",
                    sources=[],
                    session_id=session_id
                )
            
            chunks_with_scores = valid_chunks
            
            # Apply MMR reranking
            reranked_chunks = self._maximal_marginal_relevance(
                chunks_with_scores,
                query_embedding,
                settings.retrieval_top_n
            )
            
            chunks = [chunk for chunk, _ in reranked_chunks]
            
            logger.info(f"Retrieved and reranked chunks", extra={
                "initial_count": len(chunks_with_scores),
                "final_count": len(chunks),
                "chunk_ids": [chunk.chunk_id for chunk in chunks]
            })
            
            # Format context
            context = self._format_context(chunks)
            
            # Build messages for chat completion
            prompt = self.RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Generate response
            answer = await self.chat_service.generate_response(
                messages=messages,
                temperature=temperature
            )
            
            # Extract sources
            sources = self._extract_sources(chunks)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"RAG query processed successfully", extra={
                "session_id": session_id,
                "processing_time_seconds": processing_time,
                "sources_count": len(sources)
            })
            
            return ChatResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {str(e)}")
            raise
    
    async def stream_chat_response(
        self,
        query: str,
        session_id: str = "default",
        temperature: float = 0.2
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response through WebSocket."""
        try:
            # Send initial status
            yield {
                "type": "status",
                "message": "Processing query..."
            }
            
            # Truncate query if too long
            if self.chat_service.count_tokens(query) > settings.max_input_tokens:
                query = self.chat_service.truncate_text(query, settings.max_input_tokens)
            
            # Generate query embedding
            yield {
                "type": "status",
                "message": "Searching manuals..."
            }
            
            query_embeddings = await self.embedding_service.create_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Retrieve similar chunks
            chunks_with_scores = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=settings.top_k
            )
            
            # Log similarity scores for debugging (same as regular process_query)
            if chunks_with_scores:
                scores = [score for _, score in chunks_with_scores]
                logger.info(f"Streaming similarity scores found", extra={
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "avg_score": sum(scores) / len(scores),
                    "all_scores": scores[:5],  # Log first 5 scores
                    "total_chunks": len(chunks_with_scores)
                })
            else:
                logger.warning("No chunks returned from similarity search in streaming")
            
            # TEMPORARILY REMOVE THRESHOLD - accept all chunks to debug
            valid_chunks = chunks_with_scores  # Accept everything for now
            
            if not valid_chunks:
                logger.warning("Absolutely no chunks found in streaming query")
                yield {
                    "type": "final_response",
                    "answer": "I couldn't find this in the manuals.",
                    "sources": []
                }
                return
            
            logger.info(f"Proceeding with {len(valid_chunks)} chunks for Arabic query")
            
            # Apply MMR reranking
            reranked_chunks = self._maximal_marginal_relevance(
                chunks_with_scores,
                query_embedding,
                settings.retrieval_top_n
            )
            
            chunks = [chunk for chunk, _ in reranked_chunks]
            
            # Send sources first
            sources = self._extract_sources(chunks)
            yield {
                "type": "sources",
                "sources": [source.dict() for source in sources]
            }
            
            # Format context and build prompt
            context = self._format_context(chunks)
            prompt = self.RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            yield {
                "type": "status",
                "message": "Generating answer..."
            }
            
            full_answer = ""
            async for chunk in self.chat_service.stream_response(
                messages=messages,
                temperature=temperature
            ):
                full_answer += chunk
                yield {
                    "type": "token",
                    "content": chunk
                }
            
            # Send final response
            yield {
                "type": "final_response",
                "answer": full_answer,
                "sources": [source.dict() for source in sources]
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield {
                "type": "error",
                "message": f"An error occurred: {str(e)}"
            }
