from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
import asyncio
from config import settings
import numpy as np
import re

# --- FIXED IMPORT ---
# We use the full package path to avoid import errors
from .caching import EmbeddingCache
from .retrievers import PersistedBM25Retriever 

class SemanticChunker:
    """Intelligent chunking based on semantic similarity"""
    
    # Fixed type hint from 'None' to 'EmbeddingCache'
    def __init__(self, embedding_cache: Optional[EmbeddingCache] = None):
        self.embedding_cache = embedding_cache
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    async def sentencize(self, documents: List[Document]) -> List[str]:
        """Split into sentences while preserving metadata"""
        sentences = []
        for doc in documents:
            # Simple sentence tokenizer
            text = doc.page_content
            parts = re.split(r'[.!?]+\s+', text)
            for part in parts:
                if part.strip():
                    sentences.append({
                        'text': part.strip(),
                        'metadata': doc.metadata
                    })
        return sentences
    
    async def find_breakpoints(self, embeddings: List[List[float]], threshold: float = 0.7) -> List[int]:
        """Find indices where semantic similarity drops"""
        if not embeddings: return []
        breakpoints = [0]
        
        for i in range(1, len(embeddings)):
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i] * 1e-10)
            )
            if similarity < threshold:
                breakpoints.append(i)
        
        return breakpoints
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Placeholder for actual embedding logic. 
        In production, this connects to Ollama or OpenAI.
        """
        # Checks if a dynamic override exists (patched in document_processor)
        if hasattr(self, '_dynamic_embed_fn'):
            return await self._dynamic_embed_fn(text)
        return [0.0] * 384 # Fallback mock

    async def chunk_semantically(self, documents: List[Document]) -> List[Document]:
        """Create semantically coherent chunks"""
        if not self.embedding_cache:
            # Fallback if cache/embeddings aren't configured
            return self.base_splitter.split_documents(documents)

        sentences = await self.sentencize(documents)
        
        # Embed each sentence
        embeddings = []
        for sentence in sentences:
            # Use a lambda to defer the embedding call
            emb = await self.embedding_cache.get_or_create(
                sentence['text'],
                self.get_embedding
            )
            embeddings.append(emb)
        
        # Find breakpoints
        breakpoints = await self.find_breakpoints(
            embeddings,
            getattr(settings, 'SEMANTIC_CHUNK_THRESHOLD', 0.7)
        )
        
        # Merge sentences into chunks
        chunks = []
        for i in range(len(breakpoints)):
            start = breakpoints[i]
            end = breakpoints[i+1] if i+1 < len(breakpoints) else len(sentences)
            
            chunk_text = ". ".join([sentences[j]['text'] for j in range(start, end)])
            
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **sentences[start]['metadata'],
                    'chunk_type': 'semantic',
                    'sentence_range': f"{start}-{end}"
                }
            ))
        
        return chunks

class ParentChildChunker:
    """Enhanced parent-child with semantic boundaries"""
    
    def __init__(self, embedding_cache: EmbeddingCache):
        self.semantic_chunker = SemanticChunker(embedding_cache)
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    async def chunk_with_parent_child(self, documents: List[Document]) -> List[Document]:
        """Create parent-child chunks with semantic boundaries"""
        all_chunks = []
        
        for doc in documents:
            # Create parent chunks using semantic boundaries
            parent_chunks = await self.semantic_chunker.chunk_semantically([doc])
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                parent_id = f"{doc.metadata.get('source', 'doc')}_p{parent_idx}"
                
                # Parent metadata
                parent_chunk.metadata.update({
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                })
                all_chunks.append(parent_chunk)
                
                # Child chunks (more granular)
                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_SIZE // 2,
                    chunk_overlap=settings.CHUNK_OVERLAP // 2,
                )
                child_chunks = child_splitter.split_documents([parent_chunk])
                
                for child_idx, child_chunk in enumerate(child_chunks):
                    child_chunk.metadata.update({
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "child_id": f"{parent_id}_c{child_idx}",
                    })
                    all_chunks.append(child_chunk)
        
        return all_chunks