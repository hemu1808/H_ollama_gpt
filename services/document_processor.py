import aiofiles
import tempfile
import os
from typing import List
from langchain_core.documents import Document
from PyMuPDFLoader import PyMuPDFLoader
from core import SecurityValidator, SemanticChunker, EmbeddingCache
from config import settings
import structlog

logger = structlog.get_logger()

class DocumentProcessor:
    """Production document processor with async I/O"""
    
    def __init__(self, embedding_cache: EmbeddingCache):
        self.chunker = SemanticChunker(embedding_cache)
        self.validator = SecurityValidator()
    
    async def process_upload(
        self,
        file_content: bytes,
        filename: str
    ) -> List[Document]:
        """Process uploaded PDF with security checks"""
        # Validate size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {len(file_content)} bytes")
        
        # Write to temp file
        async with aiofiles.tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.pdf',
            delete=False
        ) as temp_file:
            await temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # Validate content type
            if not await self.validator.validate_file_content(temp_path):
                raise ValueError("Invalid PDF content")
            
            # Extract text
            loader = PyMuPDFLoader(temp_path)
            docs = await asyncio.to_thread(loader.load)
            
            if not docs:
                raise ValueError("No content extracted from PDF")
            
            # Apply semantic chunking
            chunks = await self.chunker.chunk_semantically(docs)
            
            logger.info(
                "document_processed",
                filename=filename,
                chunks=len(chunks)
            )
            
            return chunks
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)