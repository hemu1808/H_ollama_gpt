import os
import re
import logging
import asyncio
import chromadb
from typing import List
from langchain_core.documents import Document

# Internal modules
from config import settings
from core.retrievers import PersistedBM25Retriever
from core.chunkers import SemanticChunker # <--- CONNECTED NOW
from core.caching import EmbeddingCache   # <--- CONNECTED NOW

# Try importing standard PDF library
try:
    import pypdf
except ImportError:
    pypdf = None

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)
        self.bm25_retriever = PersistedBM25Retriever()
        
        # Initialize Cache and Chunker
        # Note: We assume Redis is running for the cache
        self.embedding_cache = EmbeddingCache(redis_url=settings.REDIS_URL)
        self.chunker = SemanticChunker(embedding_cache=self.embedding_cache)

    async def get_embedding_mock(self, text: str) -> List[float]:
        """
        In production, this calls Ollama/OpenAI. 
        For now, we need a placeholder or real call to make SemanticChunker work.
        """
        # TODO: Replace with actual Ollama embedding call
        # returning random vector for structure testing if Ollama fails
        return [0.1] * 384 

    async def process_upload_stream(self, file_content: bytes, filename: str) -> List[str]:
        logger.info(f"Processing file: {filename}")
        yield "extract"
        await asyncio.sleep(0.1)  
        if len(file_content) > getattr(settings, 'MAX_FILE_SIZE', 50 * 1024 * 1024):
            raise ValueError("File too large")
            
        text = self._extract_text_from_pdf_bytes(file_content)
        if not text:
            raise ValueError("Could not extract text.")

        yield "clean"
        await asyncio.sleep(0.3)    
        cleaned_text = self._clean_text(text)
        
        yield "chunk"
        # --- NEW CHUNKING LOGIC ---
        # Convert to LangChain Document format for the chunker
        input_doc = Document(page_content=cleaned_text, metadata={"source": filename})
        
        # Use the Semantic Chunker (this is async)
        # Note: We need to inject the real embedding function into the chunker logic
        # For this quick fix, we are patching the get_embedding method dynamically
        self.chunker.get_embedding = self.get_embedding_mock 
        
        if settings.ENABLE_SEMANTIC_CHUNKING:
            lc_chunks = await self.chunker.chunk_semantically([input_doc])
        else:
            # Fallback to simple splitting if semantic is disabled
            lc_chunks = [input_doc] # Add simple splitter here if needed
            
        # Extract text back from LangChain docs
        chunks = [doc.page_content for doc in lc_chunks]
        metadatas = [{"source": filename, **doc.metadata} for doc in lc_chunks]
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        
        if not chunks:
            raise ValueError("No text chunks generated.")
        
        yield "embed"
        await asyncio.sleep(0.5) 
        # (Embedding usually happens implicitly in Chroma or here if explicit)

        # 5. Indexing (Storage)
        yield "index"
        self.collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        # 2. Update BM25
        self.bm25_retriever.index_documents(chunks, ids)
        
        logger.info(f"Successfully saved {len(chunks)} chunks from {filename}")
        yield "done"
        return

    def _extract_text_from_pdf_bytes(self, file_content: bytes) -> str:
        import io
        if not pypdf: return ""
        try:
            pdf_file = io.BytesIO(file_content)
            reader = pypdf.PdfReader(pdf_file)
            text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()