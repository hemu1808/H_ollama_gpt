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

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False

# Try importing standard PDF library
try:
    import pypdf
except ImportError:
    pypdf = None

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # 1. Initialize Vector DB
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)
        
        # 2. Initialize Keyword DB
        self.bm25_retriever = PersistedBM25Retriever()
        
        # 3. Initialize Cache & Chunker
        self.embedding_cache = EmbeddingCache(redis_url=settings.REDIS_URL)
        self.chunker = SemanticChunker(embedding_cache=self.embedding_cache)
        
        # 4. Load Embedding Model (Critical for Semantic Chunking)
        self.embed_model = None
        if HAS_SENTENCE_TRANSFORMER:
            logger.info("Loading SentenceTransformer for Semantic Chunking...")
            # Use a fast, small model for chunking decisions
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # --- THE FIX: Inject real embedding function into Chunker ---
            async def real_embed_fn(text: str) -> List[float]:
                # Offload to thread to prevent blocking async loop
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.embed_model.encode(text).tolist()
                )
            
            # Patch the chunker's method dynamically
            self.chunker._dynamic_embed_fn = real_embed_fn
        else:
            logger.warning("sentence-transformers not installed. Semantic chunking will be degraded.")

    async def process_upload_stream(self, file_content: bytes, filename: str):
        """
        Generator that yields status updates while processing
        """
        logger.info(f"Processing file: {filename}")
        yield "extract"
        
        # Offload CPU-heavy PDF parsing
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._extract_text_from_pdf_bytes, file_content)
        
        if not text:
            raise ValueError("Could not extract text from PDF.")

        yield "clean"
        await asyncio.sleep(0.1)
        cleaned_text = self._clean_text(text)
        
        yield "chunk"
        # Create a LangChain Document
        input_doc = Document(page_content=cleaned_text, metadata={"source": filename})
        
        # This now uses the REAL embedding function injected in __init__
        if settings.ENABLE_SEMANTIC_CHUNKING and HAS_SENTENCE_TRANSFORMER:
            lc_chunks = await self.chunker.chunk_semantically([input_doc])
        else:
            # Fallback to standard splitter if model missing
            lc_chunks = self.chunker.base_splitter.split_documents([input_doc])
            
        chunks = [doc.page_content for doc in lc_chunks]
        # Ensure metadata contains source
        metadatas = [{"source": filename, **doc.metadata} for doc in lc_chunks]
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        
        if not chunks:
            raise ValueError("No text chunks generated.")
        
        yield "index"
        # 1. Upsert to Chroma
        self.collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        # 2. Update BM25 (Atomic Lock handled inside class)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.bm25_retriever.index_documents, chunks, ids)
        
        logger.info(f"Successfully saved {len(chunks)} chunks from {filename}")
        yield "done"

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