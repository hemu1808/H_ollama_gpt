import os
import asyncio
import logging
import tempfile
from celery import Celery
from config import settings
from document_processor import DocumentProcessor

# Setup Logging
logger = logging.getLogger(__name__)

# Initialize Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    'rag_worker',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

# --- FIX: Initialize without arguments ---
# The DocumentProcessor now initializes its own cache/retrievers internally
doc_processor = DocumentProcessor()

@celery_app.task(bind=True, name="process_document_task", max_retries=3)
def process_document_task(self, file_content_str: str, filename: str):
    """
    Background task to process uploaded documents.
    """
    try:
        logger.info(f"Worker processing file: {filename}")
        file_bytes = file_content_str.encode('utf-8')
        
        # --- FIX: Write to temp file because PDF parser needs a file path ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        # --- FIX: Consume the async generator correctly ---
        async def run_processor():
            async for status in doc_processor.process_upload_stream(temp_path, filename):
                pass # The generator yields status strings, we just consume them
            return True

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(run_processor())
        else:
            asyncio.run(run_processor())
            
        os.remove(temp_path) # Cleanup
        
        return {
            "status": "success", 
            "message": f"Processed {filename}"
        }
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Task failed: {e}")
        raise self.retry(exc=e, countdown=10)

@celery_app.task(bind=True, name="extract_graph_entities_task", max_retries=3)
def extract_graph_entities_task(self, chunks: list[str], filename: str):
    """
    Background task to extract and merge Graph RAG entities to avoid event loop blocking.
    """
    try:
        from dspy_module import RAGModule
        from neo4j import GraphDatabase
        import asyncio
        
        rag_mod = RAGModule()
        uri = getattr(settings, "NEO4J_URI", None) or "bolt://localhost:7687"
        user = getattr(settings, "NEO4J_USER", None) or "neo4j"
        password = getattr(settings, "NEO4J_PASSWORD", None) or "password"
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        async def run_extraction():
            with driver.session() as session:
                for chunk in chunks:
                    try:
                        loop = asyncio.get_running_loop()
                        pred = await loop.run_in_executor(None, rag_mod.entity_extractor, chunk)
                        relations = pred.relationships.split("\n")
                        for rel in relations:
                            parts = [p.strip() for p in rel.split("|")]
                            if len(parts) == 3:
                                src, edge, tgt = parts
                                query = (
                                    "MERGE (a:Entity {id: $src}) "
                                    "MERGE (b:Entity {id: $tgt}) "
                                    "MERGE (a)-[r:RELATES_TO {type: $edge}]->(b)"
                                )
                                await loop.run_in_executor(None, lambda: session.run(query, src=src, tgt=tgt, edge=edge))
                    except Exception as inner_e:
                        logger.warning(f"Failed to extract entities for chunk: {inner_e}")
                        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(run_extraction())
        else:
            asyncio.run(run_extraction())
            
        driver.close()
        logger.info(f"Graph extraction complete for {filename}.")
        return {"status": "success", "message": f"Graph extract done for {filename}"}
        
    except Exception as e:
        logger.error(f"Graph extraction error: {e}")
        raise self.retry(exc=e, countdown=10)