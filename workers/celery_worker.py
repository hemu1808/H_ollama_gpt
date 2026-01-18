import os
import asyncio
import logging
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
        
        # Celery passes data as strings/JSON. We might need to encode back to bytes 
        # if your frontend sent base64, or use as is if it's raw text.
        # Assuming simple string content for this example:
        file_bytes = file_content_str.encode('utf-8')
        
        # Run Async processor in Sync Celery worker
        loop = asyncio.get_event_loop()
        if loop.is_running():
            docs = loop.run_until_complete(doc_processor.process_upload(file_bytes, filename))
        else:
            docs = asyncio.run(doc_processor.process_upload(file_bytes, filename))
            
        return {
            "status": "success", 
            "chunks": len(docs), 
            "message": f"Processed {filename}"
        }
        
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise self.retry(exc=e, countdown=10)