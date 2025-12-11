from celery import Celery
from celery.signals import worker_ready
from config import settings
import asyncio
from services import DocumentProcessor
from core import EmbeddingCache

# Initialize Celery
celery_app = Celery(
    'rag_worker',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# Initialize components
embedding_cache = EmbeddingCache(settings.REDIS_URL)
doc_processor = DocumentProcessor(embedding_cache)

@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, file_content: bytes, filename: str):
    """Process document in background"""
    try:
        # Run async code in sync context
        chunks = asyncio.run(doc_processor.process_upload(file_content, filename))
        
        # Format for storage
        return {
            'status': 'success',
            'chunks': len(chunks),
            'documents': [doc.page_content for doc in chunks],
            'metadatas': [doc.metadata for doc in chunks]
        }
    except Exception as e:
        logger.error("task_failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60)