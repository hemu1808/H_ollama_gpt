from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import asyncio

from config import settings, setup_observability
from services import RAGService
from api.middleware import RateLimiter, validate_file_upload
from workers import celery_app

logger = setup_observability()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.rag = RAGService()
    await app.state.rag.initialize()
    logger.info("rag_service_initialized")
    yield
    # Shutdown
    await app.state.rag.embedding_cache.close()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limiter = RateLimiter(settings.REDIS_URL, settings.RATE_LIMIT_PER_MINUTE)

security = HTTPBearer()

@app.post("/query")
async def query_rag(
    query: QueryInput,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limiter)
):
    """Main RAG query endpoint"""
    try:
        return await app.state.rag.answer_question(query)
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal error")

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rate_limit: None = Depends(rate_limiter)
):
    """Upload and process document"""
    try:
        # Validate file
        await validate_file_upload(file)
        
        # Read content
        content = await file.read()
        
        # Queue for background processing
        task = process_document_task.delay(
            file_content=content,
            filename=file.filename
        )
        
        logger.info("document_uploaded", filename=file.filename, task_id=task.id)
        
        return {"task_id": task.id, "status": "queued"}
        
    except Exception as e:
        logger.error("upload_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Check async task status"""
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": result.state,
        "result": result.result if result.ready() else None
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health = {
        "api": "healthy",
        "chroma": await check_chroma(),
        "redis": await check_redis(),
        "ollama": await check_ollama()
    }
    
    if all(v == "healthy" for v in health.values()):
        return health
    else:
        raise HTTPException(status_code=503, detail=health)

async def check_chroma():
    try:
        await app.state.rag.chroma_client.heartbeat()
        return "healthy"
    except:
        return "unhealthy"

async def check_redis():
    try:
        await app.state.rag.embedding_cache.cache.ping()
        return "healthy"
    except:
        return "unhealthy"

async def check_ollama():
    try:
        from ollama import AsyncClient
        client = AsyncClient(host=settings.OLLAMA_URL)
        await client.list()
        return "healthy"
    except:
        return "unhealthy"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS,
        log_level="info"
    )