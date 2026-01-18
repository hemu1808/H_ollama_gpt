import json
import asyncio
import shutil
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Imports
from config import settings
from rag_service import RAGService
from schemas import QueryInput 
from document_processor import DocumentProcessor
# import dspy_module # Keeping your import

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppState:
    rag: RAGService = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG System...")
    try:
        state.rag = RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
async def query_endpoint(input_data: QueryInput):
    if not state.rag:
        raise HTTPException(status_code=503, detail="System not ready")
    return await state.rag.answer_question(input_data)

@app.post("/query/stream")
async def query_stream_endpoint(input_data: QueryInput):
    if not state.rag:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return StreamingResponse(
        state.rag.answer_question_stream(input_data), 
        media_type="text/event-stream"
    )

@app.get("/documents")
async def get_documents():
    if not state.rag: return []
    return await state.rag.list_documents()

# --- MODIFIED UPLOAD ENDPOINT FOR STREAMING STATUS ---
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Read file into memory immediately so we don't block the stream logic
    try:
        content = await file.read()
        filename = file.filename
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # 2. Define the generator that yields status updates
    async def event_generator():
        temp_path = f"temp_{filename}"
        try:
            processor = DocumentProcessor()
            
            # This calls the NEW stream method in your DocumentProcessor
            # Ensure your DocumentProcessor has 'process_upload_stream' implemented as shown previously
            async for step in processor.process_upload_stream(content, filename):
                # Send Server-Sent Event (SSE) format
                # Format: data: {"step": "clean"}\n\n
                yield f"data: {json.dumps({'step': step})}\n\n"
                
        except Exception as e:
            logger.error(f"Upload stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    # 3. Return the StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/documents/delete_file/{filename}")
async def delete_document_endpoint(filename: str):
    if not state.rag:
        raise HTTPException(status_code=503, detail="System not ready")
    
    try:
        # This assumes your RAGService has a 'delete_document' method.
        # If it doesn't, see the "Step 2" below.
        success = await state.rag.delete_document(filename)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found inside RAG system")
            
        return {"status": "success", "message": f"{filename} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)