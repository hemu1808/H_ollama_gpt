import time
import logging
import json
import asyncio
import chromadb
from typing import List
import os

# --- IMPORTS ---
from config import settings
from core.security import SecurityValidator
from dspy_module import RAGModule 
from core.retrievers import PersistedBM25Retriever, reciprocal_rank_fusion
from schemas import QueryInput, AnswerResponse  

# Lazy Load CrossEncoder (Heavy Model)
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        logger.info("Initializing RAG Service...")
        self.security = SecurityValidator()
        self.rag_module = RAGModule() # Loads DSPy optimized logic
        
        # 1. Load Keyword Search (BM25)
        self.bm25 = PersistedBM25Retriever()
        
        # 2. Load Vector Search (Chroma)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)

        # 3. Load Re-ranker (The "Deep Think" judge)
        self.cross_encoder = None
        if HAS_CROSS_ENCODER:
            logger.info("Loading Cross-Encoder for precision re-ranking...")
            # 'ms-marco-MiniLM-L-6-v2' is fast and effective for re-ranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            logger.warning("CrossEncoder not found. Install sentence-transformers for better accuracy.")

    async def list_documents(self) -> List[str]:
        """
        Fetch unique filenames from the database for the UI.
        """
        try:
            # Efficiently get just the metadatas
            result = self.collection.get(include=['metadatas'])
            if not result['metadatas']:
                return []
            
            sources = set()
            for meta in result['metadatas']:
                if 'source' in meta:
                    sources.add(meta['source'])
            return list(sources)
        except Exception as e:
            logger.error(f"Error listing docs: {e}")
            return []
        
    async def delete_document(self, filename: str) -> bool:
        try:
            logger.info(f"Attempting to delete {filename}...")

            # --- STEP 1: Delete from Vector Database (The "Memory") ---
            # If you are using ChromaDB:
            try:
                # This deletes all chunks associated with this source file
                self.collection.delete(where={"source": filename})
                logger.info(f"Removed {filename} from Vector DB")
            except Exception as e:
                logger.error(f"Vector DB deletion failed: {e}")
                # We continue anyway to try and delete the physical file

            # --- STEP 2: Delete Physical File (The "Source") ---
            # Adjust this path if your files are stored elsewhere (e.g., in a 'data' folder)
            file_path = f"./data/{filename}" 
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted physical file: {file_path}")
            else:
                # Check if it's in the root directory
                if os.path.exists(filename):
                    os.remove(filename)
                    logger.info(f"Deleted physical file from root: {filename}")

            # --- STEP 3: Clear Internal Cache (If applicable) ---
            # If your 'list_documents' uses a cached list, clear it here.
            # self.document_cache = [] 

            return True

        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False

    async def answer_question(self, input_data: QueryInput) -> AnswerResponse:
        """
        Non-streaming wrapper for the streaming logic.
        """
        response_data = {}
        async for chunk in self.answer_question_stream(input_data):
            if chunk.startswith("data: "):
                data = json.loads(chunk.replace("data: ", ""))
                if data.get("type") == "result":
                    response_data = data
        
        return AnswerResponse(
            answer=response_data.get("answer", "Processing failed."),
            sources=response_data.get("sources", []),
            metadata=response_data.get("metadata", {}),
            processing_time=response_data.get("processing_time", 0.0),
            thoughts=response_data.get("thoughts")
        )

    async def answer_question_stream(self, input_data: QueryInput):
        """
        The Core Logic: Hybrid Search -> RRF -> Rerank -> DSPy Generate -> Guardrail
        """
        start_time = time.time()
        
        # Yield initial status
        yield f"data: {json.dumps({'type': 'status', 'content': 'Searching Knowledge Base...'})}\n\n"
        await asyncio.sleep(0.01) # Yield to event loop

        try:
            # 1. Security Checkraw_query = await self.security.sanitize_query(input_data.question)
            raw_query = await self.security.sanitize_query(input_data.question)
            # --- STEP 1: CONTEXTUALIZE (REWRITE) ---
            search_query = raw_query
            
            # Only rewrite if we actually have history
            if input_data.chat_history:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Connecting Memory...'})}\n\n"
                
                # Format last 3 turns for context (prevent token overflow)
                recent_history = input_data.chat_history[-3:]
                history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])
                
                # DSPy Rewrite
                search_query = self.rag_module.rewrite_query(raw_query, history_str)
                
                # Log the logic for debugging/UI
                logger.info(f"Rewrote '{raw_query}' to '{search_query}'")
                yield f"data: {json.dumps({'type': 'status', 'content': f'Searching: {search_query}'})}\n\n"
            
            # 2. Hybrid Retrieval (Fetch 3x candidates to allow effective filtering)
            search_k = input_data.top_k * 20
            
            # A. Vector Search
            results = self.collection.query(query_texts=[search_query], n_results=search_k)
            vector_cands = []
            if results['ids']:
                for id, doc in zip(results['ids'][0], results['documents'][0]):
                    vector_cands.append((id, 1.0)) # Score 1.0 is placeholder; RRF handles ranking

            # B. Keyword Search (BM25)
            # This uses the file-locking retriever we fixed earlier
            keyword_cands = self.bm25.retrieve(search_query, top_k=search_k)
            
            # C. Fusion (Reciprocal Rank Fusion)
            # This mathematically combines the two lists so neither dominates
            fused = reciprocal_rank_fusion(vector_cands, keyword_cands)
            top_ids = [k[0] for k in fused[:input_data.top_k * 3]] # Keep top 2x for re-ranking
            
            # Fetch content for the winning IDs
            final_docs = []
            if top_ids:
                # Batch fetch is faster than one-by-one
                fetch_res = self.collection.get(ids=top_ids)
                # Map IDs to Documents
                doc_map = {id: doc for id, doc in zip(fetch_res['ids'], fetch_res['documents'])}
                # Preserve RRF order
                final_docs = [doc_map[id] for id in top_ids if id in doc_map]

            yield f"data: {json.dumps({'type': 'status', 'content': f'Found {len(final_docs)} candidates...'})}\n\n"

            # 3. Re-Ranking (The Quality Filter)
            if self.cross_encoder and final_docs:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Re-ranking results...'})}\n\n"
                pairs = [[search_query, doc] for doc in final_docs]
                scores = self.cross_encoder.predict(pairs)
                scored_docs = sorted(zip(final_docs, scores), key=lambda x: x[1], reverse=True)
                filtered_docs = [doc for doc, score in scored_docs if score > -10.0]
                if not filtered_docs and final_docs:
                    logger.warning("Re-ranker filtered all docs! Falling back to raw results.")
                    final_docs = final_docs[:3]
                else:
                    final_docs = filtered_docs[:input_data.top_k]
            else:
                # No re-ranker, just take top K
                final_docs = final_docs[:input_data.top_k]

            # 4. DSPy Generation
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating Answer...'})}\n\n"
            
            answer = ""
            thoughts = None
            
            if not final_docs:
                answer = "I could not find any relevant information in the uploaded documents."
                sources = []
            else:
                context = "\n---\n".join(final_docs)
                
                # Run the DSPy Module
                # if mode="deep", this triggers ChainOfThought
                prediction = self.rag_module(
                    question=search_query, 
                    context=context, 
                    mode=input_data.mode
                )
                
                answer = prediction.answer
                
                # Capture thoughts if in Deep mode
                if input_data.mode == "deep" and hasattr(prediction, 'rationale'):
                    thoughts = prediction.rationale

                # 5. Anti-Hallucination / Honesty Check
                # If the model confidently says "I don't know" in the answer, trust it.
                lower_ans = answer.lower()
                if "cannot find" in lower_ans or "context does not contain" in lower_ans:
                    # We leave the answer as is, but maybe flag it in metadata
                    pass

            sources = [doc[:100] + "..." for doc in final_docs]

            # 6. Final Response Payload
            payload = {
                'type': 'result',
                'answer': answer,
                'thoughts': thoughts,
                'sources': sources,
                'metadata': {'mode': input_data.mode},
                'processing_time': time.time() - start_time
            }
            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            logger.exception("Error in RAG stream")
            yield f"data: {json.dumps({'type': 'error', 'answer': 'System Error', 'thoughts': str(e)})}\n\n"