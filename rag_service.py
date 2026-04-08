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
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException 

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
        from services.quantized_chroma import QuantizedChromaAdapter, OllamaEmbeddingFunction
        ef = OllamaEmbeddingFunction(
            model_name=getattr(settings, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=getattr(settings, "OLLAMA_URL", "http://localhost:11434")
        )
        raw_collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )
        raw_collection._embedding_function = ef
        self.collection = QuantizedChromaAdapter(raw_collection, dim=768)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # 3. Load Re-ranker (The "Deep Think" judge)
        self.cross_encoder = None
        if HAS_CROSS_ENCODER:
            logger.info("Loading Cross-Encoder for precision re-ranking...")
            try:
                # 'ms-marco-MiniLM-L-6-v2' is fast and effective for re-ranking
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}")
                self.cross_encoder = None
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
            logger.info(f"Deleting {filename}...")
            
            # --- 1. Vector DB Deletion ---
            # Query to get IDs first to confirm existence
            results = self.collection.get(where={"source": filename})
            ids_to_delete = results['ids']
            
            if ids_to_delete:
                self.collection.delete(where={"source": filename})
                logger.info(f"Removed {len(ids_to_delete)} chunks from Chroma.")
            else:
                logger.warning(f"No chunks found in Chroma for {filename}")

            # --- 2. BM25 Deletion ---
            # Note: BM25 is usually append-only for speed. 
            # Complete rebuild is safest for deletion, but slow.
            # For now, we accept it might stay in BM25 until next rebuild.
            # Or you can trigger a rebuild here if the corpus is small.
            
            # --- 3. Physical File Deletion ---
            # If you save files to a folder, delete them here.
            # Assuming files are temporary or managed externally, 
            # but if you have a './data' folder:
            file_path = os.path.join("data", filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
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
        if self.cross_encoder:
            logger.info("Re-ranker is active and filtering results.")
        
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
            
            # 1.5. Router (Adaptive Mode)
            actual_mode = input_data.mode
            if actual_mode == "adaptive":
                yield f"data: {json.dumps({'type': 'status', 'content': 'Routing Query...'})}\n\n"
                
                # DSPy router
                actual_mode = await loop.run_in_executor(
                    None,
                    self.rag_module.route_query,
                    search_query
                )
                yield f"data: {json.dumps({'type': 'status', 'content': f'Route chosen: {actual_mode.upper()}'})}\n\n"
            
            # Only rewrite if we actually have history
            if input_data.chat_history:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Connecting Memory...'})}\n\n"
                
                # Format last 3 turns for context (prevent token overflow)
                recent_history = input_data.chat_history[-3:]
                history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])
                
                # DSPy Rewrite
                search_query = await loop.run_in_executor(
                    None, 
                    self.rag_module.rewrite_query, 
                    raw_query, 
                    history_str
                )
                
                # Log the logic for debugging/UI
                logger.info(f"Rewrote '{raw_query}' to '{search_query}'")
                yield f"data: {json.dumps({'type': 'status', 'content': f'Searching: {search_query}'})}\n\n"
            
            # PHASE 4: Human-in-the-Loop Simulation
            if actual_mode == "agentic":
                # In a real bi-directional WebSocket, we would pause here.
                # For SSE, we yield an action_required event (which the UI can intercept to ask for permission).
                yield f"data: {json.dumps({'type': 'action_required', 'content': f'Agentic Mode engaged. Preparing to use external tools for: {search_query}'})}\n\n"
                await asyncio.sleep(1.0) # Simulate a brief pause
            
            # 2. Hybrid Retrieval (Fetch 3x candidates to allow effective filtering)
            search_k = input_data.top_k * 20
            
            loop = asyncio.get_running_loop()
            
            # --- PHASE 8: GRAPH RAG RETRIEVAL ---
            graph_docs = []
            if actual_mode == "graph" and getattr(settings, "ENABLE_GRAPH_RAG", False):
                yield f"data: {json.dumps({'type': 'status', 'content': 'Querying Knowledge Graph...'})}\n\n"
                try:
                    from neo4j import GraphDatabase
                    uri = getattr(settings, "NEO4J_URI", None) or "bolt://localhost:7687"
                    user = getattr(settings, "NEO4J_USER", None) or "neo4j"
                    password = getattr(settings, "NEO4J_PASSWORD", None) or "password"
                    driver = GraphDatabase.driver(uri, auth=(user, password))
                    
                    def run_cypher():
                        with driver.session() as session:
                            keywords = [kw.lower() for kw in search_query.split() if len(kw) > 3]
                            if keywords:
                                anchor = keywords[0]
                                res = session.run(
                                    "MATCH p=(n:Entity)-[r:RELATES_TO*1..2]-(m) "
                                    "WHERE toLower(n.id) CONTAINS $anchor "
                                    "RETURN n.id as src, type(r[0]) as edge, m.id as tgt LIMIT 50",
                                    anchor=anchor
                                )
                            else:
                                res = session.run("MATCH p=(n:Entity)-[r:RELATES_TO*1..2]-(m) RETURN n.id as src, type(r[0]) as edge, m.id as tgt LIMIT 50")
                            return [f"{rec['src']} {rec['edge']} {rec['tgt']}" for rec in res]
                            
                    graph_docs = await loop.run_in_executor(None, run_cypher)
                    driver.close()
                    if graph_docs:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'Found {len(graph_docs)} graph relationships.'})}\n\n"
                except ImportError:
                    logger.warning("neo4j driver not installed.")
                except Exception as e:
                    logger.warning(f"Graph retrieval failed: {e}")
            # ------------------------------------

            # A. Vector Search
            results = await loop.run_in_executor(None, lambda: self.collection.query(query_texts=[search_query], n_results=search_k))
            vector_cands = []
            if results['ids']:
                for id, doc in zip(results['ids'][0], results['documents'][0]):
                    vector_cands.append((id, 1.0)) # Score 1.0 is placeholder; RRF handles ranking

            # B. Keyword Search (BM25)
            # This uses the file-locking retriever we fixed earlier
            keyword_cands = await loop.run_in_executor(
                None, 
                lambda: self.bm25.retrieve(search_query, top_k=search_k)
            )
            
            # C. Fusion (Reciprocal Rank Fusion)
            # This mathematically combines the two lists so neither dominates
            fused = reciprocal_rank_fusion(vector_cands, keyword_cands)
            top_ids = [k[0] for k in fused[:input_data.top_k * 3]] # Keep top 2x for re-ranking
            
            # Fetch content for the winning IDs
            final_docs = []
            if top_ids:
                # Batch fetch is faster than one-by-one
                fetch_res = await loop.run_in_executor(None, lambda: self.collection.get(ids=top_ids))
                # Map IDs to Documents
                doc_map = {id: doc for id, doc in zip(fetch_res['ids'], fetch_res['documents'])}
                # Preserve RRF order
                final_docs = [doc_map[id] for id in top_ids if id in doc_map]

            yield f"data: {json.dumps({'type': 'status', 'content': f'Found {len(final_docs)} candidates...'})}\n\n"

            # PHASE 7: CORRECTIVE RAG (CRAG)
            yield f"data: {json.dumps({'type': 'status', 'content': 'Evaluating context relevance...'})}\n\n"
            relevant_docs = await loop.run_in_executor(
                None, 
                self.rag_module.evaluate_context, 
                search_query, 
                final_docs
            )
            
            if not relevant_docs and final_docs:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Context irrelevant. Querying Web...'})}\n\n"
                try:
                    import wikipedia
                    web_summary = await loop.run_in_executor(
                        None, 
                        lambda: wikipedia.summary(search_query, sentences=4)
                    )
                    relevant_docs = [f"WEB SEARCH RESULT: {web_summary}"]
                except Exception as e:
                    logger.warning(f"Web search fallback failed: {e}")
                    relevant_docs = final_docs # fallback to original docs if web fails
            
            final_docs = relevant_docs
            # Combine graph docs at the top so they aren't lost if vector search yields nothing
            if graph_docs:
                final_docs = graph_docs + final_docs

            # 3. Re-Ranking (The Quality Filter)
            if self.cross_encoder and final_docs:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Re-ranking results...'})}\n\n"
                pairs = [[search_query, doc] for doc in final_docs]
                scores = await loop.run_in_executor(None, lambda: self.cross_encoder.predict(pairs))
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
                
            if final_docs:
                yield f"data: {json.dumps({'type': 'status', 'content': f'Retained {len(final_docs)} highly relevant chunk(s).'})}\n\n"

            # 4. DSPy Generation
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating Answer...'})}\n\n"
            
            answer = "I could not find any relevant information in the uploaded documents."
            thoughts = None
            sources = []

            if final_docs:
                try:
                    @self.circuit_breaker
                    def safe_generate():
                        return self.rag_module.forward(
                            question=search_query, 
                            context="\n---\n".join(final_docs), 
                            history_str=history_str if 'history_str' in locals() else "",
                            mode=actual_mode
                        )
                    # --- FIX: Only run this once. Remove the duplicated self.rag_module() below ---
                    prediction = await loop.run_in_executor(None, safe_generate)
                    answer = prediction.answer
                    
                    if actual_mode == "deep" and hasattr(prediction, 'rationale'):
                        thoughts = prediction.rationale

                    # Anti-Hallucination check
                    lower_ans = answer.lower()
                    if "cannot find" in lower_ans or "context does not contain" in lower_ans:
                        pass # Handled normally
                        
                    sources = [doc[:100] + "..." for doc in final_docs]

                except CircuitBreakerOpenException:
                    logger.error("Circuit Breaker is OPEN.")
                    yield f"data: {json.dumps({'type': 'error', 'answer': 'Service Unavailable', 'thoughts': 'System is busy.'})}\n\n"
                    return
                except Exception as e:
                    logger.error(f"Error during generation: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'answer': 'Generation Error', 'thoughts': str(e)})}\n\n"
                    return

            # 6. Final Response Payload
            payload = {
                'type': 'result',
                'answer': answer,
                'thoughts': thoughts,
                'sources': sources,
                'metadata': {'mode': actual_mode},
                'processing_time': time.time() - start_time
            }
            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            logger.exception("Error in RAG stream")
            yield f"data: {json.dumps({'type': 'error', 'answer': 'System Error', 'thoughts': str(e)})}\n\n"