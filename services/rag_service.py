import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel
from config import settings
from core import (
    EmbeddingCache,
    BM25Retriever,
    DistributedBM25,
    ElasticsearchRetriever,
    SecurityValidator,
    CircuitBreaker
)
from observability import metrics
import structlog

logger = structlog.get_logger()

class AnswerResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    metadata: Dict[str, Any]

class RAGService:
    """Production RAG service with all advanced features"""
    
    def __init__(self):
        self.embedding_cache = EmbeddingCache(settings.REDIS_URL)
        self.security = SecurityValidator()
        
        # Initialize retriever based on config
        if settings.BM25_USE_ELASTICSEARCH and settings.ELASTICSEARCH_URL:
            self.bm25 = ElasticsearchRetriever(settings.ELASTICSEARCH_URL)
            self.use_es = True
        else:
            self.bm25 = DistributedBM25(settings.BM25_SHARD_SIZE)
            self.use_es = False
        
        self.circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        # ChromaDB client (async wrapper needed)
        self.chroma_client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize async components"""
        # Setup ChromaDB
        from chromadb import AsyncHttpClient
        self.chroma_client = await AsyncHttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT
        )
        self.collection = await self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )
        
        # Initialize BM25
        await self._rebuild_bm25_index()
    
    async def _rebuild_bm25_index(self):
        """Rebuild BM25 index from all documents"""
        # Fetch all documents from ChromaDB
        results = await self.collection.get()
        
        if results['documents']:
            if self.use_es:
                await self.bm25.add_documents(
                    results['documents'],
                    results['ids'],
                    results['metadatas']
                )
            else:
                await self.bm25.add_documents(
                    results['documents'],
                    results['ids']
                )
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str]
    ):
        """Add documents to both vector and keyword stores"""
        # Add to ChromaDB
        await self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Add to BM25
        if self.use_es:
            await self.bm25.add_documents(documents, ids, metadatas)
        else:
            await self.bm25.add_documents(documents, ids)
        
        # Invalidate cache
        await self.embedding_cache.clear_pattern("emb:*")
    
    async def query_expansion(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[str]:
        """Expand query with circuit breaker protection"""
        @self.circuit
        async def _expand():
            from ollama import AsyncClient
            client = AsyncClient(host=settings.OLLAMA_URL)
            
            prompt = f"""Generate {num_variations} query variations for: {query}"""
            
            response = await client.chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            
            variations = [
                v.strip() for v in response['message']['content'].split('\n')
                if v.strip()
            ]
            
            return [query] + variations[:num_variations]
        
        try:
            return await asyncio.wait_for(_expand(), timeout=5.0)
        except Exception as e:
            logger.warning("expansion_failed", error=str(e))
            return [query]
    
    async def hybrid_search(
        self,
        query: str,
        n_results: int = 20,
        filter_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Async hybrid search with metrics"""
        with metrics.retrieval_latency.time():
            # Semantic search
            sem_task = asyncio.create_task(
                self.collection.query(
                    query_texts=[query],
                    n_results=n_results * 2,
                    where=filter_metadata
                )
            )
            
            # BM25 search
            key_task = asyncio.create_task(
                self.bm25.retrieve(query, top_k=n_results * 2)
            )
            
            sem_results, key_results = await asyncio.gather(sem_task, key_task)
            
            # Fuse with RRF
            fused = await self._fuse_results_rrf(sem_results, key_results, n_results)
            
            metrics.retrieval_documents.set(len(fused['documents'][0]))
            return fused
    
    async def _fuse_results_rrf(
        self,
        semantic: Dict,
        keyword: List[Tuple[str, float]],
        n_results: int,
        k: int = 60
    ) -> Dict:
        """Reciprocal Rank Fusion with proper ranking"""
        scores = defaultdict(float)
        
        # Semantic scores (rank by distance)
        sem_docs = semantic['documents'][0]
        sem_ids = semantic['ids'][0]
        
        for rank, (doc_id, distance) in enumerate(zip(sem_ids, semantic['distances'][0])):
            # Convert distance to rank (lower distance = higher rank)
            scores[doc_id] += 1 / (k + rank)
        
        # Keyword scores
        for rank, (doc_id, score) in enumerate(keyword, 1):
            scores[doc_id] += 1 / (k + rank)
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final result
        return {
            'documents': [[doc_id for doc_id, _ in sorted_docs[:n_results]]],
            'ids': [[doc_id for doc_id, _ in sorted_docs[:n_results]]],
            # ... metadatas, distances
        }
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Cross-encoder reranking"""
        from sentence_transformers import CrossEncoder
        
        encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
        
        # Score pairs
        pairs = [[query, doc] for doc in documents]
        scores = encoder.predict(pairs)
        
        # Sort
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [documents[i] for i in top_indices], [float(scores[i]) for i in top_indices]
    
    async def answer_question(
        self,
        query_input: QueryInput
    ) -> AnswerResponse:
        """Main RAG pipeline with full instrumentation"""
        try:
            # Validate input
            query = await self.security.sanitize_query(query_input.question)
            
            # Expand query
            if query_input.use_query_expansion:
                queries = await self.query_expansion(query)
            else:
                queries = [query]
            
            # Multi-query search
            all_results = []
            for q in queries:
                results = await self.hybrid_search(q, n_results=settings.DEFAULT_N_RESULTS)
                all_results.append(results)
            
            # Deduplicate
            unique_docs = await self._deduplicate_results(all_results)
            
            # Rerank
            docs, scores = await self.rerank(
                query,
                unique_docs['documents'][0],
                top_k=settings.RERANK_TOP_K
            )
            
            # Build context
            context = "\n\n".join(docs)
            
            # Generate answer
            answer = await self._generate_answer(context, query)
            
            # Compute confidence
            confidence = await self._compute_confidence(
                answer, context, scores, query
            )
            
            # Build response
            return AnswerResponse(
                answer=answer,
                sources=[{
                    "doc": doc,
                    "score": float(score)
                } for doc, score in zip(docs, scores)],
                confidence=confidence,
                metadata={
                    "query": query,
                    "search_method": "hybrid",
                    "num_docs": len(docs)
                }
            )
            
        except Exception as e:
            logger.error("rag_pipeline_failed", error=str(e))
            raise
    
    async def _generate_answer(self, context: str, query: str) -> str:
        """Generate answer with streaming"""
        from ollama import AsyncClient
        client = AsyncClient(host=settings.OLLAMA_URL)
        
        prompt = f"""Context: {context}\n\nQuestion: {query}\nAnswer:"""
        
        chunks = []
        async for chunk in await client.chat(
            model=settings.OLLAMA_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            chunks.append(chunk['message']['content'])
        
        return "".join(chunks)
    
    async def _compute_confidence(
        self,
        answer: str,
        context: str,
        scores: List[float],
        query: str
    ) -> float:
        """Multi-signal confidence score"""
        signals = {
            'retrieval_confidence': np.percentile(scores, 90) if scores else 0.0,
            'answer_length': len(answer) / 1000,  # Normalize
            'context_overlap': len(set(answer.lower().split()) & set(context.lower().split())) / len(set(query.lower().split())),
        }
        
        # Weighted average
        weights = {'retrieval_confidence': 0.5, 'answer_length': 0.2, 'context_overlap': 0.3}
        return sum(signals[k] * weights[k] for k in signals)