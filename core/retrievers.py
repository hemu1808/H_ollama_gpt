from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import asyncio
from collections import defaultdict
from config import settings
import logging

logger = logging.getLogger(__name__)

class BM25Retriever:
    """Production BM25 with proper parameters"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None
        self.tokenizer = lambda text: re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: List[str], doc_ids: List[str]):
        """Build BM25 index"""
        self.corpus = [self.tokenizer(doc) for doc in documents]
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k document IDs with scores"""
        if not self.bm25:
            return []
        
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top indices
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]

class DistributedBM25:
    """Shard BM25 across multiple indices for scalability"""
    
    def __init__(self, shard_size: int = 10000):
        self.shard_size = shard_size
        self.shards: List[BM25Retriever] = []
        self.doc_id_to_shard: Dict[str, int] = {}
        self.lock = asyncio.Lock()
    
    async def add_documents(
        self,
        docs: List[str],
        doc_ids: List[str]
    ):
        """Add documents to appropriate shards"""
        async with self.lock:
            for i, (doc, doc_id) in enumerate(zip(docs, doc_ids)):
                shard_idx = len(self.shards) - 1
                if not self.shards or len(self.shards[shard_idx].corpus) >= self.shard_size:
                    self.shards.append(BM25Retriever())
                
                # Add to the last shard
                shard = self.shards[-1]
                shard.corpus.append(shard.tokenizer(doc))
                shard.doc_ids.append(doc_id)
                self.doc_id_to_shard[doc_id] = len(self.shards) - 1
            
            # Rebuild IDF for affected shards
            for shard in self.shards:
                if shard.corpus:
                    shard.bm25 = BM25Okapi(shard.corpus, k1=settings.BM25_K1, b=settings.BM25_B)
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Parallel search across all shards"""
        # Search each shard
        shard_results = await asyncio.gather(*[
            asyncio.to_thread(shard.retrieve, query, top_k=top_k*2)
            for shard in self.shards
        ])
        
        # Flatten and merge results
        all_results = []
        for results in shard_results:
            all_results.extend(results)
        
        # Sort by score and deduplicate
        seen = set()
        final_results = []
        for doc_id, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if doc_id not in seen:
                seen.add(doc_id)
                final_results.append((doc_id, score))
                if len(final_results) >= top_k:
                    break
        
        return final_results

class ElasticsearchRetriever:
    """Elasticsearch-based retrieval for web-scale"""
    
    def __init__(self, es_url: str):
        from elasticsearch import AsyncElasticsearch
        self.es = AsyncElasticsearch(es_url)
    
    async def create_index(self, index_name: str):
        """Create index with BM25-like settings"""
        await self.es.indices.create(
            index=index_name,
            body={
                "settings": {
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "k1": settings.BM25_K1,
                            "b": settings.BM25_B
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "similarity": "custom_bm25"
                        },
                        "metadata": {
                            "type": "object"
                        }
                    }
                }
            }
        )
    
    async def add_documents(
        self,
        docs: List[str],
        doc_ids: List[str],
        metadatas: List[dict]
    ):
        """Bulk index documents"""
        from elasticsearch.helpers import async_bulk
        
        actions = [
            {
                "_index": settings.COLLECTION_NAME,
                "_id": doc_id,
                "_source": {
                    "content": doc,
                    "metadata": meta
                }
            }
            for doc, doc_id, meta in zip(docs, doc_ids, metadatas)
        ]
        
        await async_bulk(self.es, actions)
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search with BM25 scoring"""
        resp = await self.es.search(
            index=settings.COLLECTION_NAME,
            body={
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": top_k
            }
        )
        
        return [
            (hit["_id"], hit["_score"])
            for hit in resp["hits"]["hits"]
        ]