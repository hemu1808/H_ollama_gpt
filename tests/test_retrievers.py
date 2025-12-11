import pytest
import asyncio
from core.retrievers import BM25Retriever, DistributedBM25

@pytest.mark.asyncio
async def test_bm25_retriever():
    retriever = BM25Retriever()
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A fast dark-colored fox leaps above a sleepy canine",
        "Machine learning is fascinating",
    ]
    ids = ["doc1", "doc2", "doc3"]
    
    retriever.fit(docs, ids)
    results = retriever.retrieve("quick fox", top_k=2)
    
    assert len(results) == 2
    assert results[0][0] in ["doc1", "doc2"]  # Should match either

@pytest.mark.asyncio
async def test_distributed_bm25():
    retriever = DistributedBM25(shard_size=2)
    docs = [f"Document {i}" for i in range(100)]
    ids = [f"id_{i}" for i in range(100)]
    
    await retriever.add_documents(docs, ids)
    results = await retriever.retrieve("Document 50", top_k=5)
    
    assert len(results) == 5
    assert "id_50" in [r[0] for r in results]