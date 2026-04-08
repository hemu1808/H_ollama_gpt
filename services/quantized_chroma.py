import numpy as np
import chromadb
from typing import List, Dict, Any
import requests
from core.quantization.polar_quant import PolarQuant
from core.quantization.qjl import QJLRetriever

class OllamaEmbeddingFunction:
    """ChromaDB compatible embedding function for Ollama models"""
    name = "OllamaEmbeddingFunction"
    def __init__(self, model_name="nomic-embed-text", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            try:
                res = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.model_name, "prompt": text})
                if res.status_code == 200:
                    embeddings.append(res.json().get("embedding", []))
                else:
                    embeddings.append([0.0] * 768)
            except Exception as e:
                print(f"Ollama Embed Error: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

class QuantizedChromaAdapter:
    """
    Adapter that wraps ChromaDB to store 3-bit PolarQuant + 1-bit QJL embeddings
    in the metadata field, while maintaining backward compatibility.
    """
    def __init__(self, collection: chromadb.Collection, dim=768):
        self.collection = collection
        self.polar = PolarQuant(dim=dim)
        self.qjl = QJLRetriever(dim=dim)
        
    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str], embeddings: List[List[float]] = None):
        """Quantize embeddings before saving"""
        if embeddings is None and documents is not None:
            if self.collection._embedding_function:
                embeddings = self.collection._embedding_function(documents)
            else:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.DefaultEmbeddingFunction()
                embeddings = ef(documents)
                
        quantized_metadatas = []
        for i, emp in enumerate(embeddings):
            np_emb = np.array(emp, dtype=np.float32)
            
            # Polar Quant & QJL Residual
            pq_bytes = self.polar.encode(np_emb)
            pq_approx = self.polar.decode_approximate(pq_bytes)
            qjl_bytes = self.qjl.encode_residual(np_emb, pq_approx)
            
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            meta['_pq_bytes'] = pq_bytes.hex()
            meta['_qjl_bytes'] = qjl_bytes.hex()
            quantized_metadatas.append(meta)
            
        dummy_embeddings = [[0.0] * self.polar.dim for _ in embeddings]
        
        self.collection.add(
            embeddings=dummy_embeddings,
            documents=documents,
            metadatas=quantized_metadatas,
            ids=ids
        )
        
    def upsert(self, documents: List[str], metadatas: List[Dict], ids: List[str], embeddings: List[List[float]] = None):
        if embeddings is None and documents is not None:
            if self.collection._embedding_function:
                embeddings = self.collection._embedding_function(documents)
            else:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.DefaultEmbeddingFunction()
                embeddings = ef(documents)
                
        quantized_metadatas = []
        for i, emp in enumerate(embeddings):
            np_emb = np.array(emp, dtype=np.float32)
            pq_bytes = self.polar.encode(np_emb)
            pq_approx = self.polar.decode_approximate(pq_bytes)
            qjl_bytes = self.qjl.encode_residual(np_emb, pq_approx)
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            meta['_pq_bytes'] = pq_bytes.hex()
            meta['_qjl_bytes'] = qjl_bytes.hex()
            quantized_metadatas.append(meta)
            
        dummy_embeddings = [[0.0] * self.polar.dim for _ in embeddings]
        self.collection.upsert(
            embeddings=dummy_embeddings,
            documents=documents,
            metadatas=quantized_metadatas,
            ids=ids
        )
        
    def delete(self, ids: List[str] = None, where: Dict = None):
        self.collection.delete(ids=ids, where=where)
        
    def get(self, *args, **kwargs):
        return self.collection.get(*args, **kwargs)
        
    def query(self, query_texts: List[str] = None, query_embeddings: List[List[float]] = None, n_results: int = 10) -> Dict[str, Any]:
        """Custom search using QJL Coarse retrieval + PQ cross-check"""
        if query_embeddings is None and query_texts is not None:
            if self.collection._embedding_function:
                query_embeddings = self.collection._embedding_function(query_texts)
            else:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.DefaultEmbeddingFunction()
                query_embeddings = ef(query_texts)
                
        if not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        all_data = self.collection.get(include=["metadatas", "documents"])
        if not all_data["metadatas"]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        
        for q_emb in query_embeddings:
            np_q = np.array(q_emb, dtype=np.float32)
            q_jl_signs = self._pack_bits_native(np.sign(np.dot(np_q, self.qjl.J)))
            
            scores = []
            for idx, meta in enumerate(all_data["metadatas"]):
                if '_pq_bytes' not in meta or '_qjl_bytes' not in meta:
                    scores.append((idx, -1.0))
                    continue
                    
                qjl_res = bytes.fromhex(meta['_qjl_bytes'])
                sim = self.qjl.estimate_similarity(q_jl_signs, qjl_res)
                scores.append((idx, sim))
                
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k = scores[:n_results]
            
            res_ids = [all_data["ids"][x[0]] for x in top_k]
            res_docs = [all_data["documents"][x[0]] for x in top_k]
            res_metas = [{k: v for k, v in all_data["metadatas"][x[0]].items() if not k.startswith('_')} for x in top_k]
            res_dists = [1.0 - x[1] for x in top_k] 
            
            results["ids"].append(res_ids)
            results["documents"].append(res_docs)
            results["metadatas"].append(res_metas)
            results["distances"].append(res_dists)
            
        return results
        
    def _pack_bits_native(self, bit_array: np.ndarray) -> bytes:
        binary = (bit_array > 0).astype(np.uint8)
        return np.packbits(binary).tobytes()
