import pickle
import os
import time
import numpy as np
import logging
import sys
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re
from config import settings

logger = logging.getLogger(__name__)

# --- CROSS-PLATFORM FILE LOCKING ---
class FileLock:
    def __init__(self, filename):
        self.filename = filename
        self.handle = None

    def acquire(self):
        if os.name == 'nt':  # Windows
            import msvcrt
            self.handle = open(self.filename, 'w')
            # Lock the first byte of the file
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                # If already locked, wait and retry (simple spinlock)
                time.sleep(0.1)
                try:
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                except IOError:
                    raise BlockingIOError("Resource locked")
        else:  # Linux/Mac
            import fcntl
            self.handle = open(self.filename, 'w')
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self):
        if self.handle:
            if os.name == 'nt':
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

class PersistedBM25Retriever:
    def __init__(self, index_path: str = "./data/bm25_index.pkl"):
        self.index_path = index_path
        self.k1 = getattr(settings, 'BM25_K1', 1.5)
        self.b = getattr(settings, 'BM25_B', 0.75)
        self.corpus = []
        self.doc_ids = []
        self.bm25 = None
        self.last_mtime = 0
        
        self.load_index_if_fresh()

    def tokenizer(self, text: str):
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    def _get_file_mtime(self):
        if os.path.exists(self.index_path):
            return os.path.getmtime(self.index_path)
        return 0

    def load_index_if_fresh(self):
        current_mtime = self._get_file_mtime()
        if current_mtime > self.last_mtime:
            self.load_index()
            self.last_mtime = current_mtime

    def index_documents(self, documents: List[str], doc_ids: List[str]):
        """Thread-safe indexing using cross-platform lock."""
        lock_path = self.index_path + ".lock"
        
        try:
            with FileLock(lock_path):
                # Reload explicitly to get latest state from other workers
                self.load_index() 
                
                # Update Memory
                tokenized_docs = [self.tokenizer(doc) for doc in documents]
                self.corpus.extend(tokenized_docs)
                self.doc_ids.extend(doc_ids)
                
                # Re-calculate BM25
                self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
                
                # Save
                self.save_index()
        except BlockingIOError:
            logger.warning("Could not acquire lock for BM25 index, skipping update this time.")
        except Exception as e:
            logger.error(f"Error updating BM25: {e}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        self.load_index_if_fresh()
        
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = self.tokenizer(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        return results

    def save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'doc_ids': self.doc_ids
            }, f)
        self.last_mtime = os.path.getmtime(self.index_path)

    def load_index(self):
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.corpus = data.get('corpus', [])
                self.doc_ids = data.get('doc_ids', [])
                if self.corpus:
                    self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")

def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]], 
    keyword_results: List[Tuple[str, float]], 
    k: int = 60
) -> List[Tuple[str, float]]:
    fused_scores = {}
    if not vector_results: vector_results = []
    if not keyword_results: keyword_results = []

    for rank, (doc_id, _) in enumerate(vector_results):
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)
        
    for rank, (doc_id, _) in enumerate(keyword_results):
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)
    
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)