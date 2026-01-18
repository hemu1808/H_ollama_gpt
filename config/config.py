from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RAG"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WORKERS: int = 4
    
    # Security
    SECRET_KEY: str = "change-in-production"
    ALLOWED_HOSTS: List[str] = ["*"]
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # LLM
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3.1:8b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    os.environ["OLLAMA_API_KEY"] = "ollama"

    # Vector Store
    CHROMADB_PATH: str = "./data/chroma"
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8000
    COLLECTION_NAME: str = "rag"
    DEFAULT_N_RESULTS: int = 20
    
    # Retrieval
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_MULTI_QUERY: bool = True
    ENABLE_QUERY_EXPANSION: bool = True
    HYBRID_SEARCH_ALPHA: float = 0.5  # Weight for semantic
    RERANK_TOP_K: int = 10
    
    # BM25
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    BM25_SHARD_SIZE: int = 10000
    BM25_USE_ELASTICSEARCH: bool = False
    ELASTICSEARCH_URL: Optional[str] = None
    
    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    ENABLE_SEMANTIC_CHUNKING: bool = True
    SEMANTIC_CHUNK_THRESHOLD: float = 0.7
    
    # Context Compression
    ENABLE_CONTEXT_COMPRESSION: bool = True
    CONTEXT_COMPRESSION_RATIO: float = 0.5
    MAX_CONTEXT_LENGTH: int = 8192
    
    # Caching
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600
    EMBEDDING_CACHE_SIZE: int = 100000
    
    # Background Processing
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Observability
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    PROMETHEUS_PORT: int = 9090
    JAEGER_ENDPOINT: Optional[str] = None
    
    # Evaluation
    EVALUATION_DATASET_PATH: Optional[str] = None
    MIN_FAITHFULNESS_SCORE: float = 0.85
    
    # Multi-Modal
    ENABLE_MULTI_MODAL: bool = False
    CLIP_MODEL: str = "clip-ViT-B-32"
    
    # Graph RAG
    ENABLE_GRAPH_RAG: bool = False
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    
    # --- FIXED: Modern Pydantic V2 Config ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Prevents crashes if your .env has extra keys
    )

settings = Settings()