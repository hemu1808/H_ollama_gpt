from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

registry = CollectorRegistry()

# Core RAG metrics
rag_query_latency = Histogram(
    'rag_query_duration_seconds',
    'Full RAG query latency',
    registry=registry
)

retrieval_documents = Histogram(
    'rag_retrieval_documents',
    'Number of documents retrieved',
    registry=registry
)

generation_tokens = Counter(
    'rag_generation_tokens_total',
    'Total tokens generated',
    registry=registry
)

hallucination_rate = Gauge(
    'rag_hallucination_rate',
    'Rate of detected hallucinations',
    registry=registry
)

# System metrics
chroma_connection_failures = Counter(
    'chroma_connection_failures_total',
    'ChromaDB connection failures',
    registry=registry
)

ollama_errors = Counter(
    'ollama_errors_total',
    'Ollama API errors',
    ['model', 'error_type'],
    registry=registry
)