from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

registry = CollectorRegistry()

rag_query_latency = Histogram(
    'rag_query_duration_seconds',
    'Full RAG query latency',
    registry=registry
)

retrieval_documents = Gauge(
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

retrieval_latency = Gauge(
    'rag_retrieval_latency_seconds',
    'Time taken to retrieve relevant documents from DB',
    registry=registry
)

generation_latency = Gauge(
    'rag_generation_latency_seconds',
    'Time taken by the LLM to generate the response',
    registry=registry
)

# 3. Common aliases (rag_service might ask for these specific names)
query_counter = Counter(
    'rag_queries_total',
    'Total number of RAG queries processed',
    registry=registry
)

token_counter = generation_tokens  # Alias to existing counter

# --- SYSTEM METRICS ---
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