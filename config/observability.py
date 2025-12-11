import structlog
from prometheus_client import CollectorRegistry, start_http_server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from config import settings

def setup_observability(app=None):
    """Initialize logging, metrics, and tracing"""
    
    # Structured Logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Metrics
    if settings.ENABLE_METRICS:
        registry = CollectorRegistry()
        start_http_server(settings.PROMETHEUS_PORT, registry=registry)
    
    # Tracing
    if settings.ENABLE_TRACING and settings.JAEGER_ENDPOINT:
        trace.set_tracer_provider(TracerProvider())
        exporter = OTLPSpanExporter(endpoint=settings.JAEGER_ENDPOINT, insecure=True)
        span_processor = BatchSpanProcessor(exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        if app:
            FastAPIInstrumentor.instrument_app(app)
    
    return structlog.get_logger()