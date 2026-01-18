from pydantic import BaseModel, validator
from typing import Optional
import re
import logging
from config import settings

# Try importing magic, but don't crash if it fails (common Windows issue)
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryInput(BaseModel):
    """Validated query input"""
    question: str
    collection_name: Optional[str] = "default"
    top_k: int = 4
    use_hybrid: bool = settings.ENABLE_HYBRID_SEARCH
    use_multi_query: bool = settings.ENABLE_MULTI_QUERY
    use_query_expansion: bool = settings.ENABLE_QUERY_EXPANSION
    
    @validator('question')
    def validate_question(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Question must be at least 3 characters")
        if len(v) > 1000:
            raise ValueError("Question too long (max 1000 characters)")
        
        # Sanitize
        v = re.sub(r'[<>]', '', v)
        return v.strip()

class DocumentInput(BaseModel):
    """Validated document upload"""
    filename: str
    size: int
    content_type: str
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v.endswith('.pdf'):
            raise ValueError("Only PDF files supported")
        return v
    
    @validator('size')
    def validate_size(cls, v):
        if v > 50 * 1024 * 1024:  # 50MB
            raise ValueError("File too large (max 50MB)")
        return v

class SecurityValidator:
    """Security utilities"""
    
    # --- THIS WAS MISSING AND CAUSED THE ERROR ---
    @staticmethod
    async def sanitize_query(query: str) -> str:
        """Sanitize user query"""
        if not query:
            return ""
        # Remove potentially dangerous characters like HTML tags
        return re.sub(r'[<>]', '', query).strip()
    # ---------------------------------------------

    @staticmethod
    async def validate_file_content(file_path: str) -> bool:
        """Verify file is actually PDF"""
        if not MAGIC_AVAILABLE:
            logger.warning("python-magic not installed, skipping deep file inspection")
            return True
            
        try:
            # Different versions of python-magic use different syntax
            # This is the safest way to cover both
            if hasattr(magic, 'from_file'):
                file_type = magic.from_file(file_path, mime=True)
            else:
                m = magic.Magic(mime=True)
                file_type = m.from_file(file_path)
                
            return file_type == 'application/pdf'
        except Exception as e:
            logger.error(f"Error validating file content: {e}")
            # Fail safe: allow if check fails, or deny if strict security needed
            return True 
    
    @staticmethod
    def sanitize_metadata(metadata: dict) -> dict:
        """Remove dangerous characters from metadata"""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                sanitized[k] = re.sub(r'[<>\"\'\n\r]', '', v)
            else:
                sanitized[k] = v
        return sanitized