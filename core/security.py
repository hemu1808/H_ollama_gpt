from pydantic import BaseModel, validator
from typing import Optional
import re
import magic
import aiofiles
from config import settings

class QueryInput(BaseModel):
    """Validated query input"""
    question: str
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
    
    @staticmethod
    async def validate_file_content(file_path: str) -> bool:
        """Verify file is actually PDF"""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type == 'application/pdf'
    
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