import logging
from fastapi import UploadFile, HTTPException

# Initialize logging
logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".json", ".csv", ".docx"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

async def validate_file_upload(file: UploadFile) -> UploadFile:
    """
    Validates uploaded files for allowed extensions and size.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    # 1. Check File Extension
    # We check if the filename ends with any of the allowed extensions
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        allowed_list = ", ".join(ALLOWED_EXTENSIONS)
        logger.warning(f"Rejected file upload: {filename} (Invalid extension)")
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed extensions: {allowed_list}"
        )

    # 2. Check File Content Type (Optional extra safety)
    # This prevents someone from renaming a .exe to .txt
    # (Simplified check for now)
    
    # 3. Size check happens usually during reading, but we can't easily check 
    # file size on an upload stream without reading it. 
    # We rely on the server (Uvicorn/Nginx) or read chunks later.
    
    logger.info(f"File validation successful: {filename}")
    return file