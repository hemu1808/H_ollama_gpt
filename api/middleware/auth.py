import os
import logging
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

# Initialize logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    Verifies the Bearer token provided in the Authorization header.
    
    logic:
    1. Checks if API_KEY is set in environment variables.
    2. If NOT set, it allows the request (Development Mode).
    3. If SET, it checks if the incoming token matches the API_KEY.
    """
    env_api_key = os.getenv("API_KEY")
    
    # --- DEV MODE (Safe Default) ---
    # If no API key is set in .env, we assume this is a local test and allow access.
    if not env_api_key:
        logger.debug("No API_KEY set in environment. Allowing request (Dev Mode).")
        return {"user": "dev_user", "status": "authenticated"}

    # --- PRODUCTION MODE ---
    token = credentials.credentials
    if token != env_api_key:
        logger.warning("Invalid API Key attempt.")
        raise HTTPException(
            status_code=403, 
            detail="Invalid authentication credentials"
        )
    
    return {"user": "authenticated_admin", "status": "authenticated"}