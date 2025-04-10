from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

# Replace with your secure key and store in environment variables in production
API_KEY = "WISHLINK_GRAPHITI_API_KEY_2025_VERY_LFKASDJFLKASDJFLKAFS12341412894u10298hjkf1938ndnirlfiu_PASSWORD" 
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate API key"
    )