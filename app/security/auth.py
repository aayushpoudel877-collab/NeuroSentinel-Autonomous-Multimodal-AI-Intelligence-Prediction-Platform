import hashlib
import hmac
import os
from fastapi import Header, HTTPException

API_KEY_HASH=os.getenv('NEUROSENTINEL_API_KEY_SHA256','')

def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY_HASH:
        return None
    if not x_api_key or not hmac.compare_digest(hashlib.sha256(x_api_key.encode()).hexdigest(),API_KEY_HASH):
        raise HTTPException(status_code=401,detail='invalid API key')
    return x_api_key
