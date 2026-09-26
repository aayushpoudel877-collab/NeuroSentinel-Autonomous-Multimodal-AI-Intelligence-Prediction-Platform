from app.security.auth import require_api_key
def test_auth_disabled_by_default(): assert require_api_key(None) is None
