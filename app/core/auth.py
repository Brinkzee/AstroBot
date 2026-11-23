"""Optional admin token authentication dependency for mutating/privileged endpoints."""

import hmac
from typing import Optional
from fastapi import Header, HTTPException

from app.config import settings


def verify_admin_token(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    """Enforce admin token when ``settings.admin_api_token`` is configured (non-empty).

    When ``admin_api_token`` is empty (default local dev and test mode), requests are
    allowed without a token to preserve backward compatibility.
    """
    expected = (settings.admin_api_token or "").strip()
    if not expected:
        return

    provided: Optional[str] = None
    if x_admin_token is not None and x_admin_token.strip():
        provided = x_admin_token.strip()
    elif authorization is not None:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()

    if not provided:
        raise HTTPException(status_code=401, detail="缺少管理员鉴权令牌 (X-Admin-Token 或 Authorization: Bearer)")

    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="管理员鉴权令牌无效")
