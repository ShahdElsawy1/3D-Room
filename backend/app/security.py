"""Password hashing (bcrypt) and JWT access tokens (HS256)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def create_access_token(user_id: str, secret: str, expire_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "iat": now, "exp": now + timedelta(minutes=expire_minutes), "typ": "access"}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> str | None:
    """Returns the user id, or None if the token is invalid/expired."""
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "access":
        return None
    return payload.get("sub")


def create_file_token(project_id: str, secret: str, expire_minutes: int) -> tuple[str, datetime]:
    """Read-only token for one project's files, put in URLs as ?t=... (splat viewers and <video>/<img> tags
    cannot send an Authorization header)."""
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=expire_minutes)
    return jwt.encode({"sub": project_id, "iat": now, "exp": exp, "typ": "file"}, secret, algorithm=ALGORITHM), exp


def decode_file_token(token: str, secret: str) -> str | None:
    """Returns the project id the token grants read access to, or None."""
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub") if payload.get("typ") == "file" else None
