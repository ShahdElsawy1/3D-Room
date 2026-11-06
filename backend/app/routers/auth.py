"""POST /auth/signup, POST /auth/login (JSON), POST /auth/token (form, for the /docs Authorize button), GET /auth/me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_current_user, get_db, get_settings
from ..models import User
from ..schemas import Credentials, TokenOut, UserOut
from ..security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _token(user: User, settings: Settings) -> TokenOut:
    return TokenOut(access_token=create_access_token(user.id, settings.jwt_secret, settings.jwt_expire_minutes),
                    expires_in=settings.jwt_expire_minutes * 60, user=UserOut.model_validate(user))


def _authenticate(db: Session, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return user


@router.post("/signup", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def signup(body: Credentials, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    user = User(email=str(body.email).lower(), password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    return _token(user, settings)


@router.post("/login", response_model=TokenOut)
def login(body: Credentials, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    return _token(_authenticate(db, str(body.email), body.password), settings)


@router.post("/token", response_model=TokenOut, include_in_schema=True)
def token(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db),
          settings: Settings = Depends(get_settings)):
    """OAuth2 password form (username = email). Used by the Swagger UI; the frontend should use /auth/login."""
    return _token(_authenticate(db, form.username, form.password), settings)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
