"""FastAPI dependencies: settings, DB session, current user, owned project."""
from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import Settings
from .models import Project, User
from .security import decode_access_token

# tokenUrl powers the "Authorize" button in /docs; the frontend uses POST /auth/login (JSON)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db),
                     settings: Settings = Depends(get_settings)) -> User:
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                                 headers={"WWW-Authenticate": "Bearer"})
    if not token:
        raise unauthorized
    user_id = decode_access_token(token, settings.jwt_secret)
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise unauthorized
    return user


def get_owned_project(project_id: str, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)) -> Project:
    """Projects are per user (spec 4.4: "user project not global"). Someone else's project is a 404, not a 403,
    so ids of other users' projects are not revealed."""
    project = db.get(Project, project_id)
    if project is None or project.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project
