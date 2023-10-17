from typing import Optional, List, Dict

from sqlalchemy import Column, JSON
from sqlmodel import Field, UniqueConstraint, select

from app.db.base import BaseDBModel
from app.db.session import get_session
from app.settings import settings
import hashlib


class CachedResponse(BaseDBModel, table=True):
    __table_args__ = (UniqueConstraint("hash", "lens_type", "model", "version", name="unique_hash_model_version"),)
    hash: str = Field(index=True) # Hash of input prompt + functions
    lens_type: str # The type of lens we're running
    messages: List[Dict] = Field(sa_column=Column(JSON), default=list())
    functions: str | None = Field(nullable=True)
    response: Dict = Field(sa_column=Column(JSON), default=dict())
    model: str
    version: int = Field(default=1)


def get_hash(messages: List[Dict], functions: str):
    key = f"{functions}{messages}".encode("utf-8")
    hashed = hashlib.sha256(key).hexdigest()
    return hashed


def query_cached_response(lens_type: str, messages: List[Dict], functions: str, version: int = 1) -> Optional[CachedResponse]:
    model = settings.CHAT_MODEL
    hashed = get_hash(messages, functions)
    with get_session() as db:
        query = db.execute(
            select(CachedResponse).where(
                CachedResponse.hash == hashed,
                CachedResponse.lens_type == lens_type,
                CachedResponse.model == model,
                CachedResponse.version == version,
            )
        )
        cached_response = query.first()
    return cached_response[0] if cached_response else None


def save_cached_response(lens_type: str, messages: List[Dict], functions: str, response: str, version: int = 1):
    model = settings.CHAT_MODEL
    hashed = get_hash(messages, functions)
    with get_session() as db:
        cached_response = CachedResponse(
            hash=hashed,
            lens_type=lens_type,
            messages=messages,
            functions=functions,
            response=response,
            model=model,
            version=version,
        )
        db.add(cached_response)
        db.commit()


