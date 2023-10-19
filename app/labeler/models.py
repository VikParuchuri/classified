from typing import Optional, List, Dict

from sqlmodel import Field, UniqueConstraint, select

from app.db.base import BaseDBModel


class TrainingData(BaseDBModel, table=True):
    __table_args__ = (UniqueConstraint("hash", "lens_type", "model", "version", name="unique_hash_model_version"),)
    hash: str = Field(index=True) # Hash of input passage
    lens_type: str # The type of lens we're running
    passage: str # The input passage to rate
    ratings: List[int] # The list of flat ratings
    model: str
    version: int = Field(default=1)


