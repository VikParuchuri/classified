from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.settings import settings

engine = create_engine(
    settings.DATABASE_URL,
    # Ensure we only have one connection at a time active
    pool_size=1,
    max_overflow=0,
    pool_timeout=30,
    poolclass=QueuePool,
    isolation_level='SERIALIZABLE'
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_session() -> Session:
    return SessionLocal()
