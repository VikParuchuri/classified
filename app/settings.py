import os
from typing import Optional

from dotenv import find_dotenv
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Path settings
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    LENS_DIR: str = os.path.join(BASE_DIR, "lenses")

    # Database
    DATABASE_URL = f"sqlite:///{BASE_DIR}/db.sqlite3"
    DEBUG: bool = False

    # Chunks to rate
    TOKENS_PER_CHUNK: int = 2000
    CHUNKS_PER_DOC: int = 3
    MAX_SEARCH_DISTANCE: int = 100

    # LLM
    OPENAI_KEY: str = ""
    OPENAI_BASE_URL: Optional[str] = None
    CHAT_MODEL: str = "gpt-3.5-turbo"
    INSTRUCT_MODEL: str = "gpt-3.5-turbo-instruct"

    class Config:
        env_file = find_dotenv("local.env")


settings = Settings()
