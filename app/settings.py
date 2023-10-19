import os
from typing import Optional

from dotenv import find_dotenv
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Path settings
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    LENS_DIR: str = os.path.join(BASE_DIR, "lenses")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")

    # Database
    DATABASE_URL = f"sqlite:///{BASE_DIR}/db.sqlite3"
    DEBUG: bool = False

    # Chunks to rate
    TOKENS_PER_CHUNK: int = 1500  # How many tokens to take per chunk
    CHUNKS_PER_DOC: int = 2  # How many chunks to rate in each resource (averaging scores across chunks)
    MAX_SEARCH_DISTANCE: int = 100  # The maximum number of tokens to travel forward to find a line edge to split on when chunking

    # LLM
    OPENAI_KEY: str = ""
    CHAT_MODEL: str = "gpt-3.5-turbo"  # The primary model to use for chat completions (function calling)
    INSTRUCT_MODEL: str = "gpt-3.5-turbo-instruct"
    MAX_GENERATION_TOKENS: int = 256  # Max number of tokens to generate (function call response)

    class Config:
        env_file = find_dotenv("local.env")


settings = Settings()
