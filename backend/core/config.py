import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "PIM (Pakistan Immigration Manager)"
    # Will look for GROQ_API_KEY in environment or .env file
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-specdec")

    class Config:
        env_file = ".env"

settings = Settings()
