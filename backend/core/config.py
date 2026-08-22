import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "PIM (Pakistan Immigration Manager)"
    # Will look for GROQ_API_KEY in environment or .env file
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    class Config:
        env_file = ".env"

settings = Settings()
