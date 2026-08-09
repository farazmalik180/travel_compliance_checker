import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Travel Compliance Checker"
    # Will look for GROQ_API_KEY in environment or .env file
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    class Config:
        env_file = ".env"

settings = Settings()
