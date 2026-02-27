import os
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    PROJECT_NAME: str = "Titanic Chat Agent API"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # llama-3.3-70b-versatile: 14,400 req/day free tier on Groq
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def validate_api_key(self) -> None:
        """Warn loudly at startup if the API key is missing."""
        if not self.GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is not set. "
                "The agent will return an error on every query until the key is provided. "
                "Add it to your .env file and restart the server."
            )


settings = Settings()
