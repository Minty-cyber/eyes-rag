import secrets
import logging
import os
from typing import Annotated, Any, Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
    

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    DEBUG: bool = False
    PROJECT_NAME: str = 'Merve'
    DESCRIPTION: str = "RAG Chatbot"
    VERSION: str = '1.0.0'
    
    GROQ_API_KEY: str = ""
   

    
    GOOGLE_API_KEY: str = ""
    


settings = Settings()
