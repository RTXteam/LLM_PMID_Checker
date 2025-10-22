"""Configuration settings for the LLM PMID Checker system."""
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    """Application settings."""
    
    # Ollama configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    hermes_model: str = os.getenv("HERMES_MODEL", "hermes4:70b")
    
    # GPT-OSS configuration
    gpt_oss_model: str = os.getenv("GPT_OSS_MODEL", "gpt-oss:20b")
    
    # NCBI E-utilities configuration
    ncbi_email: Optional[str] = os.getenv("NCBI_EMAIL")
    ncbi_api_key: Optional[str] = os.getenv("NCBI_API_KEY")
    
    # UMLS configuration
    umls_api_key: Optional[str] = os.getenv("UMLS_API_KEY")
    use_umls: bool = os.getenv("USE_UMLS", "true").lower() == "true"
    
    # Request settings
    max_retries: int = 3
    request_timeout: int = 180

# Global settings instance
settings = Settings()