"""Factory for creating LLM clients based on configuration."""
import logging
from typing import Any, Optional
from .config import settings
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

def create_llm_client(provider: str) -> Any:
    """Create LLM client based on provider setting.
    
    Args:
        provider: LLM provider to use:
                 - 'hermes4': Hermes 4 70B via Ollama (primary reasoning model)
                 - 'gpt-oss': GPT-OSS 120B via Ollama (production quality model)
                 
    Returns:
        OllamaClient instance configured for the specified model
    """

    if provider == "hermes4":
        logger.info(f"Using Ollama client for Hermes 4 - Model: {settings.hermes_model}")
        return OllamaClient(
            model=settings.hermes_model,
            base_url=settings.ollama_base_url
        )
    elif provider == "gpt-oss":
        logger.info(f"Using Ollama client for GPT-OSS - Model: {settings.gpt_oss_model}")
        return OllamaClient(
            model=settings.gpt_oss_model,
            base_url=settings.ollama_base_url
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: hermes4, gpt-oss")