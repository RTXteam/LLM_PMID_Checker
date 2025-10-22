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
                 - 'hermes4': Hermes 4 70B via Ollama
                 - 'gpt-oss': GPT-OSS 20B via Ollama
                 
    Returns:
        OllamaClient instance configured for the specified model
    """

    # Normalize provider name - accept both short and full names
    provider_lower = provider.lower()
    
    if 'hermes4' in provider_lower or 'hermes' in provider_lower:
        # Use the full model name if provided, otherwise use default from settings
        model = provider if ':' in provider else settings.hermes_model
        logger.info(f"Using Ollama client for Hermes 4 - Model: {model}")
        return OllamaClient(
            model=model,
            base_url=settings.ollama_base_url
        )
    elif 'gpt-oss' in provider_lower or 'gpt_oss' in provider_lower:
        # Use the full model name if provided, otherwise use default from settings
        model = provider if ':' in provider else settings.gpt_oss_model
        logger.info(f"Using Ollama client for GPT-OSS - Model: {model}")
        return OllamaClient(
            model=model,
            base_url=settings.ollama_base_url
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: hermes4, hermes4:70b, gpt-oss, gpt-oss:20b")