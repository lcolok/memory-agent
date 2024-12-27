"""Utility functions used in our graph."""

from typing import Any, Dict, Optional
from memory_agent.llms.siliconflow import SiliconFlow
from memory_agent.embeddings.siliconflow import SiliconFlowEmbeddings


def split_model_and_provider(fully_specified_name: str) -> dict:
    """Initialize the configured chat model.

    Args:
        fully_specified_name: The fully specified model name, which can be in one of these formats:
            - "provider/model" (e.g., "siliconflow/Qwen/Qwen2.5-72B-Instruct")
            - "model" (e.g., "Qwen/Qwen2.5-72B-Instruct")

    Returns:
        A dictionary containing the model name and provider.
    """
    # Always use SiliconFlow as the provider
    provider = "siliconflow"
    model = fully_specified_name

    # If the model name starts with "siliconflow/", remove it
    if model.startswith("siliconflow/"):
        model = model[len("siliconflow/"):]

    return {"model": model, "provider": provider}


def initialize_chat_model(model_config: Dict[str, Any]) -> Any:
    """Initialize the chat model based on provider."""
    provider = model_config.get("provider")
    model = model_config.get("model")

    if provider == "siliconflow":
        return SiliconFlow.create(model=model)
    # Add other providers here as needed
    raise ValueError(f"Unsupported model provider: {provider}")


def initialize_embedding_model(model_name: str) -> Any:
    """Initialize the embedding model."""
    return SiliconFlowEmbeddings(model=model_name)
