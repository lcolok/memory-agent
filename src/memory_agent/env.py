"""Environment utilities."""

import os


def load_api_key(key_name: str) -> str:
    """Load an API key from environment variables.

    Args:
        key_name: The name of the environment variable.

    Returns:
        The API key value.

    Raises:
        ValueError: If the API key is not found in environment variables.
    """
    value = os.getenv(key_name)
    if not value:
        raise ValueError(f"{key_name} not found in environment variables.")
    return value
