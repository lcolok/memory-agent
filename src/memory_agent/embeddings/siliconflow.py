"""SiliconFlow embeddings API wrapper."""

import os
from typing import Dict, List

import requests
from langchain.embeddings.base import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow embeddings API."""

    def __init__(self, model: str = "netease-youdao/bce-embedding-base_v1"):
        """Initialize the embeddings API.

        Args:
            model: The model to use for embeddings.
        """
        self.model = model
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        base_url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn")
        self.base_url = f"{base_url}/v1"
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY environment variable not set")
        if not base_url:
            raise ValueError("SILICONFLOW_BASE_URL environment variable not set")

    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the API request.

        Returns:
            The headers for the API request.
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: The texts to get embeddings for.

        Returns:
            The embeddings for the texts.
        """
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._get_headers(),
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        return [data["embedding"] for data in response.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents.

        Args:
            texts: The texts to get embeddings for.

        Returns:
            The embeddings for the texts.
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Get embeddings for a query.

        Args:
            text: The text to get embeddings for.

        Returns:
            The embeddings for the text.
        """
        return self._get_embeddings([text])[0]
