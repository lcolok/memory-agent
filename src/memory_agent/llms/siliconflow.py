"""SiliconFlow LLM implementation."""

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.language_models.base import LanguageModelInput
import typing
from langchain_openai import ChatOpenAI

from memory_agent.env import load_api_key


class SiliconFlow:
    """SiliconFlow chat model."""

    @staticmethod
    def create(**kwargs) -> ChatOpenAI:
        """Create a SiliconFlow chat model.

        Args:
            **kwargs: Additional arguments to pass to the model.

        Returns:
            ChatOpenAI: The initialized chat model.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            base_url = load_api_key("SILICONFLOW_BASE_URL")
            api_key = load_api_key("SILICONFLOW_API_KEY")
            return ChatOpenAI(
                base_url=f"{base_url}/v1",
                api_key=api_key,
                **kwargs,
            )
        except ValueError as e:
            raise RuntimeError(f"Failed to initialize SiliconFlow model: {str(e)}")
