"""Define the configurable parameters for the agent."""

import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Optional, Iterator, Tuple

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="siliconflow/Qwen/Qwen2.5-72B-Instruct",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    embedding_model: str = "netease-youdao/bce-embedding-base_v1"
    """The embedding model to use for semantic search."""
    system_prompt: str = prompts.SYSTEM_PROMPT

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }

        return cls(**{k: v for k, v in values.items() if v})

    def items(self) -> Iterator[Tuple[str, dict]]:
        """Convert the configuration to a dictionary."""
        yield "configurable", asdict(self)
