"""Graphs that extract memories on a schedule."""

import asyncio
import json
import logging
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from memory_agent import configuration, tools, utils
from memory_agent.state import State
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


async def call_model(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    # 过滤掉不需要的配置参数
    config_dict = config["configurable"]
    filtered_config = {
        k: v for k, v in config_dict.items()
        if k in ["user_id", "model", "embedding_model", "system_prompt"]
    }
    configurable = configuration.Configuration(**filtered_config)

    # Initialize the models based on configuration
    model_config = utils.split_model_and_provider(configurable.model)
    llm = utils.initialize_chat_model(model_config)
    embeddings = utils.initialize_embedding_model(configurable.embedding_model)

    # Get the query text from recent messages
    query = str([m.content for m in state.messages[-3:]])

    # Calculate embeddings for the query
    query_embedding = await embeddings.aembed_query(query)

    # Retrieve the most recent memories for context
    memories = store.search(
        ("memories", configurable.user_id),
        query=query_embedding,
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories)
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    sys = configurable.system_prompt.format(
        user_info=formatted, time=datetime.now().isoformat()
    )

    # Invoke the language model with the prepared prompt and tools
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [SystemMessage(content=sys), *state.messages],
        {"configurable": model_config},
    )

    # 修复工具调用参数的格式
    if "tool_calls" in msg.additional_kwargs:
        for tc in msg.additional_kwargs["tool_calls"]:
            if tc["type"] == "function":
                tc["args"] = json.loads(tc["function"]["arguments"])

    return {"messages": [msg]}


async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    # Extract tool calls from the last message
    tool_calls = []
    for tc in state.messages[-1].additional_kwargs.get("tool_calls", []):
        if tc["type"] == "function" and tc["function"]["name"] == "upsert_memory":
            args = tc["args"]  # 不需要再次解析 JSON
            tool_calls.append({
                "id": tc["id"],
                "args": {
                    "content": args.get("content", args.get("value")),  # 兼容 value 字段
                    "context": args.get("key", args.get("context", "No context provided"))  # 兼容 key 字段
                }
            })

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(
            tools.upsert_memory(**tc["args"], config=config, store=store)
            for tc in tool_calls
        )
    )

    # Format the results of memory storage operations
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    # Check if the last message has any tool calls
    if state.messages[-1].additional_kwargs.get("tool_calls"):
        return "store_memory"
    return END


# Create the graph + all nodes
builder = StateGraph(State, config_schema=configuration.Configuration)

# Add the nodes
builder.add_node("call_model", call_model)
builder.add_node("store_memory", store_memory)

# Add the edges
builder.set_entry_point("call_model")
builder.add_conditional_edges("call_model", route_message)
builder.add_edge("store_memory", END)

graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]
