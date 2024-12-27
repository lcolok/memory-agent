"""Integration tests for the memory graph."""

import logging
import os
from typing import List

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from memory_agent.graph import builder
from memory_agent.configuration import Configuration

# Test configuration
TEST_CONFIG = {
    "user_id": "test-user",
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "embedding_model": "netease-youdao/bce-embedding-base_v1",
    "system_prompt": """你是一个智能助手。当前时间是: {time}。

以下是关于用户的一些记忆：
{user_info}

请根据这些信息，以友好和自然的方式与用户交谈。如果你发现了新的重要信息，请使用 upsert_memory 工具来存储它。""",
}

# API configuration
TEST_API_CONFIG = {
    "SILICONFLOW_API_KEY": "sk-tilrofxsioowlrlmylzwvjbghbluqsfodorcovrwilrfbead",
    "SILICONFLOW_BASE_URL": "https://api.siliconflow.cn",
}


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    # 设置环境变量
    monkeypatch.setenv("SILICONFLOW_API_KEY", TEST_API_CONFIG["SILICONFLOW_API_KEY"])
    monkeypatch.setenv("SILICONFLOW_BASE_URL", TEST_API_CONFIG["SILICONFLOW_BASE_URL"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation",
    [
        ["你好，我叫小明，我特别喜欢吃火锅。请记住这一点。"],
        [
            "你好，我是小红，我平时喜欢打羽毛球。请记住这一点。",
            "对了，我还养了一只叫旺财的狗。",
            "旺财是一只金毛，今年5岁了。也请记住这些信息。",
        ],
    ],
    ids=["短对话", "中等对话"],
)
async def test_memory_storage(mock_env, conversation: List[str], caplog):
    """测试记忆存储功能。

    这个测试用例模拟了不同场景下的对话，验证系统是否能够正确存储和记忆对话内容。

    Args:
        mock_env: pytest fixture，用于模拟环境变量
        conversation: 对话列表，每个元素是一句话
        caplog: pytest的日志捕获fixture
    """
    # 设置日志级别为INFO
    caplog.set_level(logging.INFO)

    # 初始化图
    config = Configuration(**TEST_CONFIG)
    store = InMemoryStore()
    graph = builder.compile(store=store)

    print("\n=== 开始新的对话 ===\n")

    # 运行对话
    state = {"messages": []}
    for user_input in conversation:
        print(f"用户: {user_input}")
        state["messages"].append(HumanMessage(content=user_input))
        state = await graph.ainvoke(
            state,
            {"configurable": {
                "user_id": config.user_id,
                "model": config.model,
                "embedding_model": config.embedding_model,
                "system_prompt": config.system_prompt,
            }},
        )
        for message in state["messages"][-2:]:  # 只打印最后两条消息
            if isinstance(message, AIMessage):
                print(f"AI: {message.content}")

    # 检查存储的记忆
    memories = store.search(("memories", config.user_id), limit=10)
    assert len(memories) > 0, "没有存储任何记忆"

    # 打印存储的记忆
    print("\n=== 存储的记忆 ===\n")
    for memory in memories:
        print(f"[{memory.key}]: {memory.value} (similarity: {memory.score})")
