"""
配置管理模块 —— 从 .env 读取配置，统一封装模型客户端。

提供：
    - get_config()       : 获取所有配置的字典
    - get_openai_client(): 返回 OpenAI 兼容的 Client 实例
    - 配置常量           : MAX_CONTEXT_ROUNDS, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件 —— 优先从项目根目录查找
load_dotenv()

# ---------- 配置常量 ----------

MAX_CONTEXT_ROUNDS = 5          # 保留最近 N 轮对话
RAG_TOP_K = 3                   # RAG 检索 Top K 段落
RAG_SIMILARITY_THRESHOLD = 0.15  # RAG 相似度阈值（all-MiniLM-L6-v2 对中文的余弦相似度偏低，调低阈值）

# ---------- 记忆概括配置 ----------
MEMORY_SUMMARIZE_INTERVAL = 6        # 每 N 轮对话触发一次概括
MEMORY_DIR = "./data/memory"         # 记忆文件存储根目录
MEMORY_MAX_SUMMARIES_BEFORE_META = 5 # 触发元概括的个体摘要数阈值


def get_config() -> dict:
    """
    读取所有配置项，返回字典。
    若缺少必要环境变量，抛出明确的 RuntimeError。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    embedding_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/all-MiniLM-L6-v2")

    if not api_key:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY 环境变量！"
            "请将 .env.example 复制为 .env，并填入你的 API 密钥。"
        )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "embedding_model_path": embedding_path,
    }


def get_openai_client() -> OpenAI:
    """
    返回 OpenAI 兼容的 Client 实例，用于调用 DeepSeek API。
    """
    cfg = get_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
