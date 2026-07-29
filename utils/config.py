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
RAG_SIMILARITY_THRESHOLD = 0.35  # RAG 相似度阈值（bge-small-zh-v1.5 中文模型相似度更准确）

# ---------- API 超时配置 ----------
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "25"))   # API 单次调用超时（Flash 模型通常 <10s）
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "0"))            # 不重试——快速失败比长时间等待更好

# ---------- 模型配置 ----------
# KP 主力模型（叙事 + 检定判断，需要强创造力）—— 默认 Flash 以加快响应
KP_MODEL = os.getenv("KP_MODEL", "deepseek-v4-flash")
# Render 模型（环境润色，较简单任务）—— 始终用 Flash
RENDER_MODEL = os.getenv("RENDER_MODEL", "deepseek-v4-flash")

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
    embedding_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/bge-small-zh-v1.5")

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
    设置超时防止请求无限挂起。
    """
    import httpx

    cfg = get_config()
    timeout = httpx.Timeout(
        connect=10.0,          # 连接超时
        read=API_TIMEOUT_SECONDS,   # 读取超时
        write=10.0,            # 写入超时
        pool=5.0,              # 连接池超时
    )
    return OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=timeout,
        max_retries=0,  # 我们自己处理重试
    )
