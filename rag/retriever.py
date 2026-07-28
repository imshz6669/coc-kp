"""
RAG 检索器模块 —— 将玩家输入向量化，检索 Top K 相似段落。

提供：
    - retrieve_context() : 检索与玩家输入最相关的知识库片段
"""

from typing import List

import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)


class _NoOpEmbedding(EmbeddingFunction):
    """空 embedding，阻止 ChromaDB 自动下载模型。"""
    def __init__(self):
        pass

    def __call__(self, input):
        return [[0.0]] * len(input)

from utils.config import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD


def retrieve_context(
    query: str,
    session_id: str,
    model: SentenceTransformer,
    top_k: int = RAG_TOP_K,
    threshold: float = RAG_SIMILARITY_THRESHOLD,
) -> str:
    """
    检索与查询最相关的知识库上下文。

    参数：
        query      : 查询文本（玩家输入）
        session_id : 会话 ID（用于定位 Collection）
        model      : embedding 模型实例
        top_k      : 返回的最大段落数
        threshold  : 相似度阈值，低于此值的段落被丢弃

    返回：
        拼接后的上下文字符串。若无匹配或 Collection 不存在，返回空字符串。
    """
    if not query or not session_id:
        return ""

    collection_name = f"coc_rag_{session_id}"

    # 尝试获取 Collection
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(collection_name)
    except Exception:
        logger.info(f"Collection '{collection_name}' 不存在或无知识库，跳过 RAG 检索。")
        return ""

    # 获取 Collection 中的文档数
    try:
        count = collection.count()
        if count == 0:
            logger.info("Collection 为空，跳过检索。")
            return ""
    except Exception:
        return ""

    # 向量化查询
    try:
        query_embedding = model.encode([query]).tolist()
    except Exception as e:
        logger.error(f"查询向量化失败: {e}")
        return ""

    # 检索
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
        )
    except Exception as e:
        logger.error(f"ChromaDB 检索失败: {e}")
        return ""

    if not results or not results.get("documents") or not results["documents"][0]:
        logger.info("未检索到相关段落。")
        return ""

    documents = results["documents"][0]
    distances = results.get("distances", [[]])[0]

    # 根据相似度阈值过滤
    # ChromaDB 返回的距离因后端不同而异：
    # - Rust 后端 (PersistentClient): L2 距离 (0~2, 越小越相似)
    # - 旧后端 (duckdb+parquet): 余弦距离 (0~1, 越小越相似)
    relevant_parts = []
    for i, doc in enumerate(documents):
        dist = distances[i] if i < len(distances) else 0.0

        # 统一转为相似度：L2 距离用 1 - dist/2 近似余弦相似度
        if dist > 1.0:
            similarity = max(0.0, 1.0 - dist / 2.0)
        else:
            similarity = 1.0 - dist

        if similarity >= threshold:
            relevant_parts.append(doc)
            logger.info(f"RAG 命中 #{i}: dist={dist:.3f} similarity={similarity:.3f}")

    if not relevant_parts:
        logger.info(f"所有段落相似度均低于阈值 {threshold}，已丢弃。")
        return ""

    context = "\n---\n".join(relevant_parts)
    logger.info(f"RAG 检索完成: {len(relevant_parts)}/{top_k} 段落命中。")
    return context
