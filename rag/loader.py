"""
RAG 文件解析与入库模块 —— 将文档分块向量化后存入 ChromaDB。

支持格式：TXT、PDF（PyPDF2）
分块策略：500 字符，重叠 50 字符
向量化模型：sentence-transformers/all-MiniLM-L6-v2
Collection 命名：coc_rag_{session_id}
"""

import os
import uuid
from typing import List

import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


# ===================== 空 Embedding 函数 =====================

class _NoOpEmbedding(EmbeddingFunction):
    """
    空操作 embedding 函数 —— 阻止 ChromaDB 自动下载 ONNX 模型。

    本项目使用 sentence-transformers 独立计算向量，
    所有向量通过 collection.add(embeddings=...) 显式传入，
    ChromaDB 的内置 embedding 从未被使用。
    """
    def __init__(self):
        pass

    def __call__(self, input):
        return [[0.0]] * len(input)


# ===================== 文本解析 =====================

def parse_file(file_path: str) -> str:
    """
    解析上传的文件，提取纯文本内容。

    参数：
        file_path : 文件路径

    返回：
        提取的文本内容。

    异常：
        ValueError : 文件格式不支持
        IOError    : 文件读取失败
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return _parse_txt(file_path)
    elif ext == ".pdf":
        return _parse_pdf(file_path)
    else:
        raise ValueError(
            f"不支持的文件格式: {ext}。RAG 知识库仅支持 TXT 和 PDF 文件。"
        )


def _parse_txt(file_path: str) -> str:
    """
    解析 TXT 文件。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # 回退到 GBK 编码
        with open(file_path, "r", encoding="gbk") as f:
            return f.read()
    except Exception as e:
        raise IOError(f"读取 TXT 文件失败 ({file_path}): {e}")


def _parse_pdf(file_path: str) -> str:
    """
    使用 PyPDF2 解析 PDF 文件。
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PyPDF2 未安装，无法解析 PDF 文件。请执行: pip install PyPDF2>=3.0.0"
        )

    try:
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        raise IOError(f"读取 PDF 文件失败 ({file_path}): {e}")


# ===================== 文本分块 =====================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将长文本拆分为重叠的分块。

    参数：
        text       : 原始文本
        chunk_size : 每块最大字符数（默认 500）
        overlap    : 块间重叠字符数（默认 50）

    返回：
        分块列表。
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())

        if end >= text_len:
            break
        start = end - overlap

    logger.info(f"文本分块完成: {len(chunks)} 块 (chunk_size={chunk_size}, overlap={overlap})")
    return chunks


# ===================== 向量化与入库 =====================

def create_embedding_model() -> SentenceTransformer:
    """
    加载 embedding 模型。

    返回：
        SentenceTransformer 实例。
    """
    cfg = get_config()
    model_path = cfg.get("embedding_model_path", "all-MiniLM-L6-v2")

    try:
        model = SentenceTransformer(model_path)
        logger.info(f"Embedding 模型加载成功: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Embedding 模型加载失败 ({model_path}): {e}")
        raise RuntimeError(
            f"无法加载 embedding 模型 '{model_path}'。"
            f"请确认模型已下载到本地，或网络可访问 HuggingFace。\n"
            f"错误详情: {e}"
        )


def create_chroma_collection(session_id: str) -> chromadb.Collection:
    """
    为指定会话创建/获取 ChromaDB Collection。

    命名规则：coc_rag_{session_id}

    参数：
        session_id : 会话唯一标识（UUID 字符串）

    返回：
        ChromaDB Collection 实例。
    """
    collection_name = f"coc_rag_{session_id}"

    try:
        client = chromadb.PersistentClient(path="./chroma_db")

        # 删除同名旧 Collection（如果存在）
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            embedding_function=_NoOpEmbedding(),
        )
        logger.info(f"ChromaDB Collection 创建成功: {collection_name}")
        return collection

    except Exception as e:
        logger.error(f"ChromaDB Collection 创建失败: {e}")
        raise RuntimeError(f"无法创建 ChromaDB Collection: {e}")


def index_file(
    file_path: str,
    session_id: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    """
    解析文件、分块、向量化并存入 ChromaDB。

    参数：
        file_path  : 文件路径
        session_id : 会话 ID
        model      : embedding 模型
        collection : ChromaDB Collection

    返回：
        成功入库的块数。
    """
    try:
        text = parse_file(file_path)
    except Exception as e:
        logger.error(f"文件解析失败: {e}")
        raise

    chunks = chunk_text(text)

    if not chunks:
        logger.warning(f"文件内容为空或无有效文本: {file_path}")
        return 0

    # 向量化
    try:
        embeddings = model.encode(chunks).tolist()
    except Exception as e:
        logger.error(f"向量化失败: {e}")
        raise RuntimeError(f"文本向量化失败: {e}")

    # 入库
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(file_path), "index": i} for i in range(len(chunks))]

    try:
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"入库完成: {len(chunks)} 块 → Collection '{collection.name}'")
        return len(chunks)
    except Exception as e:
        logger.error(f"入库失败: {e}")
        raise RuntimeError(f"ChromaDB 写入失败: {e}")


def clear_collection(session_id: str) -> bool:
    """
    删除指定会话的 ChromaDB Collection。

    返回：
        是否成功删除。
    """
    collection_name = f"coc_rag_{session_id}"

    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        client.delete_collection(collection_name)
        logger.info(f"Collection 已删除: {collection_name}")
        return True
    except Exception as e:
        logger.warning(f"删除 Collection 失败（可能不存在）: {e}")
        return False
