"""
rag 包懒加载证明测试：rag.loader / rag.retriever 的顶层导入
不得依赖 chromadb / sentence_transformers 等重依赖。

背景：Streamlit Cloud 上报 ImportError（from rag.loader import ...），
根因之一是 RAG 重依赖（chromadb / torch）在模块顶层导入，
任何版本/平台不匹配都会拖垮 app 主流程。本测试通过阻断重依赖
验证顶层导入的轻量化，防止回归。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


print("=" * 60)
print("1. 阻断重依赖后 rag 包顶层导入")
print("=" * 60)

# 置 None 使 import chromadb / sentence_transformers 抛 ImportError
sys.modules["chromadb"] = None
sys.modules["sentence_transformers"] = None

try:
    import rag.loader  # noqa: F401
    check("rag.loader 顶层导入成功（无 chromadb/torch 依赖）", True)
except ImportError as e:
    check("rag.loader 顶层导入成功（无 chromadb/torch 依赖）", False, str(e))

try:
    import rag.retriever  # noqa: F401
    check("rag.retriever 顶层导入成功", True)
except ImportError as e:
    check("rag.retriever 顶层导入成功", False, str(e))

# 清理阻断，验证函数内部懒加载按预期工作
del sys.modules["chromadb"]
del sys.modules["sentence_transformers"]


print("=" * 60)
print("2. 恢复依赖后功能完整")
print("=" * 60)

from rag.loader import create_chroma_collection, cleanup_stale_collections  # noqa: E402

coll = create_chroma_collection("lazy_import_test")
check("懒加载下可正常创建 Collection", coll.name == "coc_rag_lazy_import_test",
      getattr(coll, "name", None))
check("Collection 空间为 cosine", coll.metadata.get("hnsw:space") == "cosine",
      str(coll.metadata))

# 清理测试库
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("coc_rag_lazy_import_test")
except Exception:
    pass


print("\n" + "=" * 60)
print(f"Results: {passed} passed / {passed + failed} total")
if failed > 0:
    print(f"FAILED {failed} test(s)!")
    sys.exit(1)
else:
    print("All tests passed!")
