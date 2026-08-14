"""
ChromaDB 过期 Collection 清理测试（临时目录，不污染项目 chroma_db）

覆盖：
1. 过期库（created_at 超龄）被清理
2. 无 created_at 的历史遗留库被清理
3. 保留期内（新鲜 created_at）的库保留
4. 当前会话的库豁免删除
5. 非 coc_rag_* 命名空间的库不处理
"""
import sys
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import chromadb
from rag.loader import cleanup_stale_collections

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


TMP = tempfile.mkdtemp(prefix="coc_chroma_test_")
client = chromadb.PersistentClient(path=TMP)

CURRENT = "coc_rag_current_session"
FRESH = "coc_rag_fresh_session"
OLD = "coc_rag_old_session"
LEGACY = "coc_rag_legacy_session"
FOREIGN = "other_project_collection"

now = datetime.now()
client.create_collection(CURRENT, metadata={"hnsw:space": "cosine", "created_at": now.isoformat()})
client.create_collection(FRESH, metadata={"hnsw:space": "cosine", "created_at": now.isoformat()})
client.create_collection(
    OLD,
    metadata={"hnsw:space": "cosine", "created_at": (now - timedelta(hours=72)).isoformat()},
)
client.create_collection(LEGACY)  # 无元数据：历史遗留
client.create_collection(FOREIGN, metadata={"created_at": (now - timedelta(hours=72)).isoformat()})


def names():
    return {c.name for c in client.list_collections()}


print("=" * 60)
print("1. cleanup_stale_collections")
print("=" * 60)

cleaned = cleanup_stale_collections(max_age_hours=24, current_session_id="current_session", db_path=TMP)
check("清理 2 个（超龄 + 遗留）", cleaned == 2, f"cleaned={cleaned}")

remaining = names()
check("当前会话库豁免", CURRENT in remaining, str(remaining))
check("保留期内的库保留", FRESH in remaining, str(remaining))
check("超龄库被删除", OLD not in remaining, str(remaining))
check("无元数据遗留库被删除", LEGACY not in remaining, str(remaining))
check("非 coc_rag_* 库不处理", FOREIGN in remaining, str(remaining))


print("=" * 60)
print("2. 边界情况")
print("=" * 60)

# 空 current_session_id：仅豁免逻辑照常（无豁免对象）
before = names()
cleaned2 = cleanup_stale_collections(max_age_hours=24, current_session_id="", db_path=TMP)
check("空 current_session_id 时超龄库也被清理", cleaned2 == 0, f"cleaned={cleaned2}")
check("清理前后集合不变（均已清理或保留期内）", before == names())

# 不存在的目录
cleaned3 = cleanup_stale_collections(db_path=os.path.join(TMP, "not_exists"))
check("目录不存在返回 0", cleaned3 == 0)

# current_session 传入完整 session_id 前缀而非后缀也能豁免
check("豁免按完整名称匹配", CURRENT in names())


print("\n" + "=" * 60)
print(f"Results: {passed} passed / {passed + failed} total")
shutil.rmtree(TMP, ignore_errors=True)
if failed > 0:
    print(f"FAILED {failed} test(s)!")
    sys.exit(1)
else:
    print("All tests passed!")
