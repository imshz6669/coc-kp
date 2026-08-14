"""
RAG 相似度转换 + 生成文本规范化测试（纯逻辑，不调用 LLM API）

覆盖：
1. _dist_to_similarity：dist≈1 附近连续无跳变、全程单调、[0,1] 钳制
2. 阈值判定稳定性：dist 在 1.0 附近微抖不再导致命中翻转
3. _normalize_generated_text：
   - em-dash / en-dash 替换
   - 未完成句尾裁剪到最后一个完整句 + "……"
   - 完整句保持原样
   - 句末闭合引号保留
   - 无句界文本保持原样
4. _call_with_retry：finish_reason=length 时翻倍 max_tokens 重试
"""
import sys
import os
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag.retriever import _dist_to_similarity
from backend.agents import _normalize_generated_text, _normalize_kp_result, _call_with_retry
import backend.agents as agents_module

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


# ================================================================
# 1. 相似度公式：连续性
# ================================================================
print("=" * 60)
print("1. _dist_to_similarity: continuity")
print("=" * 60)

# dist=1.0 两侧微抖：旧公式 0.01 ↔ 0.495 跳变，新公式应为小步连续
s_below = _dist_to_similarity(0.99)
s_at = _dist_to_similarity(1.0)
s_above = _dist_to_similarity(1.01)
check("dist=0.99 与 dist=1.01 差异 < 0.05", abs(s_below - s_above) < 0.05,
      f"s(0.99)={s_below:.4f}, s(1.01)={s_above:.4f}")
check("dist=1.0 处与两侧连续", abs(s_at - s_below) < 0.05 and abs(s_at - s_above) < 0.05,
      f"s(1.0)={s_at:.4f}")

# 边界值
check("dist=0 → 1.0", abs(_dist_to_similarity(0.0) - 1.0) < 1e-9)
check("dist=2 → 0.0 附近", abs(_dist_to_similarity(2.0)) < 1e-9)
check("dist=0.5 → 0.875", abs(_dist_to_similarity(0.5) - 0.875) < 1e-9)

# 单调性
dists = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
sims = [_dist_to_similarity(d) for d in dists]
monotonic = all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))
check("全程单调递减", monotonic, f"sims={[round(s, 3) for s in sims]}")

# 钳制
check("dist=3.0 钳制到 0", _dist_to_similarity(3.0) == 0.0)
check("dist 负值钳制到 1", _dist_to_similarity(-0.5) == 1.0)

# cosine 空间（新 Collection）：sim = 1 - dist，余弦距离 ∈ [0, 2]
check("cosine dist=0.3 → 0.7", abs(_dist_to_similarity(0.3, "cosine") - 0.7) < 1e-9)
check("cosine dist=0 → 1.0", abs(_dist_to_similarity(0.0, "cosine") - 1.0) < 1e-9)
check("cosine dist=1 → 0.0", abs(_dist_to_similarity(1.0, "cosine")) < 1e-9)
check("cosine dist=2 钳制到 0", _dist_to_similarity(2.0, "cosine") == 0.0)
check("cosine dist 负值钳制到 1", _dist_to_similarity(-0.1, "cosine") == 1.0)
cos_sims = [_dist_to_similarity(d, "cosine") for d in [0.0, 0.5, 1.0, 1.5, 2.0]]
check("cosine 空间单调", all(cos_sims[i] >= cos_sims[i + 1] for i in range(len(cos_sims) - 1)))


# ================================================================
# 2. 阈值判定稳定性（0.35 阈值在 dist≈1 微抖下不再翻转）
# ================================================================
print("=" * 60)
print("2. Threshold stability around dist=1")
print("=" * 60)

THRESHOLD = 0.35
for d in [0.99, 1.00, 1.01]:
    s = _dist_to_similarity(d)
    assert s >= THRESHOLD, f"dist={d} similarity={s:.3f} 低于阈值 {THRESHOLD}"
check("dist≈1 微抖时命中判定保持一致（均 ≥ 0.35）", True)


# ================================================================
# 3. _normalize_generated_text
# ================================================================
print("=" * 60)
print("3. _normalize_generated_text")
print("=" * 60)

# em-dash 替换
t = "指尖的记忆比视觉更早复现出那涡流的弧度——你画得很快"
n = _normalize_generated_text(t)
check("'——' 替换为 '，'", "——" not in n and "弧度，你画得很快" in n, n)

t2 = "他顿了顿—说—没什么"
n2 = _normalize_generated_text(t2)
check("单 '—' 替换为 '，'", "—" not in n2, n2)

t3 = "页码 12–15"
n3 = _normalize_generated_text(t3)
check("en-dash '–' 替换为 '-'", "-" in n3 and "–" not in n3, n3)

# 截断句尾裁剪
t4 = "你从口袋里翻出一截软芯铅笔。笔尖贴上纸面的瞬间，指尖的记忆比视觉更早复现。落笔完成时，纸"
n4 = _normalize_generated_text(t4)
check("半截句尾裁剪到最后一个完整句", n4.endswith("……") and "落笔完成时" not in n4, n4)
check("裁剪后保留前面的完整句子", "指尖的记忆比视觉更早复现。" in n4, n4)

# 完整句保持原样
t5 = "夜色笼罩着阿卡姆。你推开了那扇门。"
n5 = _normalize_generated_text(t5)
check("完整句保持原样", n5 == t5, n5)

# 以闭合引号结尾视为完整
t6 = "他说：「别让他们找到它。」"
n6 = _normalize_generated_text(t6)
check("闭合引号结尾不裁剪", n6 == t6, n6)

# 无句界文本保持原样
t7 = "这是没有句号的一句话"
n7 = _normalize_generated_text(t7)
check("无句界文本保持原样", n7 == t7, n7)

# 空文本
check("空文本安全", _normalize_generated_text("") == "")

# _normalize_kp_result 覆盖全部文本字段
r = _normalize_kp_result({
    "kp_response": "你撞开了门——门后",
    "narrative": "门后是一条漆黑的走廊。你走了进去，忽然",
    "scene": "旧宅",
})
check("kp_result 字段规范化", "——" not in r["kp_response"] and r["narrative"].endswith("……"), str(r))


# ================================================================
# 4. _call_with_retry：finish_reason=length 翻倍重试
# ================================================================
print("=" * 60)
print("4. _call_with_retry: length truncation retry")
print("=" * 60)

class FakeClient:
    """记录每次调用的 max_tokens，按剧本返回 finish_reason。"""
    def __init__(self, reasons, contents):
        self.reasons = list(reasons)   # 每次调用的 finish_reason
        self.contents = list(contents) # 每次调用返回的内容
        self.calls = []                # 记录 (max_tokens, ...)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        reason = self.reasons[min(idx, len(self.reasons) - 1)]
        content = self.contents[min(idx, len(self.contents) - 1)]
        msg = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=msg, finish_reason=reason)
        resp = SimpleNamespace(choices=[choice])
        return resp

orig_get_client = agents_module._get_client

# 场景 A：1024 起步，第一次截断 → 2048 重试一次，再截断 → 直接返回
# （整段重新生成代价大，交由上游句界裁剪，避免几十秒延迟）
fake = FakeClient(["length", "length", "stop"], ["截断内容A", "截断内容B", "完整内容"])
agents_module._get_client = lambda: fake
result = _call_with_retry(messages=[], model="test", temperature=0.7,
                          max_tokens=1024, description="fake")
check("第二次截断直接返回（仅一次翻倍重试）", result == "截断内容B", result)
check("调用 2 次", len(fake.calls) == 2, f"calls={len(fake.calls)}")
check("max_tokens 1024 → 2048",
      [c["max_tokens"] for c in fake.calls] == [1024, 2048],
      str([c["max_tokens"] for c in fake.calls]))

# 场景 B：512 起步，两次翻倍后达到 2048 上限直接返回，不抛异常
fake2 = FakeClient(["length"] * 4, ["截断内容"] * 4)
agents_module._get_client = lambda: fake2
result2 = _call_with_retry(messages=[], model="test", temperature=0.7,
                           max_tokens=512, description="fake")
check("全截断时不抛异常", result2 == "截断内容", result2)
check("max_tokens 512 → 1024 → 2048",
      [c["max_tokens"] for c in fake2.calls] == [512, 1024, 2048],
      str([c["max_tokens"] for c in fake2.calls]))

# 场景 B2：2048 起步已到上限，截断直接返回（仅 1 次调用）
fake2b = FakeClient(["length"] * 4, ["截断内容"] * 4)
agents_module._get_client = lambda: fake2b
result2b = _call_with_retry(messages=[], model="test", temperature=0.7,
                            max_tokens=2048, description="fake")
check("已达上限不再重试（1 次调用）", result2b == "截断内容" and len(fake2b.calls) == 1,
      f"calls={len(fake2b.calls)}")

# 场景 C：正常 stop → 一次调用完成
fake3 = FakeClient(["stop"], ["正常内容"])
agents_module._get_client = lambda: fake3
result3 = _call_with_retry(messages=[], model="test", temperature=0.7,
                           max_tokens=1024, description="fake")
check("正常返回不重试", result3 == "正常内容" and len(fake3.calls) == 1)

agents_module._get_client = orig_get_client


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print(f"Results: {passed} passed / {passed + failed} total")
if failed > 0:
    print(f"FAILED {failed} test(s)!")
    sys.exit(1)
else:
    print("All tests passed!")
