"""
MemoryManager 概括容错测试（纯逻辑，Fake LLM，不调用真实 API）

覆盖：
1. 概括失败：返回 False、缓冲区保留、落盘备份
2. 重试成功：更大范围合并概括、缓冲区清空、备份清理、rounds_range 连续
3. 进程重启恢复：备份恢复进待概括队列、round_counter 对齐不错位
4. _load_summaries 正确跳过 failed_backup 文件
5. 连续多次失败：备份覆盖更新、缓冲区持续累积
"""
import sys
import os
import json
import shutil
import tempfile
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.memory import MemoryManager

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


class FakeLLM:
    """fake OpenAI 客户端：fail=True 时模拟 API 波动。"""
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise ConnectionError("API 波动，连接超时")
        content = ('{"summary": "玩家在图书馆调查，发现了失踪教授留下的手稿，'
                   '并遭遇了书架后方的诡异声响。", "key_events": ["发现手稿"]}')
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


TMP = tempfile.mkdtemp(prefix="coc_mem_test_")


def list_files(session_dir):
    return sorted(f for f in os.listdir(session_dir))


def find_backup(session_dir):
    return [f for f in list_files(session_dir) if f.startswith("pending_rounds_")]


# ================================================================
# 1. 概括失败：不丢数据
# ================================================================
print("=" * 60)
print("1. summarize failure: no data loss")
print("=" * 60)

sid1 = "test-fail-001"
mm = MemoryManager(session_id=sid1, base_dir=TMP)
for i in range(6):
    mm.add_round(f"行动{i + 1}", f"回复{i + 1}")

ok = mm.summarize(FakeLLM(fail=True))
check("失败返回 False", ok is False)
check("失败后缓冲区保留 6 轮", len(mm.pending_rounds) == 6, f"len={len(mm.pending_rounds)}")
check("失败后 round_counter 不变", mm.round_counter == 6, f"counter={mm.round_counter}")

backups = find_backup(mm.memory_dir)
check("备份文件已落盘", len(backups) == 1, f"backups={backups}")
if backups:
    with open(os.path.join(mm.memory_dir, backups[0]), "r", encoding="utf-8") as f:
        backup_data = json.load(f)
    check("备份类型为 failed_backup", backup_data.get("type") == "failed_backup")
    check("备份包含 6 轮数据", len(backup_data.get("rounds", [])) == 6,
          f"rounds={len(backup_data.get('rounds', []))}")


# ================================================================
# 2. 重试成功：范围合并 + 清理
# ================================================================
print("=" * 60)
print("2. retry success: merged range + cleanup")
print("=" * 60)

# 失败后继续 2 轮，达到阈值再次触发
mm.add_round("行动7", "回复7")
mm.add_round("行动8", "回复8")
check("重试前缓冲区 8 轮", len(mm.pending_rounds) == 8, f"len={len(mm.pending_rounds)}")

ok = mm.summarize(FakeLLM(fail=False))
check("重试成功返回 True", ok is True)
check("成功后缓冲区清空", len(mm.pending_rounds) == 0, f"len={len(mm.pending_rounds)}")

individual = [s for s in mm.summaries if s.get("type") != "meta_summary"]
check("生成 1 条摘要", len(individual) == 1, f"count={len(individual)}")
if individual:
    check("rounds_range 连续覆盖 1-8", individual[0].get("rounds_range") == "1-8",
          individual[0].get("rounds_range"))
check("备份文件已清理", len(find_backup(mm.memory_dir)) == 0,
      f"backups={find_backup(mm.memory_dir)}")
check("摘要文件存在", any(f.startswith("summary_") for f in list_files(mm.memory_dir)),
      str(list_files(mm.memory_dir)))

# 后续轮次编号连续
mm.add_round("行动9", "回复9")
check("后续轮次从 9 继续（不错位）", mm.round_counter == 9, f"counter={mm.round_counter}")


# ================================================================
# 3. 进程重启恢复：备份加载 + round_counter 对齐
# ================================================================
print("=" * 60)
print("3. restart recovery from backup")
print("=" * 60)

sid2 = "test-restart-002"
mm_a = MemoryManager(session_id=sid2, base_dir=TMP)
for i in range(6):
    mm_a.add_round(f"行动{i + 1}", f"回复{i + 1}")
mm_a.summarize(FakeLLM(fail=True))  # 模拟进程在概括失败后崩溃

# 模拟重启：同一目录新建实例
mm_b = MemoryManager(session_id=sid2, base_dir=TMP)
check("重启后恢复 6 轮到待概括队列", len(mm_b.pending_rounds) == 6,
      f"len={len(mm_b.pending_rounds)}")
check("round_counter 对齐到 6", mm_b.round_counter == 6, f"counter={mm_b.round_counter}")
check("备份文件在恢复后删除", len(find_backup(mm_b.memory_dir)) == 0,
      f"backups={find_backup(mm_b.memory_dir)}")

# 恢复的轮次内容完整
rounds = {r["round_num"]: r for r in mm_b.pending_rounds}
check("恢复的第 1 轮内容完整", rounds.get(1, {}).get("user_action") == "行动1",
      str(rounds.get(1)))
check("恢复的第 6 轮内容完整", rounds.get(6, {}).get("user_action") == "行动6",
      str(rounds.get(6)))

# 下一轮从 7 继续，且成功概括后范围连续
mm_b.add_round("行动7", "回复7")
check("重启后新轮次从 7 继续", mm_b.round_counter == 7, f"counter={mm_b.round_counter}")
ok = mm_b.summarize(FakeLLM(fail=False))
check("恢复后概括成功", ok is True)
individual_b = [s for s in mm_b.summaries if s.get("type") != "meta_summary"]
if individual_b:
    check("恢复后摘要范围 1-7", individual_b[0].get("rounds_range") == "1-7",
          individual_b[0].get("rounds_range"))


# ================================================================
# 4. 连续多次失败：备份覆盖 + 缓冲区累积
# ================================================================
print("=" * 60)
print("4. repeated failures")
print("=" * 60)

sid3 = "test-repeat-003"
mm_c = MemoryManager(session_id=sid3, base_dir=TMP)
for i in range(6):
    mm_c.add_round(f"行动{i + 1}", f"回复{i + 1}")

mm_c.summarize(FakeLLM(fail=True))
first_backup = find_backup(mm_c.memory_dir)
mm_c.summarize(FakeLLM(fail=True))
second_backup = find_backup(mm_c.memory_dir)
check("重复失败备份文件仍存在", len(second_backup) == 1, f"backups={second_backup}")
check("缓冲区不因失败丢失", len(mm_c.pending_rounds) == 6, f"len={len(mm_c.pending_rounds)}")

# 范围扩大后失败 → 生成新备份（覆盖旧范围文件）
for i in range(3):
    mm_c.add_round(f"追加行动{i + 1}", f"追加回复{i + 1}")
mm_c.summarize(FakeLLM(fail=True))
backups3 = find_backup(mm_c.memory_dir)
check("范围扩大后失败生成新备份", any("0009" in b for b in backups3), f"backups={backups3}")


# ================================================================
# 5. _load_summaries 跳过 failed_backup（类型过滤）
# ================================================================
print("=" * 60)
print("5. failed_backup excluded from summaries")
print("=" * 60)

sid4 = "test-skip-004"
mm_d = MemoryManager(session_id=sid4, base_dir=TMP)
for i in range(6):
    mm_d.add_round(f"行动{i + 1}", f"回复{i + 1}")
mm_d.summarize(FakeLLM(fail=True))   # 留下备份文件
mm_d.summarize(FakeLLM(fail=False))  # 成功后清理
mm_d.summarize(FakeLLM(fail=True)) if False else None

# 手工放一个备份文件 + 正常摘要文件，验证加载不混淆
mm_e = MemoryManager(session_id="test-skip-005", base_dir=TMP)
for i in range(6):
    mm_e.add_round(f"行动{i + 1}", f"回复{i + 1}")
mm_e.summarize(FakeLLM(fail=False))  # 1 条正常摘要
mm_e.add_round("行动7", "回复7")
mm_e.summarize(FakeLLM(fail=True))   # 备份文件存在

mm_f = MemoryManager(session_id="test-skip-005", base_dir=TMP)
individual = [s for s in mm_f.summaries if s.get("type") != "meta_summary"]
check("摘要加载数正确（备份不计入）", len(individual) == 1, f"count={len(individual)}")
check("备份轮次恢复进队列", len(mm_f.pending_rounds) == 1, f"len={len(mm_f.pending_rounds)}")
check("round_counter 对齐到备份轮次", mm_f.round_counter == 7, f"counter={mm_f.round_counter}")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print(f"Results: {passed} passed / {passed + failed} total")
shutil.rmtree(TMP, ignore_errors=True)
if failed > 0:
    print(f"FAILED {failed} test(s)!")
    sys.exit(1)
else:
    print("All tests passed!")
