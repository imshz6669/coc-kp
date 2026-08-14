"""
HP / SAN 增减机制综合测试（纯逻辑，不调用 LLM API）

覆盖：
1. tool_node 链式执行：战斗伤害 → HP 扣减与重伤标记
2. tool_node 链式执行：理智损失 → SAN 扣减
3. tool_node 同时执行：检定 + 伤害 + 理智（角色状态连续传递）
4. route_after_kp：仅有伤害/理智损失（无属性检定）时也路由到 tool_node
5. render_node：内部标记之后的多条工具消息全部合并、标记清理
6. SAN 损失 ≥ 5 的智力检定：成功 → 临时疯狂；失败 → 压抑
7. SAN 归零 → 永久疯狂，不触发临时疯狂判定
8. HP 归零 → 死亡状态传播
9. _validate_kp_output：hp_damage/san_loss 字段安全转换（含字符串数字）
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.character import generate_random_character
from backend.graph import build_graph, create_initial_state, tool_node, route_after_kp, render_node
from backend.agents import _validate_kp_output
import backend.graph as graph_module
import backend.tools as tools_module

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


def make_state(char=None, pending=None, messages=None):
    """构造最小可用的 KeeperState。"""
    base = {
        "messages": messages or [],
        "character": char or generate_random_character(),
        "game_over": False,
        "pending_check": pending or {},
        "rag_context": "",
        "temp_status": {},
        "rendered_text": "",
        "memory_summary": "",
        "scene_context": "",
        "suggestions": [],
        "current_scene": "",
    }
    return base


# ================================================================
# 1. tool_node：战斗伤害
# ================================================================
print("=" * 60)
print("1. tool_node: combat damage")
print("=" * 60)

char = generate_random_character()
char["HP"] = 10
char["MAX_HP"] = 10
state = make_state(char=char, pending={
    "need_check": "None", "difficulty": "普通",
    "hp_damage": 4, "san_loss": 0, "damage_source": "邪教徒的匕首", "san_reason": "",
})
result = tool_node(state)
check("HP 10 -> 6", result["character"]["HP"] == 6, f"HP={result['character']['HP']}")
check("wounded=False (6/10 > 30%)", result["character"]["wounded"] is False)
check("status 仍 alive", result["character"]["status"] == "alive")
sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
check("生成 1 条 system 消息", len(sys_msgs) == 1, f"count={len(sys_msgs)}")
check("消息含伤害来源", "邪教徒的匕首" in sys_msgs[0]["content"])

# 重伤阈值：HP 降到 30% 以下
char2 = generate_random_character()
char2["HP"] = 10
char2["MAX_HP"] = 10
state2 = make_state(char=char2, pending={
    "need_check": "None", "difficulty": "普通",
    "hp_damage": 8, "san_loss": 0, "damage_source": "坠落", "san_reason": "",
})
result2 = tool_node(state2)
check("HP 10 -> 2", result2["character"]["HP"] == 2, f"HP={result2['character']['HP']}")
check("wounded=True (2/10 < 30%)", result2["character"]["wounded"] is True)

# 致死伤害
char3 = generate_random_character()
char3["HP"] = 5
char3["MAX_HP"] = 10
state3 = make_state(char=char3, pending={
    "need_check": "None", "difficulty": "普通",
    "hp_damage": 9, "san_loss": 0, "damage_source": "神话生物", "san_reason": "",
})
result3 = tool_node(state3)
check("HP 钳制到 0", result3["character"]["HP"] == 0)
check("status=dead", result3["character"]["status"] == "dead")


# ================================================================
# 2. tool_node：理智损失
# ================================================================
print("=" * 60)
print("2. tool_node: sanity loss")
print("=" * 60)

char = generate_random_character()
char["SAN"] = 50
char["MAX_SAN"] = 50
state = make_state(char=char, pending={
    "need_check": "None", "difficulty": "普通",
    "hp_damage": 0, "san_loss": 3, "damage_source": "", "san_reason": "目睹尸体",
})
result = tool_node(state)
check("SAN 50 -> 47", result["character"]["SAN"] == 47, f"SAN={result['character']['SAN']}")
check("temp_insanity=False (损失 < 5)", result["character"]["temp_insanity"] is False)
sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
check("生成 1 条 system 消息", len(sys_msgs) == 1)
check("消息含理智损失原因", "目睹尸体" in sys_msgs[0]["content"])


# ================================================================
# 3. tool_node：检定 + 伤害 + 理智 同时发生
# ================================================================
print("=" * 60)
print("3. tool_node: check + damage + sanity in one round")
print("=" * 60)

char = generate_random_character()
char["HP"] = 10
char["MAX_HP"] = 10
char["SAN"] = 50
char["MAX_SAN"] = 50
state = make_state(char=char, pending={
    "need_check": "敏捷", "difficulty": "困难",
    "hp_damage": 3, "san_loss": 6, "damage_source": "陷阱", "san_reason": "目睹仪式",
})
result = tool_node(state)
check("HP 10 -> 7", result["character"]["HP"] == 7, f"HP={result['character']['HP']}")
check("SAN 50 -> 44", result["character"]["SAN"] == 44, f"SAN={result['character']['SAN']}")
sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
check("生成 3 条 system 消息（检定/伤害/理智）", len(sys_msgs) == 3, f"count={len(sys_msgs)}")
check("第 1 条为检定结果", "检定" in sys_msgs[0]["content"])
check("第 2 条为伤害结果", "受到伤害" in sys_msgs[1]["content"])
check("第 3 条为理智结果", "理智损失" in sys_msgs[2]["content"])


# ================================================================
# 4. route_after_kp：仅有伤害/理智时也路由到 tool_node
# ================================================================
print("=" * 60)
print("4. route_after_kp")
print("=" * 60)

s1 = make_state(pending={"need_check": "None", "hp_damage": 3, "san_loss": 0})
check("纯伤害路由到 tool_node", route_after_kp(s1) == "tool_node")

s2 = make_state(pending={"need_check": "None", "hp_damage": 0, "san_loss": 4})
check("纯理智损失路由到 tool_node", route_after_kp(s2) == "tool_node")

s3 = make_state(pending={"need_check": "感知", "hp_damage": 0, "san_loss": 0})
check("属性检定路由到 tool_node", route_after_kp(s3) == "tool_node")

s4 = make_state(pending={"need_check": "None", "hp_damage": 0, "san_loss": 0})
check("无事件路由到 END", route_after_kp(s4) == "end")

s5 = make_state(pending={})
check("空 pending 路由到 END", route_after_kp(s5) == "end")


# ================================================================
# 5. render_node：多条工具消息合并 + 标记清理
# ================================================================
print("=" * 60)
print("5. render_node: tool messages merge")
print("=" * 60)

# 模拟 call_render（不调用 LLM）
graph_module.call_render = lambda narrative, check_result: narrative

char = generate_random_character()
char["HP"] = 7
char["MAX_HP"] = 10
char["SAN"] = 44
char["MAX_SAN"] = 50
msgs = [
    {"role": "user", "content": "我冲向祭坛"},
    {"role": "assistant", "content": "[KP回应] 你冲向祭坛，需要敏捷检定。"},
    {"role": "assistant", "content": "[KP梗概] 祭坛上的符文开始发光。"},
    {"role": "system", "content": "🎲 【敏捷检定 · 困难难度】\n属性值: 50 | 掷出: 40\n结果: 检定成功！"},
    {"role": "system", "content": "⚡ 【受到伤害 · 陷阱】\n损失 3 点 HP（10 → 7/10）"},
    {"role": "system", "content": "🧠 【理智损失 · 目睹仪式】\n😱 丧失 6 点 SAN（当前 SAN: 44/50）。"},
]
state = make_state(char=char, pending={}, messages=msgs)
result = render_node(state)

marker_msgs = [m for m in result["messages"] if "[KP回应]" in m.get("content", "") or "[KP梗概]" in m.get("content", "")]
check("内部标记已清理", len(marker_msgs) == 0, f"count={len(marker_msgs)}")

sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
check("合并为 1 条 system 消息", len(sys_msgs) == 1, f"count={len(sys_msgs)}")
merged = sys_msgs[0]["content"] if sys_msgs else ""
check("合并消息含检定结果", "敏捷检定" in merged)
check("合并消息含伤害结果", "受到伤害" in merged)
check("合并消息含理智结果", "理智损失" in merged)
check("合并消息含 KP 回应", "你冲向祭坛" in merged)

# 历史 system 消息（如压缩后的前情提要）不受影响
msgs2 = [
    {"role": "system", "content": "📜 前情提要：你在阿卡姆调查失踪案。"},
    {"role": "user", "content": "我推开门"},
    {"role": "assistant", "content": "[KP回应] 你推开门。"},
    {"role": "assistant", "content": "[KP梗概] 门后是漆黑的走廊。"},
    {"role": "system", "content": "⚡ 【受到伤害 · 木刺】\n损失 1 点 HP（9 → 8/10）"},
]
char5 = generate_random_character()
char5["HP"] = 8
state5 = make_state(char=char5, pending={}, messages=msgs2)
result5 = render_node(state5)
sys_msgs5 = [m for m in result5["messages"] if m["role"] == "system"]
check("前情提要 system 消息保留", any("前情提要" in m["content"] for m in sys_msgs5))
check("本轮工具消息合并为 1 条", len(sys_msgs5) == 2, f"count={len(sys_msgs5)}")


# ================================================================
# 6. SAN 损失 ≥ 5：智力检定（真实掷骰）
# ================================================================
print("=" * 60)
print("6. sanity loss >= 5: INT check")
print("=" * 60)

orig_roll = tools_module.roll_d100

char = generate_random_character()
char["INT"] = 80
char["SAN"] = 50
char["MAX_SAN"] = 50

# 智力检定成功 → 临时疯狂
tools_module.roll_d100 = lambda: 30
result = tools_module.execute_tool("sanity_loss", {"loss": 6, "reason": "目睹怪物"}, dict(char))
check("INT 检定成功 → temp_insanity=True", result["updated_character"]["temp_insanity"] is True)
check("描述含检定成功文本", "智力检定" in result["message"] and "成功" in result["message"])

# 智力检定失败 → 压抑
tools_module.roll_d100 = lambda: 95
result = tools_module.execute_tool("sanity_loss", {"loss": 6, "reason": "目睹怪物"}, dict(char))
check("INT 检定失败 → temp_insanity=False", result["updated_character"]["temp_insanity"] is False)
check("描述含压抑文本", "压抑" in result["message"])

tools_module.roll_d100 = orig_roll


# ================================================================
# 7. SAN 归零：永久疯狂，无临时疯狂判定
# ================================================================
print("=" * 60)
print("7. SAN to zero: indefinite insanity")
print("=" * 60)

char = generate_random_character()
char["INT"] = 80
char["SAN"] = 5
char["MAX_SAN"] = 50
tools_module.roll_d100 = lambda: 10  # 即便掷出成功也不该出现临时疯狂文本
result = tools_module.execute_tool("sanity_loss", {"loss": 10, "reason": "直面神话"}, dict(char))
check("SAN 钳制到 0", result["updated_character"]["SAN"] == 0)
check("status=insane", result["updated_character"]["status"] == "insane")
check("无智力检定文本", "智力检定" not in result["message"])
tools_module.roll_d100 = orig_roll


# ================================================================
# 8. 完整图流程：伤害 + 理智 → 角色更新传播（monkeypatch LLM 节点）
# ================================================================
print("=" * 60)
print("8. full graph flow: character propagation")
print("=" * 60)

# 替换 graph 命名空间中的 KP / Render 绑定（graph.py 是 from-import，需 patch 其自身命名空间）
import backend.graph as graph_module
import backend.agents as agents_module

orig_call_kp = graph_module.call_kp
orig_call_render = graph_module.call_render

graph_module.call_kp = lambda **kwargs: {
    "kp_response": "邪教徒挥刀砍向你，你来不及躲闪。",
    "narrative": "刀锋划破你的手臂，疼痛让你清醒了几分。",
    "scene": "地下祭坛",
    "need_check": "None",
    "difficulty": "普通",
    "hp_damage": 3,
    "san_loss": 0,
    "damage_source": "邪教徒的短刀",
    "san_reason": "",
    "story_end": False,
    "suggestions": ["反击", "逃跑", "寻找掩体"],
}
agents_module.call_render = lambda narrative, check_result: narrative

char = generate_random_character()
char["HP"] = 10
char["MAX_HP"] = 10
state = create_initial_state(character=char, scene_context="地下祭坛")
state["messages"] = [{"role": "user", "content": "我站在原地不动"}]

graph = build_graph()
result = graph.invoke(state)

check("HP 10 -> 7", result["character"]["HP"] == 7, f"HP={result['character']['HP']}")
check("game_over=False", result["game_over"] is False)
check("消息包含伤害结果", any("受到伤害" in m.get("content", "") for m in result["messages"]))
check("消息包含 KP 回应", any("来不及躲闪" in m.get("content", "") for m in result["messages"]))

graph_module.call_kp = orig_call_kp
graph_module.call_render = orig_call_render


# ================================================================
# 9. _validate_kp_output：新字段容错
# ================================================================
print("=" * 60)
print("9. _validate_kp_output: field coercion")
print("=" * 60)

r = _validate_kp_output({
    "kp_response": "x", "narrative": "y", "scene": "z",
    "hp_damage": "3", "san_loss": "6",  # 字符串数字
    "damage_source": "刀", "san_reason": "尸体",
})
check("字符串 hp_damage → int 3", r["hp_damage"] == 3, f"got {r['hp_damage']}")
check("字符串 san_loss → int 6", r["san_loss"] == 6, f"got {r['san_loss']}")

r2 = _validate_kp_output({"kp_response": "x", "narrative": "y", "scene": "z"})
check("缺省字段 → 0 / 空串", r2["hp_damage"] == 0 and r2["san_loss"] == 0
      and r2["damage_source"] == "" and r2["san_reason"] == "")

r3 = _validate_kp_output({"kp_response": "x", "narrative": "y", "hp_damage": -5, "san_loss": 999})
check("负数钳制到 0", r3["hp_damage"] == 0)
check("san_loss 钳制到 100", r3["san_loss"] == 100)

r4 = _validate_kp_output({"kp_response": "x", "narrative": "y", "hp_damage": "abc", "san_loss": None})
check("非法值 → 0", r4["hp_damage"] == 0 and r4["san_loss"] == 0)


# ================================================================
# 10. story_end → game_over 传播（确定性验证，无 LLM）
# ================================================================
print("=" * 60)
print("10. story_end propagation (deterministic)")
print("=" * 60)

graph_module.call_kp = lambda **kwargs: {
    "kp_response": "故事落幕。",
    "narrative": "你转身离开，永不复返。",
    "scene": "归途",
    "need_check": "None",
    "difficulty": "普通",
    "hp_damage": 0,
    "san_loss": 0,
    "damage_source": "",
    "san_reason": "",
    "story_end": True,
    "suggestions": [],
}
graph_module.call_render = lambda narrative, check_result: narrative

char10 = generate_random_character()
state10 = create_initial_state(character=char10, scene_context="")
state10["messages"] = [{"role": "user", "content": "我放弃调查，回家"}]

graph10 = build_graph()
result10 = graph10.invoke(state10)
check("story_end=True → game_over=True", result10["game_over"] is True,
      f"game_over={result10['game_over']}")
check("故事落幕消息保留", any("故事落幕" in m.get("content", "") for m in result10["messages"]))

graph_module.call_kp = orig_call_kp
graph_module.call_render = orig_call_render


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
