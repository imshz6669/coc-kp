"""
LangGraph 主控工作流模块 —— COC AI KP 引擎的核心编排逻辑。

工作流：
    START → kp_node → [条件路由] → tool_node → render_node → END
                                              ↘ render_node → END

每次玩家输入触发一次 invoke：
- 无需检定：KP 输出直接作为最终回复展示，本轮结束
- 需要检定：执行工具 → 渲染检定结果 → 结束
"""

import json
import copy
from typing import Any, Dict, List, TypedDict, Literal

from langgraph.graph import StateGraph, END

from .agents import call_kp, call_render
from .tools import execute_tool
from .character import update_character, validate_character
from .state_manager import trim_messages, get_temp_status
from utils.logger import get_logger

logger = get_logger(__name__)


# ===================== 消息过滤工具 =====================

def _clean_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    过滤内部标记消息（[KP梗概] / [KP回应]），返回干净的对话历史。

    KP 不应该看到这些内部标记——它们仅用于 render_node 的处理流程。
    """
    return [
        m for m in messages
        if not (
            m.get("role") == "assistant"
            and ("[KP梗概]" in m.get("content", "") or "[KP回应]" in m.get("content", ""))
        )
    ]


# ===================== State 定义 =====================

class KeeperState(TypedDict):
    """
    LangGraph 工作流状态。

    messages       : 最近 N 轮对话历史（每条含 role + content）
    character      : 完整角色字典
    game_over      : 是否游戏结束
    pending_check  : 待执行的检定信息 {"need_check": str, "difficulty": str,
                                        "hp_damage": int, "san_loss": int,
                                        "damage_source": str, "san_reason": str}
    rag_context    : 当前轮次检索到的知识库上下文
    temp_status    : 临时状态摘要（is_wounded, is_temp_insane, etc.）
    rendered_text  : 最终润色文本
    memory_summary : 历史记忆摘要文本（注入 KP 系统提示词）
    scene_context  : 当前场景设定（仅在会话开始时有值，用于引导 KP 叙述方向）
    suggestions    : 当前轮的 3 个行动建议（供前端渲染新手引导按钮）
    current_scene  : 当前场景位置名称（由 KP 输出，供侧边栏进度追踪）
    """
    messages: List[Dict[str, str]]
    character: Dict[str, Any]
    game_over: bool
    pending_check: Dict[str, Any]
    rag_context: str
    temp_status: Dict[str, Any]
    rendered_text: str
    memory_summary: str
    scene_context: str
    suggestions: List[str]
    current_scene: str


# ===================== 节点定义 =====================

def kp_node(state: KeeperState) -> KeeperState:
    """
    KP 节点 —— 调用 KP Agent（DeepSeek V4 Pro）。

    职责：
        1. 接收玩家输入 + 角色状态 + RAG 上下文
        2. 生成剧情梗概与检定需求（JSON）
        3. 将结果写入 state
    """
    logger.info("=== KP Node Start ===")

    character = state.get("character", {})
    messages = state.get("messages", [])
    rag_context = state.get("rag_context", "")
    memory_summary = state.get("memory_summary", "")
    scene_context = state.get("scene_context", "")

    # 取最后一条 user 消息作为当前输入
    player_input = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            player_input = msg["content"]
            break

    if not player_input:
        logger.warning("未找到玩家输入，使用默认占位。")
        player_input = "（等待玩家行动……）"

    # 构建干净的对话历史给 KP（不含内部标记消息，不含本轮 user 输入）
    clean_messages = _clean_messages(messages)
    history = [m for m in clean_messages if m.get("role") != "user" or m["content"] != player_input]
    history = trim_messages(history)

    # 如果是首轮（没有历史消息），在 player_input 前加上场景引导前缀，
    # 确保 KP 同时理解场景背景和玩家行动意图
    is_first_round = (not history or len(history) == 0)
    effective_input = player_input
    if is_first_round and scene_context:
        effective_input = (
            f"【场景背景】\n{scene_context}\n\n"
            f"【玩家行动：请针对此行动做出回应，不要重复开场白】\n{player_input}"
        )

    # 调用 KP
    kp_result = call_kp(
        player_input=effective_input,
        character=character,
        rag_context=rag_context,
        messages_history=history,
        memory_summary=memory_summary,
        scene_context="" if is_first_round else scene_context,  # 首轮已合并到 input 中
    )

    logger.info(f"KP 结果: need_check={kp_result['need_check']}, difficulty={kp_result['difficulty']}, "
                f"hp_damage={kp_result.get('hp_damage', 0)}, san_loss={kp_result.get('san_loss', 0)}")

    need_check = kp_result.get("need_check", "None")
    kp_response = kp_result.get("kp_response", "")
    narrative = kp_result.get("narrative", "")

    # 本轮工具执行条件：属性检定 / 受到伤害 / 理智损失，任一触发则进入 tool_node
    hp_damage = max(0, int(kp_result.get("hp_damage", 0) or 0))
    san_loss = max(0, int(kp_result.get("san_loss", 0) or 0))
    needs_tools = (need_check and need_check != "None") or hp_damage > 0 or san_loss > 0

    # 将 KP 输出存入消息列表
    new_messages = list(messages)
    if needs_tools:
        # 需要工具执行：用特殊标记暂存 kp_response 和 narrative，render_node 后续融合结果
        new_messages.append({
            "role": "assistant",
            "content": f"[KP回应] {kp_response}",
        })
        new_messages.append({
            "role": "assistant",
            "content": f"[KP梗概] {narrative}",
        })
    else:
        # 无需工具：两层直接展示 —— 先游戏层面回应，再环境渲染
        if kp_response:
            new_messages.append({
                "role": "system",
                "content": kp_response,
            })
        new_messages.append({
            "role": "assistant",
            "content": narrative,
        })

    # 叙事结局：KP 主动判定故事到达终点
    story_end = kp_result.get("story_end", False)
    if story_end:
        logger.info("KP 判定：故事到达叙事终点（story_end=true）")

    # 行动建议（新手引导）
    suggestions = kp_result.get("suggestions", [])
    # 当前场景（KP 输出，用于侧边栏进度）
    kp_scene = kp_result.get("scene", "")

    return {
        **state,
        "messages": new_messages,
        "pending_check": {
            "need_check": need_check,
            "difficulty": kp_result.get("difficulty", "普通"),
            "hp_damage": hp_damage,
            "san_loss": san_loss,
            "damage_source": str(kp_result.get("damage_source", "") or ""),
            "san_reason": str(kp_result.get("san_reason", "") or ""),
        },
        "game_over": state.get("game_over", False) or story_end,
        "suggestions": suggestions,
        "current_scene": kp_scene if kp_scene else state.get("current_scene", ""),
        "rag_context": "",  # 本轮 RAG 上下文已使用完毕
        "scene_context": "",  # 首轮后清除，避免每轮都重复注入
    }


def tool_node(state: KeeperState) -> KeeperState:
    """
    工具节点 —— 执行属性检定 / 战斗伤害 / 理智损失。

    职责：
        1. 根据 pending_check 链式调用 tools.execute_tool()
           （检定 → 伤害 → 理智，依次作用于更新后的角色）
        2. 更新角色状态
        3. 将每条工具结果写入消息列表
    """
    logger.info("=== Tool Node Start ===")

    pending = state.get("pending_check", {})
    character = state.get("character", {})
    messages = list(state.get("messages", []))

    need_check = pending.get("need_check", "None")
    difficulty = pending.get("difficulty", "普通")
    hp_damage = max(0, int(pending.get("hp_damage", 0) or 0))
    san_loss = max(0, int(pending.get("san_loss", 0) or 0))

    updated_character = character
    tool_messages = []

    # 1. 属性检定（仅当 KP 要求检定）
    if need_check and need_check != "None":
        result = execute_tool(
            action="attribute_check",
            params={"need_check": need_check, "difficulty": difficulty},
            character=updated_character,
        )
        updated_character = result.get("updated_character", updated_character)
        tool_messages.append(result.get("message", ""))
        logger.info(f"属性检定: success={result['success']}, rolled={result.get('rolled_value')}")

    # 2. 战斗伤害（仅当本轮实际受伤）
    if hp_damage > 0:
        result = execute_tool(
            action="combat_damage",
            params={
                "damage": hp_damage,
                "source": pending.get("damage_source") or "不明伤害",
            },
            character=updated_character,
        )
        updated_character = result.get("updated_character", updated_character)
        tool_messages.append(result.get("message", ""))
        logger.info(f"战斗伤害: {hp_damage} 点, HP={updated_character.get('HP')}")

    # 3. 理智损失（仅当本轮目睹恐怖）
    if san_loss > 0:
        result = execute_tool(
            action="sanity_loss",
            params={
                "loss": san_loss,
                "reason": pending.get("san_reason") or "不明恐怖事件",
            },
            character=updated_character,
        )
        updated_character = result.get("updated_character", updated_character)
        tool_messages.append(result.get("message", ""))
        logger.info(f"理智损失: {san_loss} 点, SAN={updated_character.get('SAN')}")

    # 将工具结果逐条写入消息
    for msg in tool_messages:
        messages.append({
            "role": "system",
            "content": msg,
        })

    return {
        **state,
        "messages": messages,
        "character": updated_character,
        "temp_status": get_temp_status(updated_character),
    }


def render_node(state: KeeperState) -> KeeperState:
    """
    渲染节点 —— 将工具结果（检定/伤害/理智）融入 KP 的环境叙事。

    仅当本轮有工具执行时运行（tool_node 之后进入）。
    职责：
        1. 提取 [KP回应]（游戏层面）+ [KP梗概]（环境叙事种子）
        2. 收集内部标记之后 tool_node 追加的全部 system 消息（检定/伤害/理智）
        3. 调用 call_render() 将工具结果自然融入环境叙事
        4. 将合并后的结果与 rendered narrative 写入消息
        5. 判定 game_over 条件
    """
    logger.info("=== Render Node Start ===")

    character = state.get("character", {})
    messages = list(state.get("messages", []))

    # 提取 KP 回应（游戏层面）
    kp_response = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "[KP回应]" in msg.get("content", ""):
            kp_response = msg["content"].replace("[KP回应] ", "")
            break

    # 提取 KP 梗概（环境叙事种子）
    narrative = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "[KP梗概]" in msg.get("content", ""):
            narrative = msg["content"].replace("[KP梗概] ", "")
            break

    if not narrative:
        narrative = "周围是一片令人窒息的黑暗……"

    # 提取本轮工具结果：内部标记之后的所有 system 消息
    # （tool_node 可能追加多条：检定 / 伤害 / 理智，全部收集合并）
    marker_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and (
            "[KP回应]" in msg.get("content", "") or "[KP梗概]" in msg.get("content", "")
        ):
            marker_idx = i

    tool_msgs = []
    for i, msg in enumerate(messages):
        if i > marker_idx and msg.get("role") == "system":
            tool_msgs.append(msg["content"])
    check_result = "\n\n".join(tool_msgs)

    # 调用渲染 Agent —— 将检定结果融入环境叙事
    rendered = call_render(narrative, check_result)

    # game_over 判定（HP 归零 / SAN 归零 / KP 叙事结局）
    game_over = state.get("game_over", False)
    if character.get("HP", 1) <= 0:
        game_over = True
    elif character.get("SAN", 1) <= 0:
        game_over = True

    # 单次遍历清理：移除内部标记消息 + 标记之后本轮追加的 tool 消息
    cleaned = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and (
            "[KP回应]" in m.get("content", "") or "[KP梗概]" in m.get("content", "")
        ):
            continue
        if i > marker_idx and m.get("role") == "system":
            continue
        cleaned.append(m)
    messages = cleaned

    # 合并工具结果与 KP 回应为一条 system 消息
    combined_system = ""
    if check_result:
        combined_system += check_result
    if kp_response:
        if combined_system:
            combined_system += "\n\n" + kp_response
        else:
            combined_system = kp_response

    if combined_system:
        messages.append({
            "role": "system",
            "content": combined_system,
        })

    # 环境渲染作为最终 assistant 消息
    messages.append({
        "role": "assistant",
        "content": rendered,
    })

    logger.info(f"Render 完成，game_over={game_over}")

    return {
        **state,
        "messages": messages,
        "rendered_text": rendered,
        "game_over": game_over,
        "rag_context": "",  # 清空，下一轮重新检索
    }


# ===================== 路由逻辑 =====================

def route_after_kp(state: KeeperState) -> Literal["tool_node", "end"]:
    """
    根据 KP 输出决定下一步。
    - 属性检定 / 受到伤害 / 理智损失 任一存在 → tool_node 执行 → render_node 融合结果
    - 均无 → 直接结束（KP 输出已作为最终回复展示给玩家）
    """
    pending = state.get("pending_check", {})
    need_check = pending.get("need_check", "None")
    hp_damage = max(0, int(pending.get("hp_damage", 0) or 0))
    san_loss = max(0, int(pending.get("san_loss", 0) or 0))

    if (need_check and need_check != "None") or hp_damage > 0 or san_loss > 0:
        logger.info(f"路由 → tool_node (need_check={need_check}, hp_damage={hp_damage}, san_loss={san_loss})")
        return "tool_node"
    else:
        logger.info("路由 → END (无需工具执行，KP 输出直接展示)")
        return "end"


# ===================== 图构建 =====================

def build_graph() -> StateGraph:
    """
    构建并编译 LangGraph 工作流。

    返回：
        编译好的 StateGraph 实例，可直接调用 .invoke()。
    """
    workflow = StateGraph(KeeperState)

    # 添加节点
    workflow.add_node("kp_node", kp_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("render_node", render_node)

    # 设置入口
    workflow.set_entry_point("kp_node")

    # 条件路由：kp_node → tool_node 或 END
    workflow.add_conditional_edges(
        "kp_node",
        route_after_kp,
        {
            "tool_node": "tool_node",
            "end": END,  # 无需检定时直接结束
        },
    )

    # tool_node → render_node（检定完成后必然进入渲染）
    workflow.add_edge("tool_node", "render_node")

    # render_node 为终点
    workflow.add_edge("render_node", END)

    compiled = workflow.compile()
    logger.info("LangGraph 工作流编译完成。")
    return compiled


# ===================== 便捷函数 =====================

def create_initial_state(
    character: Dict[str, Any],
    rag_context: str = "",
    memory_summary: str = "",
    scene_context: str = "",
) -> Dict[str, Any]:
    """
    创建初始的 KeeperState，用于每次 invoke。

    参数：
        character      : 角色字典
        rag_context    : RAG 检索到的上下文
        memory_summary : 历史记忆摘要（用于恢复已有会话）
        scene_context  : 当前场景设定（开场白文本，引导 KP 叙述方向）

    返回：
        初始状态字典。
    """
    return {
        "messages": [],
        "character": character,
        "game_over": False,
        "pending_check": {},
        "rag_context": rag_context,
        "temp_status": get_temp_status(character),
        "rendered_text": "",
        "memory_summary": memory_summary,
        "scene_context": scene_context,
        "suggestions": [],
        "current_scene": "",
    }


def run_one_round(
    graph: StateGraph,
    state: Dict[str, Any],
    player_input: str,
    memory_manager=None,
) -> Dict[str, Any]:
    """
    执行一轮对话，返回更新后的状态。

    在 graph.invoke() 之后自动触发记忆概括（每 N 轮一次）。

    参数：
        graph          : 编译好的 LangGraph 工作流
        state          : 当前状态字典
        player_input   : 玩家输入文本
        memory_manager : MemoryManager 实例（可选，用于记忆概括）

    返回：
        更新后的状态字典。
    """
    # 将玩家输入加入消息列表（保留完整历史，UI 可回溯到故事开头）
    # KP 的上下文截断在 kp_node 中单独处理
    messages = _clean_messages(list(state.get("messages", [])))
    messages.append({"role": "user", "content": player_input})

    # 注入记忆上下文（在 graph 执行前设置，确保 kp_node 能读到）
    new_state = {**state, "messages": messages}
    if memory_manager:
        new_state["memory_summary"] = memory_manager.get_context()

    # 调用 graph
    try:
        result = graph.invoke(new_state)
        logger.info(f"一轮对话完成。game_over={result.get('game_over', False)}")
    except Exception as e:
        logger.error(f"Graph invoke 异常: {e}")
        # 异常时返回原始状态 + 错误信息
        messages.append({
            "role": "assistant",
            "content": f"（系统异常：{e}）",
        })
        return {**state, "messages": messages}

    # ---- 记忆概括（在 graph 完成后触发） ----
    if memory_manager:
        # 提取本轮 KP 回复（润色后的最终输出，不含梗概）
        assistant_msg = ""
        for msg in reversed(result.get("messages", [])):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant" and "[KP梗概]" not in content and "[KP回应]" not in content:
                assistant_msg = content
                break

        # 记录本轮对话
        should_summarize = memory_manager.add_round(player_input, assistant_msg)

        if should_summarize:
            logger.info("Memory: 触发概括阈值，调用 LLM 概括...")
            try:
                from utils.config import get_openai_client
                client = get_openai_client()
                memory_manager.summarize(client)
            except Exception as e:
                logger.error(f"Memory: 概括流程异常（不影响游戏）: {e}")

        # 更新 state 中的 memory_summary（供下一轮使用）
        result["memory_summary"] = memory_manager.get_context()

        # ---- 消息压缩：利用 memory summary 替代膨胀的历史 ----
        result = _compress_messages(result, memory_manager)

    return result


def _compress_messages(state: dict, memory_manager) -> dict:
    """
    当记忆概括触发后，压缩消息列表：保留最近 N 轮 + 插入摘要标记。
    防止消息列表无限膨胀导致 UI 卡顿和 API token 浪费。
    """
    from utils.config import MAX_CONTEXT_ROUNDS

    messages = state.get("messages", [])
    summary = state.get("memory_summary", "")

    # 统计 user 消息数量（= 轮次）
    user_count = sum(1 for m in messages if m.get("role") == "user")
    if user_count <= MAX_CONTEXT_ROUNDS + 2:
        return state  # 轮次还少，不需要压缩

    # 保留最近 MAX_CONTEXT_ROUNDS 轮（从后往前找 N 条 user 消息）
    kept = []
    user_seen = 0
    for msg in reversed(messages):
        kept.insert(0, msg)
        if msg.get("role") == "user":
            user_seen += 1
            if user_seen >= MAX_CONTEXT_ROUNDS:
                break

    # 如果有摘要，在最前面插入一条摘要标记（仅用于 UI 展示）
    if summary and len(summary) > 20:
        kept.insert(0, {
            "role": "system",
            "content": f"📜 前情提要：{summary[:200]}",
        })

    logger.info(f"消息压缩: {len(messages)} → {len(kept)} 条 (保留最近 {MAX_CONTEXT_ROUNDS} 轮)")

    return {**state, "messages": kept}
