"""
状态管理模块 —— 对话截断、游戏状态存取、临时状态维护。

提供：
    - trim_messages()        : 对话截断，仅保留最近 N 轮
    - save_game_state()      : 保存游戏状态到内存
    - load_game_state()      : 从内存加载游戏状态
    - get_temp_status()      : 获取角色临时状态描述
    - set_temp_status()      : 设置临时状态
"""

import copy
from typing import Any, Dict, List
from utils.config import MAX_CONTEXT_ROUNDS


# ---------- 对话截断 ----------

def trim_messages(messages: List[Dict[str, str]], max_rounds: int = None) -> List[Dict[str, str]]:
    """
    截断消息列表，仅保留最近 max_rounds 轮对话。

    一轮 = 一条 user 消息 + 一条 assistant 消息。

    参数：
        messages   : 消息列表，每项含 "role" 和 "content"
        max_rounds : 保留轮数，默认使用 config.MAX_CONTEXT_ROUNDS

    返回：
        截断后的新列表。
    """
    if max_rounds is None:
        max_rounds = MAX_CONTEXT_ROUNDS

    if len(messages) <= max_rounds * 2:
        return list(messages)

    # 从末尾向前保留 max_rounds 轮
    return messages[-(max_rounds * 2):]


# ---------- 游戏状态存储（会话级内存） ----------

_game_state_registry: Dict[str, Dict[str, Any]] = {}


def save_game_state(session_id: str, character: Dict[str, Any], game_over: bool = False) -> None:
    """
    保存指定会话的游戏状态。

    参数：
        session_id : 会话唯一标识
        character  : 角色字典
        game_over  : 游戏是否结束
    """
    _game_state_registry[session_id] = {
        "character": copy.deepcopy(character),
        "game_over": game_over,
    }


def load_game_state(session_id: str) -> Dict[str, Any]:
    """
    加载指定会话的游戏状态。

    返回：
        {
            "character": dict | None,
            "game_over": bool,
        }
    """
    if session_id in _game_state_registry:
        state = _game_state_registry[session_id]
        return {
            "character": copy.deepcopy(state["character"]),
            "game_over": state["game_over"],
        }
    return {
        "character": None,
        "game_over": False,
    }


def clear_game_state(session_id: str) -> None:
    """
    清除指定会话的游戏状态。
    """
    _game_state_registry.pop(session_id, None)


# ---------- 临时状态维护 ----------

def get_temp_status(character: Dict[str, Any]) -> Dict[str, Any]:
    """
    从角色字典中提取临时状态摘要。

    返回：
        {
            "is_wounded": bool,
            "is_temp_insane": bool,
            "hp_ratio": float,     # HP 百分比
            "san_ratio": float,    # SAN 百分比
            "status": str,         # alive / dead / insane
        }
    """
    max_hp = character.get("MAX_HP", 1)
    max_san = character.get("MAX_SAN", 1)

    return {
        "is_wounded": character.get("wounded", False),
        "is_temp_insane": character.get("temp_insanity", False),
        "hp_ratio": character["HP"] / max_hp if max_hp > 0 else 0,
        "san_ratio": character["SAN"] / max_san if max_san > 0 else 0,
        "status": character.get("status", "alive"),
    }


def set_temp_status(character: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """
    设置角色的临时状态值，返回更新后的角色字典。

    参数：
        character : 角色字典
        key       : 状态键名（如 "wounded", "temp_insanity"）
        value     : 新值

    返回：
        更新后的角色字典（新对象）。
    """
    char = copy.deepcopy(character)
    char[key] = value
    return char
