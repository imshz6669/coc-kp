"""
统一工具入口模块 —— 整合所有工具函数，提供 execute_tool() 统一入口。

所有工具返回格式：
    {
        "success": bool,
        "message": str,
        "updated_character": dict,
        "rolled_value": int,
    }
"""

from typing import Any, Dict

from .dice import roll_d100, check_difficulty
from .character import update_character
from .sanity import reduce_sanity

# 属性名称映射：从 KP 输出的 need_check → 角色字典键名
ATTRIBUTE_MAP = {
    "力量": "STR",
    "敏捷": "DEX",
    "感知": "POW",
    "智力": "INT",
    "灵感": "IDEA",
    "意志": "WILL",
}


def execute_tool(action: str, params: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一工具入口函数。

    参数：
        action    : 工具动作名称（"attribute_check" | "sanity_loss" | "combat_damage"）
        params    : 动作参数（need_check, difficulty, loss, damage 等）
        character : 当前角色状态字典

    返回：
        统一格式的检定结果字典。
    """
    try:
        if action == "attribute_check":
            return _tool_attribute_check(params, character)
        elif action == "sanity_loss":
            return _tool_sanity_loss(params, character)
        elif action == "combat_damage":
            return _tool_combat_damage(params, character)
        else:
            return {
                "success": False,
                "message": f"未知工具动作: {action}",
                "updated_character": character,
                "rolled_value": 0,
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"工具执行异常: {str(e)}",
            "updated_character": character,
            "rolled_value": 0,
        }


def _tool_attribute_check(params: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
    """
    属性检定工具：掷 D100，根据难度判定成功/失败。

    参数 params 需包含：
        - need_check : 检定属性名称（如"力量"）
        - difficulty : 难度（"普通"/"困难"/"极难"）
    """
    need_check = params.get("need_check", "None")
    if need_check == "None" or need_check not in ATTRIBUTE_MAP:
        return {
            "success": True,
            "message": "无需检定",
            "updated_character": character,
            "rolled_value": 0,
        }

    attr_key = ATTRIBUTE_MAP[need_check]
    attr_value = character.get(attr_key, 50)
    difficulty = params.get("difficulty", "普通")

    roll = roll_d100()
    success, desc = check_difficulty(roll, attr_value, difficulty)

    message = f"🎲 【{need_check}检定 · {difficulty}难度】\n" \
              f"属性值: {attr_value} | 掷出: {roll}\n" \
              f"结果: {desc}"

    return {
        "success": success,
        "message": message,
        "updated_character": character,
        "rolled_value": roll,
    }


def _tool_sanity_loss(params: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
    """
    理智损失工具：扣减 SAN 并判定疯狂状态。

    参数 params 需包含：
        - loss     : SAN 损失值（整数）
        - reason   : 损失原因描述（可选）
    """
    loss = params.get("loss", 0)
    reason = params.get("reason", "不明恐怖事件")

    if loss <= 0:
        return {
            "success": True,
            "message": "无理智损失",
            "updated_character": character,
            "rolled_value": 0,
        }

    result = reduce_sanity(character["SAN"], loss, character.get("MAX_SAN", character["SAN"]))

    updated = update_character(character, {
        "SAN": result["new_san"],
        "temp_insanity": result["is_temporary_insane"],
    })

    message = f"🧠 【理智损失 · {reason}】\n{result['description']}"

    return {
        "success": True,
        "message": message,
        "updated_character": updated,
        "rolled_value": result["loss"],
    }


def _tool_combat_damage(params: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, Any]:
    """
    战斗伤害工具：直接扣减 HP。

    参数 params 需包含：
        - damage   : 伤害值
        - source   : 伤害来源描述（可选）
    """
    damage = params.get("damage", 0)
    source = params.get("source", "不明伤害")

    if damage <= 0:
        return {
            "success": True,
            "message": "未受到伤害",
            "updated_character": character,
            "rolled_value": 0,
        }

    new_hp = max(0, character["HP"] - damage)
    updated = update_character(character, {"HP": new_hp})

    desc = f"⚡ 【受到伤害 · {source}】\n" \
           f"损失 {damage} 点 HP（{character['HP']} → {new_hp}/{character.get('MAX_HP', '?')}）"

    if new_hp <= 0:
        desc += "\n\n💀 生命值归零——调查员已死亡！"

    return {
        "success": True,
        "message": desc,
        "updated_character": updated,
        "rolled_value": damage,
    }
