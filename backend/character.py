"""
角色模块 —— 调查员角色生成、更新与验证。

COC 第七版属性规则：
    - 力量/敏捷/感知/意志：3D6 × 5（15-90）
    - 智力/灵感：(2D6 + 6) × 5（40-90）
    - HP = round((力量 + 意志) / 2)
    - SAN = 意志（初始）
    - STA = round((力量 + 敏捷) / 2)
"""

import random
from typing import Any, Dict, Optional


def _roll_3d6() -> int:
    """
    内部函数：掷 3 个六面骰求和。
    """
    return sum(random.randint(1, 6) for _ in range(3))


def _roll_2d6() -> int:
    """
    内部函数：掷 2 个六面骰求和。
    """
    return sum(random.randint(1, 6) for _ in range(2))


def generate_random_character(name: str = "调查员") -> Dict[str, Any]:
    """
    随机生成一名调查员角色。

    参数：
        name : 角色名称，默认为"调查员"

    返回：
        包含完整属性的角色字典。
    """
    # 基础六维
    strength = _roll_3d6() * 5   # 力量 STR
    dexterity = _roll_3d6() * 5  # 敏捷 DEX
    power = _roll_3d6() * 5      # 感知 / 意志 POW
    will = power                  # 意志 = 感知（COC 中 POW = 意志力）
    intelligence = (_roll_2d6() + 6) * 5  # 智力 INT
    idea = intelligence           # 灵感 = 智力

    # 派生属性
    max_hp = round((strength + will) / 2)
    max_san = will
    max_sta = round((strength + dexterity) / 2)

    character = {
        "name": name,
        "STR": strength,
        "DEX": dexterity,
        "POW": power,
        "WILL": will,
        "INT": intelligence,
        "IDEA": idea,
        "HP": max_hp,
        "MAX_HP": max_hp,
        "SAN": max_san,
        "MAX_SAN": max_san,
        "STA": max_sta,
        "MAX_STA": max_sta,
        # 状态标记
        "temp_insanity": False,   # 临时疯狂
        "wounded": False,         # 重伤标记
        "status": "alive",        # alive / dead / insane
    }
    return character


def update_character(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新现有角色属性，并自动重算派生值 / 状态标记。

    参数：
        existing : 现有角色字典
        updates  : 需要更新的键值对（如 {"HP": 8, "SAN": 45}）

    返回：
        更新后的角色字典（新字典，不修改原对象）。
    """
    char = {**existing, **updates}

    # 限制 HP / SAN / STA 不超过最大值
    char["HP"] = max(0, min(char["HP"], char.get("MAX_HP", char["HP"])))
    char["SAN"] = max(0, min(char["SAN"], char.get("MAX_SAN", char["SAN"])))
    char["STA"] = max(0, min(char["STA"], char.get("MAX_STA", char["STA"])))

    # 更新状态标记
    max_hp = char.get("MAX_HP", char["HP"])
    max_san = char.get("MAX_SAN", char["SAN"])

    char["wounded"] = char["HP"] < max_hp * 0.3 if max_hp > 0 else True

    if char["HP"] <= 0:
        char["status"] = "dead"
    elif char["SAN"] <= 0:
        char["status"] = "insane"
    else:
        char["status"] = "alive"

    return char


def validate_character(char: Dict[str, Any]) -> tuple:
    """
    验证角色属性的合法性。

    返回：
        (is_valid: bool, error_message: str)
    """
    required = ["STR", "DEX", "POW", "WILL", "INT", "IDEA", "HP", "MAX_HP", "SAN", "MAX_SAN"]
    missing = [k for k in required if k not in char]
    if missing:
        return False, f"缺少必要属性: {', '.join(missing)}"

    # 范围校验
    if not (15 <= char["STR"] <= 90):
        return False, f"力量 STR 超出范围 (15-90): {char['STR']}"
    if not (15 <= char["DEX"] <= 90):
        return False, f"敏捷 DEX 超出范围 (15-90): {char['DEX']}"
    if not (15 <= char["POW"] <= 90):
        return False, f"感知 POW 超出范围 (15-90): {char['POW']}"
    if not (15 <= char["WILL"] <= 90):
        return False, f"意志 WILL 超出范围 (15-90): {char['WILL']}"
    if not (40 <= char["INT"] <= 90):
        return False, f"智力 INT 超出范围 (40-90): {char['INT']}"
    if not (40 <= char["IDEA"] <= 90):
        return False, f"灵感 IDEA 超出范围 (40-90): {char['IDEA']}"
    if not (0 <= char["HP"] <= char.get("MAX_HP", 99)):
        return False, f"HP 超出范围 (0-{char.get('MAX_HP', '?')}): {char['HP']}"
    if char["SAN"] > char.get("MAX_SAN", 99):
        return False, f"SAN 超出最大理智值: {char['SAN']}"

    return True, ""
