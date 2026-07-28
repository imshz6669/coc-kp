"""
骰子引擎模块 —— COC 第七版核心骰子机制。

提供：
    - roll_d100()          : 百分骰（1-100）
    - roll_d6()            : 六面骰（1-6）
    - roll_d10()           : 十面骰（1-10）
    - check_difficulty()   : 基于难度等级的检定判定
"""

import random
from typing import Tuple


def roll_d100() -> int:
    """
    掷百分骰，返回 1-100 的整数。
    """
    return random.randint(1, 100)


def roll_d6() -> int:
    """
    掷六面骰，返回 1-6 的整数。
    """
    return random.randint(1, 6)


def roll_d10() -> int:
    """
    掷十面骰，返回 1-10 的整数。
    """
    return random.randint(1, 10)


def roll_dice(dice_str: str) -> int:
    """
    解析掷骰表达式（如 '2D6'、'1D10'、'1D100'），返回总结果。

    注意：D100 视为 1-100 均匀分布，非两个 D10 组合。
    """
    dice_str = dice_str.strip().upper()
    try:
        if "D" not in dice_str:
            raise ValueError(f"无效的掷骰表达式: {dice_str}")

        parts = dice_str.split("D")
        count = int(parts[0]) if parts[0] != "" else 1
        sides = int(parts[1])

        total = 0
        for _ in range(count):
            if sides == 100:
                total += roll_d100()
            else:
                total += random.randint(1, sides)
        return total
    except Exception as e:
        raise ValueError(f"无法解析掷骰表达式 '{dice_str}': {e}")


def check_difficulty(roll: int, attribute: int, difficulty: str) -> Tuple[bool, str]:
    """
    根据掷骰结果、属性值和难度等级，判定检定是否成功。

    参数：
        roll       : D100 掷骰结果（1-100）
        attribute  : 角色的对应属性值
        difficulty : 难度等级 —— "普通" | "困难" | "极难"

    返回：
        (是否成功, 结果描述字符串)
    """
    difficulty_map = {
        "普通": 1.0,
        "困难": 0.5,
        "极难": 0.2,
    }

    if difficulty not in difficulty_map:
        difficulty = "普通"

    threshold = int(attribute * difficulty_map[difficulty])

    # 大成功: 掷出 01
    if roll == 1:
        return True, f"🎯 大成功！掷出 01（目标 ≤ {threshold}）"
    # 大失败: 掷出 96-100
    if roll >= 96:
        return False, f"💀 大失败！掷出 {roll}（目标 ≤ {threshold}）"
    # 普通判定
    if roll <= threshold:
        return True, f"✅ 检定成功！掷出 {roll}（目标 ≤ {threshold}）"
    else:
        return False, f"❌ 检定失败！掷出 {roll}（目标 > {threshold}）"
