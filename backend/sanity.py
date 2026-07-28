"""
理智模块 —— 理智值扣减与疯狂判定。

COC 第七版理智规则：
    - 目睹超自然事件导致 SAN 损失
    - 单次损失 ≥ 5 SAN 触发智力检定，可能临时疯狂
    - SAN ≤ 0 导致永久疯狂
"""

import random
from typing import Any, Dict, Tuple


def reduce_sanity(current_san: int, loss: int, max_san: int) -> Dict[str, Any]:
    """
    扣减理智值，返回更新后的 SAN 及疯狂判定。

    参数：
        current_san : 当前理智值
        loss        : 本次损失值
        max_san     : 最大理智值

    返回：
        {
            "new_san": int,           # 新理智值
            "loss": int,              # 实际损失
            "is_indefinite_insane": bool,  # 永久疯狂（SAN ≤ 0）
            "is_temporary_insane": bool,   # 临时疯狂（单次损失 ≥ 5）
            "description": str,            # 描述文本
        }
    """
    actual_loss = min(loss, current_san)  # 不会扣到负数
    new_san = max(0, current_san - actual_loss)

    is_temporary_insane = actual_loss >= 5 and current_san > 0
    is_indefinite_insane = new_san <= 0

    if is_indefinite_insane:
        desc = f"💀 理智彻底崩溃！丧失 {actual_loss} 点 SAN（当前 SAN: {new_san}）。调查员陷入永久疯狂，无法继续冒险。"
    elif is_temporary_insane:
        temp_duration = random.randint(1, 10)
        desc = (
            f"😱 剧烈的精神冲击！丧失 {actual_loss} 点 SAN（当前 SAN: {new_san}/{max_san}）。\n"
            f"进行一次智力检定（D100 ≤ INT）以判断是否意识到恐怖真相……\n"
            f"若智力检定成功，将触发临时疯狂（持续 {temp_duration} 回合/小时）。"
        )
    elif actual_loss > 0:
        desc = f"😨 感到不安——丧失 {actual_loss} 点 SAN（当前 SAN: {new_san}/{max_san}）。"
    else:
        desc = "😐 内心毫无波澜，未受到影响。"

    return {
        "new_san": new_san,
        "loss": actual_loss,
        "is_indefinite_insane": is_indefinite_insane,
        "is_temporary_insane": is_temporary_insane,
        "description": desc,
    }


def sanity_loss_roll(severity: str) -> int:
    """
    根据恐怖程度投掷 SAN 损失值。

    参数：
        severity : "轻微" | "中度" | "严重" | "神话"

    返回：
        SAN 损失点数
    """
    loss_map = {
        "轻微": lambda: random.randint(1, 3),      # 1D3
        "中度": lambda: random.randint(1, 6),       # 1D6
        "严重": lambda: random.randint(1, 10),      # 1D10
        "神话": lambda: random.randint(1, 100),     # 1D100
    }

    roller = loss_map.get(severity, loss_map["中度"])
    return roller()
