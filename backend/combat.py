"""
战斗模块 —— 攻击、闪避、伤害计算。

COC 第七版战斗流程：
    1. 攻击方进行攻击检定：D100 ≤ 攻击技能值
    2. 目标可选择闪避：D100 ≤ 敏捷值
    3. 命中后根据武器类型投伤害骰
"""

from typing import Any, Dict, Tuple

from .dice import roll_d100, roll_dice


def attack_check(attacker_skill: int) -> Tuple[bool, int, str]:
    """
    执行攻击检定。

    参数：
        attacker_skill : 攻击方的技能值（如格斗 50、射击 40）

    返回：
        (是否命中, D100掷骰结果, 描述信息)
    """
    roll = roll_d100()
    if roll == 1:
        return True, roll, f"🎯 攻击大成功！掷出 01"
    if roll >= 96:
        return False, roll, f"💀 攻击大失败！掷出 {roll}"
    if roll <= attacker_skill:
        return True, roll, f"⚔️ 攻击命中！掷出 {roll} ≤ {attacker_skill}"
    else:
        return False, roll, f"🏃 攻击落空！掷出 {roll} > {attacker_skill}"


def dodge_check(dex: int) -> Tuple[bool, int, str]:
    """
    执行闪避检定。

    参数：
        dex : 目标的敏捷值

    返回：
        (是否闪避成功, D100掷骰结果, 描述信息)
    """
    roll = roll_d100()
    if roll == 1:
        return True, roll, f"🌀 完美闪避！掷出 01"
    if roll >= 96:
        return False, roll, f"💀 闪避失败（大失败）！掷出 {roll}"
    if roll <= dex:
        return True, roll, f"🌀 闪避成功！掷出 {roll} ≤ {dex}"
    else:
        return False, roll, f"💥 闪避失败！掷出 {roll} > {dex}"


def weapon_damage(weapon_dice: str = "1D6") -> Tuple[int, str]:
    """
    计算武器伤害。

    参数：
        weapon_dice : 武器伤害骰表达式（如 "1D4"、"1D6"、"2D6"）

    返回：
        (伤害值, 描述信息)
    """
    try:
        dmg = roll_dice(weapon_dice)
        return dmg, f"造成 {dmg} 点伤害（{weapon_dice}）"
    except ValueError as e:
        # 解析失败时默认 1D6
        dmg = roll_dice("1D6")
        return dmg, f"武器伤害表达式无效({e})，默认 1D6：造成 {dmg} 点伤害"


def resolve_attack(
    attacker_skill: int,
    target_dex: int,
    target_hp: int,
    target_max_hp: int,
    weapon_dice: str = "1D6",
) -> Dict[str, Any]:
    """
    一次完整攻击判定：攻击检定 → 闪避检定 → 伤害计算。

    返回：
        {
            "attack_roll": int,
            "attack_success": bool,
            "dodge_roll": int | None,
            "dodge_success": bool | None,
            "damage": int,
            "description": str,
            "updated_hp": int,
        }
    """
    # 攻击检定
    atk_hit, atk_roll, atk_msg = attack_check(attacker_skill)

    if not atk_hit:
        return {
            "attack_roll": atk_roll,
            "attack_success": False,
            "dodge_roll": None,
            "dodge_success": None,
            "damage": 0,
            "description": atk_msg,
            "updated_hp": target_hp,
        }

    # 闪避检定
    dodge_ok, dodge_roll, dodge_msg = dodge_check(target_dex)

    if dodge_ok:
        return {
            "attack_roll": atk_roll,
            "attack_success": True,
            "dodge_roll": dodge_roll,
            "dodge_success": True,
            "damage": 0,
            "description": f"{atk_msg}\n{dodge_msg}",
            "updated_hp": target_hp,
        }

    # 命中，计算伤害
    dmg, dmg_msg = weapon_damage(weapon_dice)
    new_hp = max(0, target_hp - dmg)

    return {
        "attack_roll": atk_roll,
        "attack_success": True,
        "dodge_roll": dodge_roll,
        "dodge_success": False,
        "damage": dmg,
        "description": f"{atk_msg}\n{dodge_msg}\n{dmg_msg}",
        "updated_hp": new_hp,
    }
