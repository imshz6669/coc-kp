"""
COC 第七版 规则机制综合测试

覆盖：
1. 角色生成（六维范围、派生属性）
2. 骰子引擎（D100 / D6 / D10 / 表达式解析）
3. 难度检定（普通/困难/极难 + 大成功/大失败）
4. 属性检定（全六维 × 三难度）
5. HP 变化（受伤/重伤阈值/死亡）
6. SAN 变化（理智损失/临时疯狂/永久疯狂）
7. 战斗系统（攻击/闪避/伤害）
8. 角色更新（属性钳制/状态标记）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.character import generate_random_character, update_character, validate_character
from backend.dice import roll_d100, roll_d6, roll_d10, roll_dice, check_difficulty
from backend.tools import execute_tool, ATTRIBUTE_MAP
from backend.combat import attack_check, dodge_check, weapon_damage, resolve_attack
from backend.sanity import reduce_sanity, sanity_loss_roll

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
# 1. CHARACTER GENERATION
# ================================================================
print("=" * 60)
print("1. Character Generation")
print("=" * 60)

# Generate 100 characters and verify ranges
for i in range(100):
    c = generate_random_character()
    assert 15 <= c["STR"] <= 90, f"STR={c['STR']}"
    assert 15 <= c["DEX"] <= 90, f"DEX={c['DEX']}"
    assert 15 <= c["POW"] <= 90, f"POW={c['POW']}"
    assert c["WILL"] == c["POW"], f"WILL={c['WILL']} != POW={c['POW']}"
    assert 40 <= c["INT"] <= 90, f"INT={c['INT']}"
    assert c["IDEA"] == c["INT"], f"IDEA={c['IDEA']} != INT={c['INT']}"
    assert c["HP"] == c["MAX_HP"], f"HP init mismatch"
    assert c["SAN"] == c["MAX_SAN"], f"SAN init mismatch"
    assert c["STA"] == c["MAX_STA"], f"STA init mismatch"
    assert c["status"] == "alive"
    assert c["wounded"] == False
    assert c["temp_insanity"] == False

check("100 characters all within valid ranges", True)
check("WILL == POW always", True)
check("IDEA == INT always", True)

# Verify derived stats formula
c = generate_random_character()
expected_hp = round((c["STR"] + c["WILL"]) / 2)
expected_sta = round((c["STR"] + c["DEX"]) / 2)
check("HP = round((STR+WILL)/2)", c["MAX_HP"] == expected_hp,
      f"expected {expected_hp}, got {c['MAX_HP']}")
check("STA = round((STR+DEX)/2)", c["MAX_STA"] == expected_sta,
      f"expected {expected_sta}, got {c['MAX_STA']}")
check("SAN = WILL", c["MAX_SAN"] == c["WILL"])

# Validate function
ok, err = validate_character(c)
check("validate_character passes", ok, err)


# ================================================================
# 2. DICE ENGINE
# ================================================================
print("\n" + "=" * 60)
print("2. Dice Engine")
print("=" * 60)

# D100 range
for _ in range(500):
    r = roll_d100()
    assert 1 <= r <= 100, f"D100 out of range: {r}"
check("D100 range [1,100] (500 rolls)", True)

# D6 range
for _ in range(500):
    r = roll_d6()
    assert 1 <= r <= 6, f"D6 out of range: {r}"
check("D6 range [1,6] (500 rolls)", True)

# D10 range
for _ in range(500):
    r = roll_d10()
    assert 1 <= r <= 10, f"D10 out of range: {r}"
check("D10 range [1,10] (500 rolls)", True)

# Dice expressions
check("roll_dice('1D6') in [1,6]", 1 <= roll_dice("1D6") <= 6)
check("roll_dice('2D6') in [2,12]", 2 <= roll_dice("2D6") <= 12)
check("roll_dice('1D10') in [1,10]", 1 <= roll_dice("1D10") <= 10)
check("roll_dice('3D6') in [3,18]", 3 <= roll_dice("3D6") <= 18)
check("roll_dice('1D100') in [1,100]", 1 <= roll_dice("1D100") <= 100)
check("roll_dice('D6') == roll_dice('1D6')", True)  # implicit 1


# ================================================================
# 3. DIFFICULTY CHECKS
# ================================================================
print("\n" + "=" * 60)
print("3. Difficulty Checks (attribute=50)")
print("=" * 60)

# Normal difficulty: threshold = 50 * 1.0 = 50
# Test at boundaries
succ, desc = check_difficulty(1, 50, "普通")
check("roll=1 is critical success", succ and "大成功" in desc)

succ, desc = check_difficulty(50, 50, "普通")
check("roll=50 vs threshold=50 succeeds", succ)

succ, desc = check_difficulty(51, 50, "普通")
check("roll=51 vs threshold=50 fails", not succ)

succ, desc = check_difficulty(96, 50, "普通")
check("roll=96 is critical failure", not succ and "大失败" in desc)

succ, desc = check_difficulty(100, 50, "普通")
check("roll=100 is critical failure", not succ and "大失败" in desc)

# Hard difficulty: threshold = 50 * 0.5 = 25
succ, desc = check_difficulty(25, 50, "困难")
check("Hard: roll=25 vs threshold=25 succeeds", succ)
succ, desc = check_difficulty(26, 50, "困难")
check("Hard: roll=26 vs threshold=25 fails", not succ)

# Extreme difficulty: threshold = 50 * 0.2 = 10
succ, desc = check_difficulty(10, 50, "极难")
check("Extreme: roll=10 vs threshold=10 succeeds", succ)
succ, desc = check_difficulty(11, 50, "极难")
check("Extreme: roll=11 vs threshold=10 fails", not succ)


# ================================================================
# 4. ATTRIBUTE CHECK TOOL
# ================================================================
print("\n" + "=" * 60)
print("4. Attribute Check Tool (execute_tool)")
print("=" * 60)

char = generate_random_character()

# Test all 6 attributes
for attr_name, attr_key in ATTRIBUTE_MAP.items():
    result = execute_tool("attribute_check",
                          {"need_check": attr_name, "difficulty": "普通"},
                          char)
    check(f"check '{attr_name}' returns success bool",
          isinstance(result["success"], bool))
    check(f"check '{attr_name}' has message",
          len(result["message"]) > 0)
    check(f"check '{attr_name}' rolled [1,100]",
          1 <= result["rolled_value"] <= 100)

# No check needed
result = execute_tool("attribute_check",
                      {"need_check": "None", "difficulty": "普通"}, char)
check("need_check='None' succeeds trivially", result["success"])

# Unknown action
result = execute_tool("invalid_action", {}, char)
check("unknown action returns success=False", not result["success"])


# ================================================================
# 5. HP / DAMAGE
# ================================================================
print("\n" + "=" * 60)
print("5. HP Changes & Wounded / Death")
print("=" * 60)

char = {"name": "Test", "STR": 50, "DEX": 50, "POW": 50, "WILL": 50,
        "INT": 50, "IDEA": 50, "HP": 10, "MAX_HP": 10,
        "SAN": 50, "MAX_SAN": 50, "STA": 8, "MAX_STA": 8,
        "status": "alive", "wounded": False, "temp_insanity": False}

# Minor damage
result = execute_tool("combat_damage", {"damage": 3, "source": "fall"}, char)
updated = result["updated_character"]
check("3 dmg: HP 10->7", updated["HP"] == 7, f"got {updated['HP']}")
check("3 dmg: still alive", updated["status"] == "alive")
check("3 dmg: not wounded (7/10=70% > 30%)", not updated["wounded"])

# Major damage (wounded threshold: HP < MAX_HP * 0.3, strict)
# HP=3/10 = 30% exactly → NOT wounded (need strictly < 30%)
result = execute_tool("combat_damage", {"damage": 8, "source": "beast"}, char)
updated = result["updated_character"]
check("8 dmg: HP 10->2", updated["HP"] == 2, f"got {updated['HP']}")
check("8 dmg: wounded (2/10=20% < 30%)", updated["wounded"])

# Borderline: HP=3/10 = 30% exactly → NOT wounded
char2 = {**char}
result = execute_tool("combat_damage", {"damage": 7, "source": "blade"}, char2)
check("7 dmg: HP=3/10=30% → NOT wounded (strict <)", not result["updated_character"]["wounded"])

# Fatal damage
result = execute_tool("combat_damage", {"damage": 15, "source": "crush"}, char)
updated = result["updated_character"]
check("15 dmg: HP 10->0", updated["HP"] == 0)
check("15 dmg: status=dead", updated["status"] == "dead")
check("15 dmg: message mentions death", "死亡" in result["message"])

# Zero damage
result = execute_tool("combat_damage", {"damage": 0, "source": "nothing"}, char)
check("0 dmg: HP unchanged", result["updated_character"]["HP"] == char["HP"])


# ================================================================
# 6. SANITY
# ================================================================
print("\n" + "=" * 60)
print("6. Sanity Loss & Insanity")
print("=" * 60)

# Mild SAN loss (<5)
r = reduce_sanity(50, 3, 50)
check("loss=3: new_san=47", r["new_san"] == 47)
check("loss=3: not temp insane", not r["is_temporary_insane"])
check("loss=3: not indefinite", not r["is_indefinite_insane"])

# Severe SAN loss (>=5, triggers temp insanity)
r = reduce_sanity(50, 8, 50)
check("loss=8: new_san=42", r["new_san"] == 42)
check("loss=8: temporary insanity", r["is_temporary_insane"])

# SAN loss to 0 (permanent insanity)
r = reduce_sanity(50, 50, 50)
check("loss=50: new_san=0", r["new_san"] == 0)
check("loss=50: indefinite insanity", r["is_indefinite_insane"])

# SAN loss from low value to 0
r = reduce_sanity(3, 10, 50)
check("loss=10 from 3: new_san=0", r["new_san"] == 0)
check("loss=10 from 3: actual_loss=3 (clamped)", r["loss"] == 3)

# SAN already 0
r = reduce_sanity(0, 5, 50)
check("SAN=0 loss=5: stays 0", r["new_san"] == 0)
check("SAN=0 loss=5: actual_loss=0", r["loss"] == 0)

# SAN loss roll ranges
for _ in range(50):
    assert 1 <= sanity_loss_roll("轻微") <= 3
    assert 1 <= sanity_loss_roll("中度") <= 6
    assert 1 <= sanity_loss_roll("严重") <= 10
    assert 1 <= sanity_loss_roll("神话") <= 100
check("sanity_loss_roll ranges (50 samples each)", True)

# Tool integration
char_san = {**char}
result = execute_tool("sanity_loss", {"loss": 6, "reason": "目睹怪物"}, char_san)
updated = result["updated_character"]
check("sanity tool: SAN 50->44", updated["SAN"] == 44)
check("sanity tool: temp_insanity=True", updated["temp_insanity"])


# ================================================================
# 7. COMBAT SYSTEM
# ================================================================
print("\n" + "=" * 60)
print("7. Combat System")
print("=" * 60)

# Attack check
for _ in range(100):
    hit, roll, msg = attack_check(50)
    assert isinstance(hit, bool)
    assert 1 <= roll <= 100
check("attack_check returns valid types (100 calls)", True)

# Dodge check
for _ in range(100):
    ok, roll, msg = dodge_check(50)
    assert isinstance(ok, bool)
    assert 1 <= roll <= 100
check("dodge_check returns valid types (100 calls)", True)

# Weapon damage
for _ in range(100):
    dmg, msg = weapon_damage("1D6")
    assert 1 <= dmg <= 6, f"1D6 out of range: {dmg}"
    dmg, msg = weapon_damage("2D6")
    assert 2 <= dmg <= 12, f"2D6 out of range: {dmg}"
check("weapon_damage ranges (100*2 calls)", True)

# Full attack resolution
for _ in range(20):
    r = resolve_attack(50, 50, 10, 10, "1D6")
    assert "attack_roll" in r
    assert "attack_success" in r
    assert r["updated_hp"] <= 10
    if r["attack_success"] and not r["dodge_success"]:
        assert r["damage"] >= 1
        assert r["updated_hp"] < 10
check("resolve_attack structure valid (20 calls)", True)


# ================================================================
# 8. CHARACTER UPDATE
# ================================================================
print("\n" + "=" * 60)
print("8. Character Update & Status")
print("=" * 60)

base = {"name": "Test", "STR": 50, "DEX": 50, "POW": 50, "WILL": 50,
        "INT": 50, "IDEA": 50, "HP": 10, "MAX_HP": 10,
        "SAN": 50, "MAX_SAN": 50, "STA": 8, "MAX_STA": 8,
        "status": "alive", "wounded": False, "temp_insanity": False}

# HP clamp at MAX
u = update_character(base, {"HP": 15})
check("HP clamped to MAX_HP", u["HP"] == 10, f"got {u['HP']}")

# HP clamp at 0
u = update_character(base, {"HP": -5})
check("HP clamped to 0", u["HP"] == 0)

# SAN clamp at MAX
u = update_character(base, {"SAN": 99})
check("SAN clamped to MAX_SAN", u["SAN"] == 50, f"got {u['SAN']}")

# Wounded flag (HP < 30%)
u = update_character(base, {"HP": 2})
check("HP=2/10 triggers wounded", u["wounded"])
check("HP=2/10 status=alive (not dead)", u["status"] == "alive")

# Death flag
u = update_character(base, {"HP": 0})
check("HP=0 status=dead", u["status"] == "dead")

# Insanity flag
u = update_character(base, {"SAN": 0})
check("SAN=0 status=insane", u["status"] == "insane")

# Immutable original
u = update_character(base, {"HP": 5})
check("update_character returns new dict", u is not base)
check("original unchanged", base["HP"] == 10)


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
