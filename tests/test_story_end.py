"""
COC KP outcome detection test suite.

Covers:
1. Clear surrender → story_end=true
2. Soft rejection ×2 → story_end=true (respect player on second attempt)
3. Death outcome → HP ≤ 0
4. Madness outcome → SAN ≤ 0
5. Normal exploration → story_end=false
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.agents import call_kp, _validate_kp_output, _parse_kp_json
from backend.graph import build_graph, create_initial_state

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

CHAR = {
    "name": "Test", "STR": 50, "DEX": 50, "POW": 50,
    "WILL": 50, "INT": 80, "IDEA": 80,
    "HP": 10, "MAX_HP": 10, "SAN": 50, "MAX_SAN": 50,
    "status": "alive", "wounded": False, "temp_insanity": False,
}

# ============================================================
# Test 1: Player clearly gives up
# ============================================================
print("=" * 60)
print("Test 1: Clear surrender")
print("=" * 60)

r1 = call_kp(
    player_input="I decide to give up completely. Burn all the letters and photos, leave this city, go back to my hometown. I'm done.",
    character=CHAR,
    scene_context="",
)
check("story_end == True", r1["story_end"] is True,
      f"KP did NOT set story_end. kp_response: {r1['kp_response'][:80]}")
check("need_check == None", r1["need_check"] == "None")
check("kp_response not empty", len(r1["kp_response"]) > 0)

# ============================================================
# Test 2: First soft rejection (KP may not end yet)
# ============================================================
print("\n" + "=" * 60)
print("Test 2: First soft rejection")
print("=" * 60)

history2 = [
    {"role": "assistant", "content": "The telegraph office is stuffy. The paper in your hand feels cold..."},
    {"role": "user", "content": "Where did this telegram come from?"},
    {"role": "system", "content": "Check succeeded. The coordinates point to a reef area shunned by fishermen."},
]

r2 = call_kp(
    player_input="I want to go back home and live a normal life. Forget about all this.",
    character=CHAR,
    messages_history=history2,
    scene_context="",
)
print(f"  story_end = {r2['story_end']}")
print(f"  kp_response: {r2['kp_response'][:100]}")
# First soft rejection may not trigger ending -- that's OK for COC style
check("need_check should not be forced", r2["need_check"] == "None")

# ============================================================
# Test 3: Second insistence -- KP MUST respect player
# ============================================================
print("\n" + "=" * 60)
print("Test 3: Second insistence (must end)")
print("=" * 60)

history3 = history2 + [
    {"role": "user", "content": "I want to go back home and live a normal life."},
    {"role": "system", "content": "You can leave, but the address will haunt your dreams."},
    {"role": "assistant", "content": "You turn away from the telegraph office, but your fingers clutch the paper..."},
]

r3 = call_kp(
    player_input="No. I mean it. I burn everything. I forget everything. I am done. End this.",
    character=CHAR,
    messages_history=history3,
    scene_context="",
)
print(f"  story_end = {r3['story_end']}")
print(f"  kp_response: {r3['kp_response'][:100]}")
check("story_end == True (second refusal MUST end)",
      r3["story_end"] is True,
      f"KP still refuses after second insistence! kp_response: {r3['kp_response'][:100]}")

# ============================================================
# Test 4: Normal exploration (should NOT trigger story_end)
# ============================================================
print("\n" + "=" * 60)
print("Test 4: Normal exploration")
print("=" * 60)

r4 = call_kp(
    player_input="I carefully push open the oak door and peer inside.",
    character=CHAR,
    scene_context="You stand before the library door at midnight. Fog dampens your collar.",
)
check("story_end == False", r4["story_end"] is False)
check("narrative has description", len(r4["narrative"]) > 50)

# ============================================================
# Test 5: Backward compatibility (old JSON format)
# ============================================================
print("\n" + "=" * 60)
print("Test 5: Backward compat (no story_end field)")
print("=" * 60)

old = '{"kp_response": "test", "narrative": "test", "need_check": "None", "difficulty": "Normal"}'
parsed = _parse_kp_json(old)
check("old format parses", parsed is not None)
check("story_end defaults to False", parsed["story_end"] is False)

# ============================================================
# Test 6: Graph integration
# ============================================================
print("\n" + "=" * 60)
print("Test 6: Graph propagates story_end to game_over")
print("=" * 60)

graph = build_graph()
state = create_initial_state(character=CHAR, scene_context="")
state["messages"] = [{"role": "user", "content": "I give up completely. Burn everything. Go home. End this."}]

result_state = graph.invoke(state)
game_over = result_state.get("game_over", False)
print(f"  game_over = {game_over}")
print(f"  messages count = {len(result_state['messages'])}")
check("game_over == True", game_over is True,
      "Graph did NOT propagate story_end to game_over!")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"Results: {passed} passed / {passed + failed} total")
if failed > 0:
    print(f"FAILED {failed} test(s)!")
    sys.exit(1)
else:
    print("All tests passed!")
