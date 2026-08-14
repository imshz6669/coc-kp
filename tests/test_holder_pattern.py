"""
模块级 holder 守卫模式回归测试。

背景 bug：Streamlit 每次整页重跑都会重新执行模块顶层代码。
app/main.py 中 _result_holder / _round_lock 若用普通赋值初始化，
整页重跑会把 holder 重新绑定为空字典、锁换成新对象，导致：
- 工作线程写入的结果在 fragment 触发 st.rerun() 后丢失（前端卡死）
- 工作线程与脚本持有两把不同的锁（跳过/提交竞态失守）

修复模式：`if "xxx" not in globals()` 守卫，已存在则不重建。
本测试以 exec 模拟同一命名空间内的重复执行（等价于整页重跑），
守护该模式不被改回普通赋值。
"""
import sys
import os

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


print("=" * 60)
print("1. 整页重跑后 holder 与锁保持不变")
print("=" * 60)

# 与 app/main.py 模块顶层的守卫模式保持一致
GUARD_CODE = '''
import threading
if "_round_lock" not in globals():
    _round_lock = threading.Lock()
if "_result_holder" not in globals():
    _result_holder = {"result": None, "error": None, "abort": False, "phase": ""}
'''

ns = {}
exec(GUARD_CODE, ns)  # 第一次执行：会话首次加载
lock_first = ns["_round_lock"]
holder_first = ns["_result_holder"]

# 模拟工作线程写入结果与跳过标志
holder_first["result"] = {"messages": ["KEEP"]}
holder_first["abort"] = True
holder_first["phase"] = "KP 思考与渲染"

exec(GUARD_CODE, ns)  # 第二次执行：Streamlit 整页重跑

check("整页重跑后 holder 是同一对象", ns["_result_holder"] is holder_first,
      "holder 被重新绑定")
check("整页重跑后结果未丢失", ns["_result_holder"]["result"] == {"messages": ["KEEP"]},
      str(ns["_result_holder"]))
check("整页重跑后 abort 标志保留", ns["_result_holder"]["abort"] is True)
check("整页重跑后锁是同一对象", ns["_round_lock"] is lock_first,
      "锁被重新创建，线程间互斥失效")


print("=" * 60)
print("2. main.py 实际代码仍使用守卫模式")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
main_path = os.path.join(PROJECT_ROOT, "app", "main.py")
with open(main_path, "r", encoding="utf-8") as f:
    src = f.read()

check("main.py 含 _result_holder 守卫",
      'if "_result_holder" not in globals():' in src,
      "守卫缺失，整页重跑会清空结果")
check("main.py 含 _round_lock 守卫",
      'if "_round_lock" not in globals():' in src,
      "守卫缺失，锁会被重新创建")
check("main.py 的 _result_holder 赋值位于守卫块内",
      'if "_result_holder" not in globals():\n    _result_holder = {' in src,
      "赋值不在守卫块内，整页重跑会清空结果")


print("\n" + "=" * 60)
print(f"Results: {passed} passed / {passed + failed} total")
if failed > 0:
    print(f"FAILED {failed} test(s)!")
    sys.exit(1)
else:
    print("All tests passed!")
