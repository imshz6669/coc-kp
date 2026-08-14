"""
Streamlit 前端 —— COC AI KP 引擎的用户界面。

功能：
    - 侧边栏：角色六维属性、HP/SAN/STA 实时展示 + 低值红色预警
    - 主页面：st.write_stream 流式展示对话
    - 底部：文本输入 + 文件上传 + 重置按钮
    - 状态栏：进行中 / 死亡 / 疯狂
"""

import sys
import os
import uuid
import time
import json
import threading

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from backend.graph import build_graph, create_initial_state, run_one_round
from backend.character import generate_random_character, validate_character
from backend.state_manager import save_game_state, load_game_state, get_temp_status, clear_game_state
from backend.memory import MemoryManager
from rag.loader import create_embedding_model, create_chroma_collection, index_file, clear_collection, parse_file
from rag.retriever import retrieve_context
from data.openings import get_random_opening
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ===================== 页面配置 =====================

st.set_page_config(
    page_title="COC AI 守秘人",
    page_icon="🐙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== 自定义 CSS =====================
# 仅保留 config.toml 主题无法覆盖的部分：
#   1. 聊天气泡「三声部字体系统」（KP 衬线 / 玩家无衬线 / 系统等宽）
#   2. 骰子动画（st.html 内联渲染，见 _show_dice_animation）
# 颜色全部与 .streamlit/config.toml 保持一致，不在此处另行定义基调色。

st.markdown("""
<style>
    /* ========== 聊天气泡：三声部字体 ==========
       KP 叙述 = 衬线（旧书口吻） / 玩家 = 无衬线（现代之口） / 系统与骰子 = 等宽 */
    .chat-message {
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        line-height: 1.8;
        font-size: 0.95rem;
        background: rgba(217, 207, 188, 0.03);
        border-left: 3px solid;
    }
    .chat-player {
        border-left-color: #7D99AE;
        font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    .chat-keeper {
        border-left-color: #C89B5A;
        font-family: 'Noto Serif SC', 'STZhongsong', 'SimSun', serif;
    }
    .chat-system {
        border-left-color: #8A8175;
        font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
        font-size: 0.88rem;
    }
    /* ========== 骰子动画 ========== */
    @keyframes diceRoll {
        0%   { transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg); }
        25%  { transform: rotateX(180deg) rotateY(90deg) rotateZ(45deg); }
        50%  { transform: rotateX(360deg) rotateY(180deg) rotateZ(90deg); }
        75%  { transform: rotateX(540deg) rotateY(270deg) rotateZ(135deg); }
        100% { transform: rotateX(720deg) rotateY(360deg) rotateZ(0deg); }
    }
    @keyframes resultPop {
        0%   { transform: scale(0.3); opacity: 0; }
        60%  { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes diceGlow {
        0%, 100% { box-shadow: 0 0 8px var(--dice); }
        50%      { box-shadow: 0 0 20px var(--dice), 0 0 40px var(--dice); }
    }
    .dice-container {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        margin: 10px 0;
        background: rgba(32, 27, 22, 0.9);
        border-radius: 6px;
        border: 1px solid rgba(217, 207, 188, 0.12);
        animation: diceGlow 2.2s infinite;
        --dice: #7FA05F;
    }
    .dice-container.dice-fail { --dice: #C96A5E; }
    .dice-container.dice-damage { --dice: #C96A5E; }
    .dice-container.dice-san { --dice: #9A86B8; }
    .dice-cube {
        width: 64px;
        height: 64px;
        perspective: 200px;
        flex-shrink: 0;
    }
    .dice-inner {
        width: 64px;
        height: 64px;
        background: #221D17;
        border: 2px solid var(--dice);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: bold;
        color: var(--dice);
        animation: diceRoll 0.8s ease-out;
    }
    .dice-result { animation: resultPop 0.4s ease-out 0.7s both; }
    .dice-info {
        flex: 1;
        color: #D9CFBC;
        font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
    }
    .dice-attr { font-size: 0.85rem; color: #8A8175; margin-bottom: 4px; }
    .dice-values { font-size: 1.1rem; font-weight: bold; }
    .dice-values span:first-child { color: var(--dice); }
    .dice-target { color: #8A8175; font-size: 0.9rem; }
    .dice-outcome { font-size: 1.3rem; font-weight: bold; color: var(--dice); margin-top: 2px; }
    /* ========== 减少动效偏好：动画全部降级为静态 ========== */
    @media (prefers-reduced-motion: reduce) {
        .dice-inner, .dice-container, .dice-result { animation: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ===================== Session State 初始化 =====================

def init_session():
    """
    初始化 Streamlit session state。
    仅在首次加载时执行，不会重复初始化已有会话。
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "character" not in st.session_state:
        st.session_state.character = generate_random_character()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "game_over" not in st.session_state:
        st.session_state.game_over = False

    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()

    if "embedding_model" not in st.session_state:
        try:
            st.session_state.embedding_model = create_embedding_model()
        except Exception as e:
            st.warning(f"Embedding 模型加载失败，RAG 功能暂不可用: {e}", icon=":material/warning:")
            st.session_state.embedding_model = None

    if "rag_collection" not in st.session_state:
        st.session_state.rag_collection = None

    if "langgraph_state" not in st.session_state:
        state = create_initial_state(character=st.session_state.character)
        # 立即注入开场白，确保第一轮 KP 能正确回应
        scene, goal = get_random_opening()
        state["scene_context"] = f"{scene}\n\n【你的目标】{goal}"
        st.session_state.langgraph_state = state

    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager(
            session_id=st.session_state.session_id,
        )
        logger.info(f"MemoryManager 初始化: {st.session_state.session_id[:8]}...")

    if "opening_narrative" not in st.session_state:
        scene, goal = get_random_opening()
        st.session_state.opening_narrative = scene
        st.session_state.story_goal = goal

    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = []

    # 处理中标志 —— 防止输入堆积
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 初始化完成标志
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

    # 当前场景位置（随着游戏推进更新）
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = ""


# ===================== 侧边栏 =====================

def render_sidebar():
    """
    渲染侧边栏：角色面板 + 知识库上传 + 重置按钮。
    全部使用原生组件（metric / progress / badge / expander），
    视觉由 .streamlit/config.toml 主题统一控制。
    """
    char = st.session_state.character

    with st.sidebar:
        st.title("🐙 COC 守秘人")
        st.caption(f"会话 {st.session_state.session_id[:8]}")

        # ---- 状态徽章 + 角色名 ----
        status = char.get("status", "alive")
        if status == "dead":
            st.badge("已死亡", icon=":material/skull:", color="red")
        elif status == "insane":
            st.badge("永久疯狂", icon=":material/cyclone:", color="violet")
        else:
            st.badge("调查进行中", icon=":material/mystery:", color="green")
        st.markdown(f"### {char.get('name', '调查员')}")

        # ---- 六维属性：2 列角色卡 ----
        attr_labels = {
            "STR": "力量 STR",
            "DEX": "敏捷 DEX",
            "POW": "感知 POW",
            "WILL": "意志 WILL",
            "INT": "智力 INT",
            "IDEA": "灵感 IDEA",
        }
        for row_keys in [("STR", "DEX"), ("POW", "WILL"), ("INT", "IDEA")]:
            cols = st.columns(2, gap="small")
            for col, key in zip(cols, row_keys):
                col.metric(label=attr_labels[key], value=char.get(key, 0), border=True)

        st.space("small")

        # ---- 派生属性 ----
        max_hp = char.get("MAX_HP", 10)
        max_san = char.get("MAX_SAN", 50)
        max_sta = char.get("MAX_STA", 50)
        hp_ratio = char["HP"] / max_hp if max_hp > 0 else 0
        san_ratio = char["SAN"] / max_san if max_san > 0 else 0

        hp_text = f"HP **{char['HP']}/{max_hp}**"
        san_text = f"SAN **{char['SAN']}/{max_san}**"
        if hp_ratio < 0.3:
            hp_text = f":red[HP {char['HP']}/{max_hp}]"
        if san_ratio < 0.3:
            san_text = f":red[SAN {char['SAN']}/{max_san}]"
        st.progress(hp_ratio, text=hp_text)
        st.progress(san_ratio, text=san_text)
        st.markdown(f"STA {char.get('STA', 0)}/{max_sta}")

        if char.get("wounded"):
            st.warning("重伤状态：HP 低于 30%", icon=":material/medical_information:")
        if char.get("temp_insanity"):
            st.warning("临时疯狂：行为可能不受控制", icon=":material/cyclone:")

        st.space("medium")

        # ---- 知识库：规则书上传 ----
        with st.expander("规则参考", icon=":material/menu_book:"):
            rule_files = st.file_uploader(
                "上传规则书、怪物图鉴或法术手册（TXT / PDF）",
                type=["txt", "pdf"],
                key="rule_uploader",
                accept_multiple_files=True,
                help="规则书仅作为 KP 的参考知识，不会改变当前剧情。",
                disabled=st.session_state.processing,
            )
            if rule_files:
                _handle_rule_uploads(rule_files)

        # ---- 知识库：剧本上传 ----
        with st.expander("剧本模组", icon=":material/description:"):
            scenario_files = st.file_uploader(
                "上传剧本模组（TXT / PDF），上传后自动生成对应开场剧情",
                type=["txt", "pdf"],
                key="scenario_uploader",
                accept_multiple_files=True,
                help="上传剧本后 KP 会根据剧本内容重新生成开场白和故事目标。",
                disabled=st.session_state.processing,
            )
            if scenario_files:
                _handle_scenario_uploads(scenario_files)

        # 索引状态
        rule_count = len(st.session_state.get("indexed_rule_names", set()))
        scenario_count = len(st.session_state.get("indexed_scenario_names", set()))
        if rule_count > 0:
            st.caption(f":green[已加载 {rule_count} 个规则文件]")
        if scenario_count > 0:
            st.caption(f":green[已加载 {scenario_count} 个剧本文件]")
        if rule_count == 0 and scenario_count == 0:
            st.caption("未加载知识库（可选）")

        st.space("medium")

        # ---- 当前进度 ----
        turn_count = len([m for m in st.session_state.get("messages", []) if m.get("role") == "user"])
        if turn_count > 0:
            st.caption(f"回合数 {turn_count}")
            current_scene = st.session_state.get("current_scene", "")
            if current_scene:
                st.caption(f"位置 {current_scene}")

        st.space("small")

        # ---- 重置按钮 ----
        if st.button("重置游戏", icon=":material/restart_alt:", type="primary",
                     width="stretch", disabled=st.session_state.processing):
            _reset_game()

        st.caption("LangGraph + DeepSeek")


def _index_files(uploaded_files: list, file_type: str) -> list:
    """
    通用文件索引入库函数。返回新索引文件的文本内容列表（供开场生成）。
    """
    import tempfile

    session_id = st.session_state.session_id

    if file_type == "rule":
        tracked_key = "indexed_rule_names"
    else:
        tracked_key = "indexed_scenario_names"

    if tracked_key not in st.session_state:
        st.session_state[tracked_key] = set()

    if st.session_state.rag_collection is None:
        try:
            st.session_state.rag_collection = create_chroma_collection(session_id)
        except Exception as e:
            st.error(f"创建知识库失败: {e}")
            return []

    total_chunks = 0
    new_contents = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state[tracked_key]:
            continue

        tmp_path = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                raw_text = parse_file(tmp_path)
                if raw_text:
                    new_contents.append(raw_text)
            except Exception:
                pass

            count = index_file(
                file_path=tmp_path,
                session_id=session_id,
                model=st.session_state.embedding_model,
                collection=st.session_state.rag_collection,
            )
            total_chunks += count
            st.session_state[tracked_key].add(uploaded_file.name)

        except Exception as e:
            st.error(f"文件 '{uploaded_file.name}' 处理失败: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if total_chunks > 0:
        st.toast(f"已索引 {total_chunks} 个文本块 ({file_type})", icon="📚")

    return new_contents


def _handle_rule_uploads(uploaded_files: list):
    """处理规则书上传：仅入库，不改变剧情。"""
    _index_files(uploaded_files, "rule")


def _handle_scenario_uploads(uploaded_files: list):
    """处理剧本上传：入库 + 触发开场白重新生成。"""
    new_contents = _index_files(uploaded_files, "scenario")

    if new_contents:
        combined = "\n\n---\n\n".join(c[:1500] for c in new_contents[:3])
        st.session_state.pending_opening_content = combined[:3000]


def _reset_game():
    """重置游戏。"""
    session_id = st.session_state.session_id

    if "memory_manager" in st.session_state:
        st.session_state.memory_manager.clear()

    if st.session_state.rag_collection is not None:
        clear_collection(session_id)
        st.session_state.rag_collection = None
    st.session_state.indexed_rule_names = set()
    st.session_state.indexed_scenario_names = set()
    st.session_state.pop("pending_opening_content", None)

    new_character = generate_random_character()
    new_session_id = str(uuid.uuid4())

    scene, goal = get_random_opening()

    st.session_state.character = new_character
    st.session_state.messages = []
    st.session_state.game_over = False
    st.session_state.processing = False
    st.session_state.session_id = new_session_id
    st.session_state.opening_narrative = scene
    st.session_state.story_goal = goal
    st.session_state.current_suggestions = []
    st.session_state.current_scene = ""

    st.session_state.langgraph_state = create_initial_state(
        character=new_character,
        scene_context=f"{scene}\n\n【你的目标】{goal}",
    )

    st.session_state.memory_manager = MemoryManager(session_id=new_session_id)

    st.rerun()


# ===================== 主区域 =====================

def render_main():
    """渲染主区域：对话历史 + 输入区域。"""
    st.title("克苏鲁的呼唤")
    st.caption("你是一名调查员。黑暗中有东西在蠕动……你准备好面对真相了吗？")

    # ---- 处理待生成的开场白（剧本上传后触发） ----
    pending = st.session_state.pop("pending_opening_content", None)
    if pending:
        with st.status("KP 正在根据剧本生成开场剧情", expanded=False) as opening_status:
            try:
                _generate_opening_from_upload(pending, st.session_state.session_id)
                opening_status.update(label="开场剧情已生成", state="complete")
            except Exception as e:
                logger.warning(f"开场生成失败: {e}")
                _fallback_opening_from_text(pending)
                opening_status.update(label="已使用剧本原文生成开场", state="complete")
        st.rerun()

    st.space("small")

    # ---- 对话历史 ----
    _render_chat_history()

    st.space("medium")

    # ---- 输入区域 ----
    _render_input_area()


def _render_chat_history():
    """渲染对话历史。"""
    messages = st.session_state.messages

    if not messages:
        opening = st.session_state.get("opening_narrative", "")
        goal = st.session_state.get("story_goal", "")
        if not opening:
            scene, goal = get_random_opening()
            st.session_state.opening_narrative = scene
            st.session_state.story_goal = goal
            opening = scene

        with st.chat_message("assistant", avatar="🐙"):
            st.markdown(
                f'<div class="chat-message chat-keeper">'
                f'<strong>🐙 KP：</strong><br>{opening}'
                f'</div>',
                unsafe_allow_html=True,
            )
        if goal:
            st.info(f"你的目标：{goal}", icon=":material/track_changes:")

        st.caption("你打算怎么做？")
        return

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user", avatar="🎭"):
                st.markdown(
                    f'<div class="chat-message chat-player">'
                    f'<strong>🎭 调查员：</strong><br>{content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif role == "system":
            with st.chat_message("assistant", avatar="🎲"):
                st.markdown(
                    f'<div class="chat-message chat-system">'
                    f'{content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif role == "assistant":
            if "[KP梗概]" in content or "[KP回应]" in content:
                continue
            with st.chat_message("assistant", avatar="🐙"):
                st.markdown(
                    f'<div class="chat-message chat-keeper">'
                    f'<strong>🐙 KP：</strong><br>{content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _generate_opening_from_upload(content: str, session_id: str):
    """使用 KP 读取上传剧本内容，生成符合 COC 风格的沉浸式开场白。"""
    from backend.agents import call_kp

    reference_text = content[:2000]
    char = st.session_state.character

    result = call_kp(
        player_input=(
            "请根据以下剧本/规则书内容，生成一段 COC 跑团的沉浸式开场白。"
            "开场白应当：\n"
            "1. 以第二人称「你」叙述\n"
            "2. 从参考内容中提取关键场景、时代背景和核心冲突\n"
            "3. 营造克苏鲁式的悬疑氛围\n"
            "4. 以开放式引导结尾，让玩家知道该做什么\n"
            "5. 故事目标要具体（如「找到失踪的XX」「揭开XX的秘密」），不要泛泛而谈\n"
            "6. 如果参考内容是规则书而非剧本，就从规则书中提取一个典型场景作为开场"
        ),
        character=char,
        rag_context=f"【用户上传的剧本/规则书内容】\n{reference_text}",
        scene_context="你正在为一场 COC 跑团游戏生成开场白，请基于上传的剧本内容创作。",
    )

    opening = result.get("narrative", "")
    kp_response = result.get("kp_response", "")

    if opening:
        st.session_state.opening_narrative = opening
        if kp_response and len(kp_response) > 10:
            st.session_state.story_goal = kp_response
        else:
            st.session_state.story_goal = "根据上传的剧本内容自由探索"
        st.session_state.current_suggestions = result.get("suggestions", [])
        st.session_state.messages = []

        scene_ctx = f"【当前剧本背景】\n{reference_text}"
        if st.session_state.get("langgraph_state"):
            st.session_state.langgraph_state["scene_context"] = scene_ctx

        logger.info("KP 已根据上传内容生成新开场白")


def _fallback_opening_from_text(content: str):
    """降级方案：当 KP 调用失败时，从上传文本中智能提取场景信息。"""
    raw = content or ""

    skip_patterns = ["剧本概述", "时代背景", "适合人数", "预计时长",
                     "难度", "主题", "第一部分", "====", "----"]
    lines = raw.split("\n")
    narrative_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(p in stripped for p in skip_patterns):
            continue
        narrative_lines.append(stripped)
        if len("".join(narrative_lines)) > 300:
            break

    text = "".join(narrative_lines)[:400] if narrative_lines else raw[:400]
    st.session_state.opening_narrative = (
        f"夜色笼罩着阿卡姆，一场新的冒险即将开始。\n\n"
        f"<em>{text}……</em>"
    )
    st.session_state.story_goal = "根据上传的剧本/规则书自由探索"
    st.session_state.current_suggestions = []
    logger.info("使用降级方案更新开场白（KP 调用失败）")


def _render_input_area():
    """渲染底部输入区域。处理中时禁用输入防止堆积。"""
    if st.session_state.game_over:
        hp = st.session_state.character.get("HP", 0)
        san = st.session_state.character.get("SAN", 0)
        if hp <= 0:
            st.error("调查员已死亡。点击侧边栏「重置游戏」开始新的冒险。", icon=":material/skull:")
        elif san <= 0:
            st.error("调查员陷入永久疯狂。点击侧边栏「重置游戏」开始新的冒险。", icon=":material/cyclone:")
        else:
            st.warning("故事落幕。点击侧边栏「重置游戏」开启一段新的故事。", icon=":material/auto_stories:")
        return

    # 处理中状态：原生 st.status + 超时跳过按钮
    if st.session_state.processing:
        elapsed = int(time.time() - st.session_state.get("_processing_start", time.time()))
        with st.status(f"KP 正在编织命运（已等待 {elapsed}s）", state="running"):
            # 超过 15 秒后显示跳过按钮
            if elapsed >= 15:
                if st.button("跳过等待", key=f"skip_{elapsed}", icon=":material/skip_next:",
                             help="中断当前请求，KP 会使用降级回复"):
                    st.session_state.processing = False
                    st.session_state.pop("_processing_start", None)
                    # 注入降级回复
                    msgs = list(st.session_state.messages)
                    msgs.append({
                        "role": "system",
                        "content": "KP 响应超时，已被跳过。",
                    })
                    msgs.append({
                        "role": "assistant",
                        "content": "你环顾四周，之前的行动似乎暂时没有结果。也许换个方式或方向会有新的发现。黑暗中有什么东西正在耐心地等待，它不急。你决定继续前行。",
                    })
                    st.session_state.messages = msgs
                    st.session_state.current_suggestions = [
                        "换个角度重新观察", "换个方向继续调查", "找附近的人打听消息"
                    ]
                    st.rerun()
                st.caption("KP 正在生成回复，请耐心等待，若超过 30 秒建议跳过")
        # 渲染建议（处理中禁用）
        suggestions = st.session_state.get("current_suggestions", [])
        if suggestions:
            cols = st.columns(len(suggestions))
            for i, (col, sug) in enumerate(zip(cols, suggestions)):
                col.button(sug, key=f"sug_disabled_{i}", disabled=True, width="stretch")
        return

    # 正常输入
    player_input = st.chat_input(
        placeholder="描述你的行动，例如：我小心翼翼地推开图书馆的门",
        disabled=st.session_state.processing,
    )

    if player_input:
        _process_player_input(player_input)

    # 渲染行动建议按钮（非处理中状态）
    suggestions = st.session_state.get("current_suggestions", [])
    if suggestions:
        cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(cols, suggestions)):
            btn_key = f"sug_{hash(sug)}_{st.session_state.session_id[:6]}_{i}"
            if col.button(sug, key=btn_key, width="stretch"):
                _process_player_input(sug)


def _process_player_input(player_input: str):
    """
    处理玩家输入：RAG 检索 → LangGraph 调用 → 流式渲染结果。
    设置 processing 标志防止输入堆积。
    """
    # 防止重复处理
    if st.session_state.processing:
        st.warning("KP 正在处理上一轮请求，请稍候……")
        return

    st.session_state.processing = True
    st.session_state._processing_start = time.time()

    try:
        # ---- RAG 检索 ----
        rag_context = ""
        if st.session_state.rag_collection is not None and st.session_state.embedding_model is not None:
            try:
                rag_context = retrieve_context(
                    query=player_input,
                    session_id=st.session_state.session_id,
                    model=st.session_state.embedding_model,
                )
            except Exception as e:
                logger.warning(f"RAG 检索异常: {e}")

        # ---- 构建状态 ----
        state = st.session_state.langgraph_state
        state["rag_context"] = rag_context
        state["character"] = st.session_state.character

        # ---- 调用 LangGraph ----
        t0 = time.time()
        try:
            new_state = run_one_round(
                graph=st.session_state.graph,
                state=state,
                player_input=player_input,
                memory_manager=st.session_state.memory_manager,
            )
        except Exception as e:
            st.error(f"系统异常: {e}")
            logger.error(f"Graph 调用异常: {e}")
            return

        elapsed = time.time() - t0
        logger.info(f"本轮处理耗时: {elapsed:.1f}s")

        # ---- 更新 Session State ----
        st.session_state.langgraph_state = new_state
        st.session_state.character = new_state.get("character", st.session_state.character)
        st.session_state.game_over = new_state.get("game_over", False)
        st.session_state.messages = new_state.get("messages", [])

        # ---- 更新当前场景（直接使用 KP 输出的 scene 字段） ----
        kp_scene = new_state.get("current_scene", "")
        if kp_scene:
            st.session_state.current_scene = kp_scene

        # ---- 更新当前轮建议 ----
        st.session_state.current_suggestions = new_state.get("suggestions", [])

        # ---- 骰子动画（有检定时展示） ----
        _show_dice_animation(new_state)

        # ---- 流式渲染最新输出 ----
        _stream_render(new_state)

        # ---- 游戏结束通知 ----
        if st.session_state.game_over:
            hp = st.session_state.character.get("HP", 0)
            san = st.session_state.character.get("SAN", 0)
            if hp <= 0:
                st.toast("调查员已死亡", icon=":material/skull:")
                st.error(
                    "## 调查员已死亡\n\n"
                    "世界将永远不知道这里发生了什么……\n\n"
                    "点击侧边栏「重置游戏」开始新的冒险。",
                    icon=":material/skull:",
                )
            elif san <= 0:
                st.toast("调查员陷入永久疯狂", icon=":material/cyclone:")
                st.error(
                    "## 调查员陷入永久疯狂\n\n"
                    "理智的最后一根弦，已经断了。\n\n"
                    "点击侧边栏「重置游戏」开始新的冒险。",
                    icon=":material/cyclone:",
                )
            else:
                st.toast("故事落幕", icon=":material/auto_stories:")
                st.warning(
                    "## 故事落幕\n\n"
                    "这段冒险就此画上句号。无论结局是平静还是遗憾，"
                    "那些无法言说的秘密将永远封存在记忆深处……\n\n"
                    "点击侧边栏「重置游戏」开启一段新的故事。",
                    icon=":material/auto_stories:",
                )

    finally:
        st.session_state.processing = False
        st.session_state.pop("_processing_start", None)
        st.rerun()


def _show_dice_animation(state: dict):
    """
    有事件发生时展示骰子动画面板（CSS 3D 骰子滚动效果）。
    从最近一条 system 消息中识别事件类型：
        - 属性检定（成功绿 / 失败红）
        - 受到伤害（血红）
        - 理智损失（灰紫）
    一轮可能同时发生多个事件，逐类渲染。
    样式定义在页面顶部的全局 CSS 中，此处用 st.html 内联渲染结构。
    """
    messages = state.get("messages", [])
    # 查找最近一条 system 消息（render_node 合并后的本轮结果）
    system_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "system":
            system_msg = msg.get("content", "")
            break

    if not system_msg:
        return

    import re
    panels = []

    # ---- 1. 属性检定面板 ----
    check_match = re.search(r'【(\S+?)检定', system_msg)
    if check_match:
        attr_name = check_match.group(1)
        roll_match = re.search(r'掷出[:：]\s*(\d{1,3})', system_msg)
        target_match = re.search(r'属性值[:：]\s*(\d{1,3})', system_msg)
        dice_roll = roll_match.group(1) if roll_match else "?"
        dice_target = target_match.group(1) if target_match else "?"

        is_success = any(w in system_msg for w in ["成功", "通过", "大成功"])
        result_emoji = "🎉" if is_success else "💥"
        result_text = "成功" if is_success else ("大失败…" if "大失败" in system_msg else "失败")
        fail_class = "" if is_success else " dice-fail"

        panels.append(f"""
    <div class="dice-container{fail_class}">
        <div class="dice-cube">
            <div class="dice-inner">{result_emoji}</div>
        </div>
        <div class="dice-info dice-result">
            <div class="dice-attr">🎲 {attr_name}检定</div>
            <div class="dice-values">
                <span>{dice_roll}</span>
                <span class="dice-target"> / 目标 {dice_target}</span>
            </div>
            <div class="dice-outcome">{result_text}</div>
        </div>
    </div>""")

    # ---- 2. 受到伤害面板 ----
    dmg_match = re.search(r'损失\s*(\d+)\s*点\s*HP', system_msg)
    if dmg_match:
        damage = dmg_match.group(1)
        hp_match = re.search(r'（(\d+)\s*→\s*(\d+)\s*/\s*(\d+)）', system_msg)
        hp_old, hp_new, hp_max = hp_match.groups() if hp_match else ("?", "?", "?")
        hp_line = f"{hp_old} → {hp_new}/{hp_max}" if hp_match else ""
        outcome = f"生命值 {hp_new}/{hp_max}" if hp_match else "受伤"
        panels.append(f"""
    <div class="dice-container dice-damage">
        <div class="dice-cube">
            <div class="dice-inner">⚔️</div>
        </div>
        <div class="dice-info dice-result">
            <div class="dice-attr">⚔️ 受到伤害</div>
            <div class="dice-values">
                <span>{damage}</span>
                <span class="dice-target"> 点 HP {hp_line}</span>
            </div>
            <div class="dice-outcome">{outcome}</div>
        </div>
    </div>""")

    # ---- 3. 理智损失面板 ----
    san_match = re.search(r'丧失\s*(\d+)\s*点\s*SAN', system_msg)
    if san_match:
        san_loss = san_match.group(1)
        cur_match = re.search(r'当前\s*SAN[:：]\s*(\d+)', system_msg)
        san_now = cur_match.group(1) if cur_match else "?"
        panels.append(f"""
    <div class="dice-container dice-san">
        <div class="dice-cube">
            <div class="dice-inner">🧠</div>
        </div>
        <div class="dice-info dice-result">
            <div class="dice-attr">🧠 理智损失</div>
            <div class="dice-values">
                <span>{san_loss}</span>
                <span class="dice-target"> 点 SAN</span>
            </div>
            <div class="dice-outcome">当前 {san_now}</div>
        </div>
    </div>""")

    if not panels:
        return

    st.html("\n".join(panels))


def _typewriter(placeholder, css_class: str, text: str, prefix: str = "",
                chunk_size: int = 8, delay: float = 0.02):
    """
    分块打字机效果：每次追加 chunk_size 个字符重绘一次。
    比逐字符渲染更省 Streamlit 往返，同时保留逐字浮现的叙述感。
    """
    displayed = ""
    for i in range(0, len(text), chunk_size):
        displayed += text[i:i + chunk_size]
        placeholder.markdown(
            f'<div class="chat-message {css_class}">{prefix}{displayed}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(delay)


def _stream_render(state: dict):
    """
    流式展示本轮新增的输出（不重复渲染历史消息）。
    先游戏层面回应（等宽），再 KP 环境叙述（衬线）。
    """
    messages = state.get("messages", [])

    system_msg = ""
    render_msg = ""

    for msg in reversed(messages):
        content = msg.get("content", "")
        role = msg.get("role", "")

        if role == "assistant":
            if "[KP梗概]" in content or "[KP回应]" in content:
                continue
            if not render_msg:
                render_msg = content
        elif role == "system" and not system_msg:
            system_msg = content

        if system_msg and render_msg:
            break

    # 先输出游戏层面回应（等宽字体，较快速度）
    if system_msg:
        with st.chat_message("assistant", avatar="🎲"):
            _typewriter(st.empty(), "chat-system", system_msg, chunk_size=12)

    # 再输出 KP 环境叙述（衬线字体）
    if render_msg:
        with st.chat_message("assistant", avatar="🐙"):
            _typewriter(st.empty(), "chat-keeper", render_msg,
                        prefix="<strong>🐙 KP：</strong><br>")


# ===================== 主入口 =====================

def main():
    """Streamlit 应用主入口。"""
    init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
