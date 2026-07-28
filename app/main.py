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

st.markdown("""
<style>
    /* ========== 聊天气泡 ========== */
    .chat-message {
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        line-height: 1.75;
        font-size: 0.95rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        transition: box-shadow 0.2s ease;
    }
    .chat-message:hover {
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    /* 调查员消息 —— 冷调蓝 */
    .chat-player {
        background: rgba(74, 158, 255, 0.07);
        border-left: 4px solid rgba(74, 158, 255, 0.7);
    }
    /* KP 消息 —— 暖紫 */
    .chat-keeper {
        background: rgba(192, 132, 252, 0.07);
        border-left: 4px solid rgba(192, 132, 252, 0.7);
    }
    /* 系统 / KP 提示消息 —— 冷灰调信息条 */
    .chat-system {
        background: rgba(160, 175, 190, 0.08);
        border-left: 4px solid rgba(140, 155, 170, 0.65);
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        font-size: 0.9rem;
    }
    /* ========== 状态徽章 ========== */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-alive { background: #166534; color: #4ade80; }
    .status-dead  { background: #7f1d1d; color: #fca5a5; }
    .status-insane { background: #4a1d7f; color: #c084fc; }
    /* ========== 侧边栏属性面板 ========== */
    .attr-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    .attr-label { color: #9ca3af; }
    .attr-value { font-weight: bold; }
    .attr-danger { color: #ef4444 !important; }
    /* ========== 行动建议按钮 ========== */
    .suggestion-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 0.5rem 0;
    }
    .suggestion-btn {
        flex: 1;
        min-width: 120px;
        padding: 0.5rem 0.8rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.04);
        color: inherit;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.15s ease;
        text-align: center;
    }
    .suggestion-btn:hover {
        background: rgba(192, 132, 252, 0.12);
        border-color: rgba(192, 132, 252, 0.5);
    }
    /* ========== 处理中状态 ========== */
    .processing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        background: rgba(192, 132, 252, 0.1);
        border: 1px solid rgba(192, 132, 252, 0.3);
        border-radius: 8px;
        font-size: 0.9rem;
        color: #c084fc;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
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
            st.warning(f"⚠️ Embedding 模型加载失败，RAG 功能暂不可用: {e}")
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

    # 初始化完成标志 —— 确保首轮不会吞输入
    if "initialized" not in st.session_state:
        st.session_state.initialized = True


# ===================== 侧边栏 =====================

def render_sidebar():
    """
    渲染侧边栏：角色属性面板 + 文件上传 + 重置按钮。
    """
    char = st.session_state.character

    with st.sidebar:
        st.title("🐙 COC 守秘人")
        st.caption(f"会话: {st.session_state.session_id[:8]}...")

        st.divider()

        # ---- 角色信息 ----
        st.subheader(f"📋 {char.get('name', '调查员')}")

        # 六维属性
        st.markdown("**基础属性**")
        attrs = [
            ("💪 力量 STR", "STR"),
            ("🏃 敏捷 DEX", "DEX"),
            ("🔮 感知 POW", "POW"),
            ("🧠 意志 WILL", "WILL"),
            ("📚 智力 INT", "INT"),
            ("💡 灵感 IDEA", "IDEA"),
        ]
        for label, key in attrs:
            val = char.get(key, 0)
            st.markdown(
                f'<div class="attr-row"><span class="attr-label">{label}</span>'
                f'<span class="attr-value">{val}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # 派生属性
        st.markdown("**派生属性**")
        max_hp = char.get("MAX_HP", 10)
        max_san = char.get("MAX_SAN", 50)
        max_sta = char.get("MAX_STA", 50)
        hp_ratio = char["HP"] / max_hp if max_hp > 0 else 0
        san_ratio = char["SAN"] / max_san if max_san > 0 else 0

        # HP
        hp_class = "attr-danger" if hp_ratio < 0.3 else ""
        st.markdown(
            f'<div class="attr-row"><span>❤️ HP</span>'
            f'<span class="attr-value {hp_class}">{char["HP"]}/{max_hp}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(hp_ratio)

        # SAN
        san_class = "attr-danger" if san_ratio < 0.3 else ""
        st.markdown(
            f'<div class="attr-row"><span>🧠 SAN</span>'
            f'<span class="attr-value {san_class}">{char["SAN"]}/{max_san}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(san_ratio)

        # STA
        st.markdown(
            f'<div class="attr-row"><span>⚡ STA</span>'
            f'<span class="attr-value">{char.get("STA", 0)}/{max_sta}</span></div>',
            unsafe_allow_html=True,
        )

        # 状态标签
        status = char.get("status", "alive")
        if status == "dead":
            badge = '<span class="status-badge status-dead">💀 已死亡</span>'
        elif status == "insane":
            badge = '<span class="status-badge status-insane">🌀 已疯狂</span>'
        else:
            badge = '<span class="status-badge status-alive">✅ 进行中</span>'
        st.markdown(badge, unsafe_allow_html=True)

        if char.get("wounded"):
            st.warning("⚠️ 重伤状态 —— HP 低于 30%")
        if char.get("temp_insanity"):
            st.warning("🌀 临时疯狂 —— 行为可能不受控制")

        st.divider()

        # ---- 知识库：规则书上传 ----
        st.subheader("📖 规则参考")
        rule_files = st.file_uploader(
            "上传规则书 / 怪物图鉴 / 法术手册（TXT / PDF）",
            type=["txt", "pdf"],
            key="rule_uploader",
            accept_multiple_files=True,
            help="规则书仅作为 KP 的参考知识，不会改变当前剧情。",
            disabled=st.session_state.processing,
        )
        if rule_files:
            _handle_rule_uploads(rule_files)

        st.divider()

        # ---- 知识库：剧本上传 ----
        st.subheader("📜 剧本模组")
        scenario_files = st.file_uploader(
            "上传剧本模组（TXT / PDF）—— 上传后将自动生成对应开场剧情",
            type=["txt", "pdf"],
            key="scenario_uploader",
            accept_multiple_files=True,
            help="上传剧本后 KP 会根据剧本内容重新生成开场白和故事目标。",
            disabled=st.session_state.processing,
        )
        if scenario_files:
            _handle_scenario_uploads(scenario_files)

        # 显示索引状态
        rule_count = len(st.session_state.get("indexed_rule_names", set()))
        scenario_count = len(st.session_state.get("indexed_scenario_names", set()))
        if rule_count > 0:
            st.success(f"📖 已加载 {rule_count} 个规则文件")
        if scenario_count > 0:
            st.success(f"📜 已加载 {scenario_count} 个剧本文件")
        if rule_count == 0 and scenario_count == 0:
            st.caption("未加载知识库（可选）")

        st.divider()

        # ---- 重置按钮 ----
        if st.button("🔄 重置游戏", use_container_width=True, type="primary",
                     disabled=st.session_state.processing):
            _reset_game()

        st.divider()
        st.caption("Powered by LangGraph + DeepSeek")


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

    st.session_state.langgraph_state = create_initial_state(
        character=new_character,
        scene_context=f"{scene}\n\n【你的目标】{goal}",
    )

    st.session_state.memory_manager = MemoryManager(session_id=new_session_id)

    st.rerun()


# ===================== 主区域 =====================

def render_main():
    """渲染主区域：对话历史 + 输入区域。"""
    st.title("🐙 克苏鲁的呼唤 · AI 守秘人")
    st.caption("你是一名调查员。黑暗中有东西在蠕动……你准备好面对真相了吗？")

    # ---- 处理待生成的开场白（剧本上传后触发） ----
    pending = st.session_state.pop("pending_opening_content", None)
    if pending:
        with st.spinner("🐙 KP 正在根据剧本生成开场剧情……"):
            try:
                _generate_opening_from_upload(pending, st.session_state.session_id)
            except Exception as e:
                logger.warning(f"开场生成失败: {e}")
                _fallback_opening_from_text(pending)
        st.rerun()

    st.divider()

    # ---- 对话历史 ----
    _render_chat_history()

    st.divider()

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
            st.info(f"🎯 **你的目标：{goal}**")

        st.caption("——你打算怎么做？")
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
        f"夜色笼罩着阿卡姆，一场新的冒险即将开始——\n\n"
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
            st.error("💀 **游戏结束 —— 调查员已死亡。** 点击侧边栏「🔄 重置游戏」开始新的冒险。")
        elif san <= 0:
            st.error("🌀 **游戏结束 —— 调查员陷入永久疯狂。** 点击侧边栏「🔄 重置游戏」开始新的冒险。")
        else:
            st.warning("📖 **故事落幕。** 点击侧边栏「🔄 重置游戏」开启一段新的故事。")
        return

    # 处理中状态：显示进度指示，禁用输入
    if st.session_state.processing:
        elapsed = int(time.time() - st.session_state.get("_processing_start", time.time()))
        st.markdown(
            f'<div class="processing-indicator">'
            f'🐙 <strong>KP 正在编织命运……</strong>'
            f'<span style="margin-left:auto;opacity:0.7">已等待 {elapsed}s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # 渲染当前场景的建议，但不允许交互
        suggestions = st.session_state.get("current_suggestions", [])
        if suggestions:
            st.caption("💡 **你可以这样做：**")
            cols = st.columns(len(suggestions))
            for i, (col, sug) in enumerate(zip(cols, suggestions)):
                col.button(sug, key=f"sug_disabled_{i}", disabled=True, use_container_width=True)
        return

    # 正常输入
    player_input = st.chat_input(
        placeholder="描述你的行动……（例如：我小心翼翼地推开图书馆的门）",
        disabled=st.session_state.processing,
    )

    if player_input:
        _process_player_input(player_input)

    # 渲染行动建议按钮（非处理中状态）
    suggestions = st.session_state.get("current_suggestions", [])
    if suggestions:
        st.caption("💡 **你可以这样做：**")
        cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(cols, suggestions)):
            btn_key = f"sug_{hash(sug)}_{st.session_state.session_id[:6]}_{i}"
            if col.button(sug, key=btn_key, use_container_width=True):
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

        # ---- 更新当前场景的建议（确保跟随最新场景） ----
        st.session_state.current_suggestions = new_state.get("suggestions", [])

        # ---- 流式渲染最新输出 ----
        _stream_render(new_state)

        # ---- 游戏结束通知 ----
        if st.session_state.game_over:
            hp = st.session_state.character.get("HP", 0)
            san = st.session_state.character.get("SAN", 0)
            if hp <= 0:
                st.toast("💀 调查员已死亡", icon="💀")
                st.error(
                    "## 💀 游戏结束 —— 调查员已死亡\n\n"
                    "世界将永远不知道这里发生了什么……\n\n"
                    "点击侧边栏「🔄 重置游戏」开始新的冒险。"
                )
            elif san <= 0:
                st.toast("🌀 调查员陷入永久疯狂", icon="🌀")
                st.error(
                    "## 🌀 游戏结束 —— 调查员陷入永久疯狂\n\n"
                    "理智的最后一根弦，已经断了。\n\n"
                    "点击侧边栏「🔄 重置游戏」开始新的冒险。"
                )
            else:
                st.toast("📖 故事落幕", icon="📖")
                st.warning(
                    "## 📖 故事落幕\n\n"
                    "这段冒险就此画上句号。无论结局是平静还是遗憾，"
                    "那些无法言说的秘密将永远封存在记忆深处……\n\n"
                    "点击侧边栏「🔄 重置游戏」开启一段新的故事。"
                )

    finally:
        st.session_state.processing = False
        st.session_state.pop("_processing_start", None)
        st.rerun()


def _stream_render(state: dict):
    """
    流式展示本轮新增的输出（不重复渲染历史消息）。
    使用更快的字符间隔以减少等待感。
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

    # 先输出游戏层面回应（较快速度）
    if system_msg:
        with st.chat_message("assistant", avatar="🎲"):
            placeholder = st.empty()
            displayed = ""
            delay = 0.003 if len(system_msg) > 100 else 0.005  # 长文本快一点
            for char in system_msg:
                displayed += char
                placeholder.markdown(
                    f'<div class="chat-message chat-system">{displayed}</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(delay)

    # 再输出 KP 环境叙述
    if render_msg:
        with st.chat_message("assistant", avatar="🐙"):
            placeholder = st.empty()
            displayed = ""
            delay = 0.004 if len(render_msg) > 150 else 0.006
            for char in render_msg:
                displayed += char
                placeholder.markdown(
                    f'<div class="chat-message chat-keeper">{displayed}</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(delay)


# ===================== 主入口 =====================

def main():
    """Streamlit 应用主入口。"""
    init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
