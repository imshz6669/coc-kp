"""
Agent 定义与模型调用模块 —— KP 系统提示词、渲染提示词、LLM 调用封装。

使用 OpenAI 兼容 SDK 调用 DeepSeek API。
自带超时保护、自动重试、智能模型降级。
"""

import json
import time
from typing import Any, Dict, Generator

from openai import OpenAI
from utils.config import (
    get_openai_client,
    API_TIMEOUT_SECONDS,
    API_MAX_RETRIES,
    KP_MODEL,
    RENDER_MODEL,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ===================== 系统提示词 =====================

KP_SYSTEM_PROMPT = """
你是一名 COC（克苏鲁的呼唤）第七版的 AI 守秘人（Keeper of Arcane Lore），主持一场沉浸式跑团游戏。

## 一、你的六大核心职能

1. **世界观与场景叙述者**：构建所有环境细节（光影、气味、声音、触感）；绝不用上帝视角直白告知"这是深潜者/这是邪教仪式"，只用人类视角的感官描写呈现未知恐怖。
2. **规则裁决者**：判定所有属性检定（力量/敏捷/感知/智力/灵感/意志）的难度与成败；统一执行 SAN 理智扣除、HP 伤害、疯狂判定、战斗计算。
3. **全 NPC 扮演者**：一人饰演所有村民、邪教徒、学者、精神病患者、神话生物；区分不同 NPC 的语气与立场，神话生物只展现扭曲非人特征，不暴露种族名称。
4. **剧情引导者**：依据场景给出线索，玩家自由探索；玩家卡壳时可通过灵感/侦查检定给出隐晦提示；不强行将玩家拉回预设路线。
5. **恐怖氛围设计师**：核心恐惧是"人类面对不可理解的宇宙未知"；节奏控制：日常压抑 → 细微诡异 → 真相崩塌 → 疯狂/死亡；未知 ＞ 直白的怪物惊吓。
6. **结局与因果负责人**：玩家每个选择产生对应后果；不恶意乱撕卡，但也不放水。

## 二、你的四种标准口吻

1. **环境叙事口吻**（最常用）：第二人称、大量感官细节、慢节奏、克制阴冷。错误示例："这个屋子很恐怖。" 正确示例："推开腐朽木门的瞬间，潮湿发霉的寒气裹着淡淡的腐腥扑面而来，屋内只有窗外惨白月光勉强勾勒出家具轮廓。木地板踩上去发出空洞、拉长的吱呀声。"
2. **NPC 对话口吻**（千人千面）：普通村民口语化拘谨回避怪事；邪教徒狂热低沉夹杂诡异祷词；理智崩溃者语序混乱喃喃自语；学者冷静克制但谈及神话会颤抖。
3. **检定触发口吻**（中性客观，不带情绪）："你试图强行撬开上锁的铁皮箱，需要进行困难力量检定。"
4. **高潮恐怖口吻**（节奏放缓、留白、弱化解释）：不一次性讲清所有真相，只抛出碎片化惊悚事实，留下大量未知。

## 三、三大严禁行为

1. **禁止上帝视角剧透**：不能说"这是米·戈"、"深潜者正在举行复活仪式"；
2. **禁止替玩家做决定**：不能说"你现在应该逃跑"、"你最好直接攻击他"；
3. **禁止浮夸中二或廉价惊吓**：维持克苏鲁的压抑、宿命式恐怖基调。

## 四、输出格式（严格遵守 JSON）

你的每次回复包含**两层内容**：

- `kp_response`：游戏层面的回应。告诉玩家其行动触发了什么、是否需要检定、有什么选择。口吻中性客观，不带情绪渲染。100字以内。
- `narrative`：环境渲染文本。用第二人称感官描写呈现玩家此刻看到、听到、闻到、感受到的一切。口吻克制阴冷，不直接命名超自然存在。200字以内。
- `need_check`：需要检定的属性名，或 "None"。
- `difficulty`：检定难度（"普通"/"困难"/"极难"）。
- `story_end`：布尔值。**重要：当玩家明确表达"放弃""结束""不再继续""回归正常生活"等收束意图时，你必须尊重玩家的选择，将 story_end 设为 true 来优雅地结束故事。** 不要反复尝试把玩家拉回剧情。玩家第二次表达同样意图时，必须结束。日常推进行动设为 false。
- `suggestions`：字符串数组，包含 3 个具体的行动建议。每个建议 6-12 字，必须是玩家在当前场景下最可能采取的合理行动。用第二人称"你"开头或不带主语均可。例如：`["仔细检查桌上的文件", "向酒保打听失踪者的消息", "沿着血迹追踪到地下室"]`。若 story_end=true，此项可为空数组。

```json
{
  "kp_response": "游戏层面回应文本，告知检定类型/结果/可选行动",
  "narrative": "环境渲染文本，第二人称感官描写",
  "need_check": "力量/敏捷/感知/智力/灵感/意志/None",
  "difficulty": "普通/困难/极难",
  "story_end": false,
  "suggestions": ["建议1", "建议2", "建议3"]
}
```

## 五、need_check 判断指南
- 玩家行动有失败风险且结果不确定时，选择最相关的属性检定。
- 战斗/运动 → 力量或敏捷；阅读古籍/分析线索 → 智力；察觉异常/搜索 → 感知；抵抗恐惧/精神压迫 → 意志。
- 闲聊、询问 NPC、无风险的观察 → need_check 设为 "None"。
"""

RENDER_SYSTEM_PROMPT = """
你是一名 COC 跑团的环境渲染师。你的任务是将 KP 的剧情梗概与检定结果融合为一段沉浸式的环境叙述。

## 渲染规则
- 以 KP 提供的环境描写为基础，融入检定结果（成功/失败），不做剧情逻辑改动。
- 使用克苏鲁风格的文学语言：细腻的环境描写、心理恐惧的渲染、感官细节。
- 输出控制在 200 字以内。
- 保持第二人称"你"的叙述视角。
- 口吻克制阴冷，不浮夸、不中二、不直接命名超自然存在。
- 输出纯文本，不使用 JSON 格式。
"""


# ===================== LLM 调用函数 =====================

def _get_client() -> OpenAI:
    """
    获取 OpenAI 兼容客户端，封装异常处理。
    """
    try:
        return get_openai_client()
    except Exception as e:
        logger.error(f"创建 OpenAI 客户端失败: {e}")
        raise


def _call_with_retry(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    description: str = "LLM",
) -> str:
    """
    带超时和重试的 LLM 调用。

    参数：
        messages     : 消息列表
        model        : 模型名称
        temperature  : 温度参数
        max_tokens   : 最大 token 数
        description  : 调用描述（用于日志）

    返回：
        LLM 响应文本。

    异常：
        RuntimeError : 所有重试均失败
    """
    client = _get_client()
    last_error = None

    for attempt in range(API_MAX_RETRIES + 1):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - t0
            logger.info(f"{description} 调用成功 (attempt {attempt + 1}, {elapsed:.1f}s)")

            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM 返回空内容")
            return content.strip()

        except Exception as e:
            last_error = str(e)
            logger.warning(f"{description} 调用失败 (attempt {attempt + 1}/{API_MAX_RETRIES + 1}): {e}")

            if attempt < API_MAX_RETRIES:
                wait = 2.0 * (attempt + 1)  # 递增等待：2s, 4s
                logger.info(f"等待 {wait:.0f}s 后重试...")
                time.sleep(wait)

    raise RuntimeError(f"{description} 调用最终失败（已重试 {API_MAX_RETRIES} 次）: {last_error}")


def call_kp(
    player_input: str,
    character: Dict[str, Any],
    rag_context: str = "",
    messages_history: list = None,
    memory_summary: str = "",
    scene_context: str = "",
) -> Dict[str, str]:
    """
    调用 KP Agent，生成剧情梗概与检定需求。

    参数：
        player_input     : 玩家的文字输入
        character        : 当前角色完整字典
        rag_context      : RAG 检索到的知识库上下文
        messages_history : 历史消息列表（可选）
        memory_summary   : 历史记忆摘要文本（可选）
        scene_context    : 当前场景设定文本（可选，首轮注入以引导 KP 叙述）

    返回：
        {"kp_response": str, "narrative": str, "need_check": str, "difficulty": str}
    """
    # ---------- 构建系统消息 ----------
    character_summary = f"""
当前角色状态：
- 名称: {character.get('name', '未知')}
- 力量(STR): {character.get('STR')} | 敏捷(DEX): {character.get('DEX')}
- 感知(POW): {character.get('POW')} | 意志(WILL): {character.get('WILL')}
- 智力(INT): {character.get('INT')} | 灵感(IDEA): {character.get('IDEA')}
- HP: {character.get('HP')}/{character.get('MAX_HP')} | SAN: {character.get('SAN')}/{character.get('MAX_SAN')}
- 状态: {character.get('status', 'alive')} | 重伤: {character.get('wounded', False)} | 临时疯狂: {character.get('temp_insanity', False)}
"""

    system_content = KP_SYSTEM_PROMPT

    # 注入场景设定（最高优先级，放在最前面确保 KP 知道当前场景）
    if scene_context:
        system_content += (
            f"\n\n【🎬 当前场景设定 —— 你正处在以下场景中，请以此为基础推进剧情】\n"
            f"{scene_context}"
        )

    # 注入历史记忆（在规则参考之前，位置更靠前表示重要性）
    if memory_summary:
        system_content += (
            f"\n\n【📜 历史剧情摘要 —— 请牢记以下已发生的事件，保持叙述连贯性】\n"
            f"{memory_summary}"
        )

    if rag_context:
        system_content += f"\n\n【参考规则/背景知识】\n{rag_context}"
    system_content += f"\n\n{character_summary}"

    # ---------- 构建消息列表 ----------
    messages = [{"role": "system", "content": system_content}]

    if messages_history:
        messages.extend(messages_history)

    messages.append({"role": "user", "content": player_input})

    # ---------- 调用 LLM，含 JSON 解析重试 ----------
    # 简化策略：最多 2 次尝试（1次正常 + 1次格式纠正），不再嵌套 API 重试
    raw_output = None

    for attempt in range(2):  # 最多 2 次
        try:
            raw_output = _call_with_retry(
                messages=messages,
                model=KP_MODEL,
                temperature=0.8,
                max_tokens=1024,
                description=f"KP(attempt {attempt + 1})",
            )

            logger.info(f"KP 原始输出 (attempt {attempt + 1}): {raw_output[:200]}...")

            result = _parse_kp_json(raw_output)
            if result:
                return result

            # JSON 解析失败，追加格式纠正提示再试一次
            if attempt == 0:
                logger.warning("KP JSON 解析失败，追加格式提示后重试...")
                messages.append({
                    "role": "user",
                    "content": "请严格以 JSON 格式回复，确保可以被 json.loads() 解析。"
                })

        except RuntimeError as e:
            logger.error(f"KP API 调用失败 (attempt {attempt + 1}): {e}")
            # API 调用失败直接退出，不重试（_call_with_retry 已经处理了重试）
            break

    # 兜底返回
    logger.error(f"KP 调用最终失败，使用兜底回复。原始输出: {str(raw_output)[:200]}")
    return {
        "kp_response": "你环顾四周，暂时没有发现明显的威胁或线索。你可以继续探索、检查身边的物品，或者尝试其他行动。",
        "narrative": "你环顾四周，黑暗中似乎有什么东西在蠕动……但你看不清它的轮廓。空气沉重而潮湿，每一步都伴随着未知的恐惧。",
        "need_check": "None",
        "difficulty": "普通",
        "story_end": False,
        "suggestions": ["仔细观察周围环境", "检查随身携带的物品", "沿着来时的路退回去"],
    }


def call_render(narrative: str, check_result: str = "") -> str:
    """
    调用渲染 Agent，将剧情梗概润色为沉浸式文本。

    参数：
        narrative    : KP 输出的剧情梗概
        check_result : 检定结果文本（可选）

    返回：
        润色后的叙事文本（≤ 200 字）。
    """
    user_content = f"剧情梗概：{narrative}"
    if check_result:
        user_content += f"\n\n检定结果：{check_result}"
    user_content += "\n\n请将以上内容润色为一段 200 字以内的沉浸式 COC 跑团叙述。"

    try:
        rendered = _call_with_retry(
            messages=[
                {"role": "system", "content": RENDER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=RENDER_MODEL,
            temperature=0.6,
            max_tokens=512,
            description="Render",
        )
        logger.info(f"Render 输出: {rendered[:200]}...")
        return rendered

    except RuntimeError as e:
        logger.error(f"Render 调用最终失败: {e}")
        # 失败时直接返回原始梗概
        return narrative


# ===================== JSON 解析辅助 =====================

def _parse_kp_json(raw: str) -> Dict[str, str] | None:
    """
    尝试多种方式从 KP 输出中解析 JSON。

    支持：
        1. 直接 json.loads()
        2. 提取 ```json ... ``` 代码块
        3. 提取 { ... } 最外层花括号
    """
    # 方式 1：直接解析
    try:
        data = json.loads(raw)
        return _validate_kp_output(data)
    except json.JSONDecodeError:
        pass

    # 方式 2：提取 ```json 代码块
    if "```json" in raw:
        try:
            start = raw.index("```json") + 7
            end = raw.index("```", start)
            json_str = raw[start:end].strip()
            data = json.loads(json_str)
            return _validate_kp_output(data)
        except (ValueError, json.JSONDecodeError):
            pass

    # 方式 3：提取最外层花括号
    if "{" in raw and "}" in raw:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
            return _validate_kp_output(data)
        except (ValueError, json.JSONDecodeError):
            pass

    return None


def _validate_kp_output(data: Dict) -> Dict[str, str] | None:
    """
    验证 KP 输出的 JSON 结构是否包含必要字段。
    兼容旧格式（仅 narrative）和新格式（kp_response + narrative）。
    """
    if not isinstance(data, dict):
        return None
    if "narrative" not in data:
        return None

    # 设置默认值
    data.setdefault("need_check", "None")
    data.setdefault("difficulty", "普通")
    data.setdefault("story_end", False)
    data.setdefault("suggestions", [])
    # kp_response 缺省时用 narrative 的前 100 字代替
    data.setdefault("kp_response", str(data["narrative"])[:100])

    # 确保 suggestions 是字符串列表，最多 3 条
    raw_suggestions = data.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        raw_suggestions = []
    suggestions = [str(s) for s in raw_suggestions[:3] if s]

    return {
        "kp_response": str(data["kp_response"]),
        "narrative": str(data["narrative"]),
        "need_check": str(data["need_check"]),
        "difficulty": str(data["difficulty"]),
        "story_end": bool(data["story_end"]),
        "suggestions": suggestions,
    }
