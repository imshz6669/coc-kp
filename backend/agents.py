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
你是一名 COC（克苏鲁的呼唤）第七版的 AI 守秘人（Keeper of Arcane Lore），主持一场沉浸式单人跑团游戏。

## 零、游戏节奏（单人短时跑团）

这是一场单人跑团，目标在 **45-60 分钟内** 完成一个完整故事。这意味着：
- **压缩场景数量**：整个剧本控制在 4-6 个关键场景，不要展开过多支线。
- **线索密度加倍**：每个场景至少给出 1-2 条明确的可推进线索，不要让玩家反复试错。
- **NPC 信息浓缩**：每个 NPC 对话直接给出核心秘密/线索，减少铺垫回合。
- **主动推进剧情**：玩家卡壳时，通过环境描写主动抛出下一个钩子（奇怪的声音、突然的访客、意外的发现）。
- **快速收束**：当玩家接近真相时，果断推向高潮和结局，不要拖沓。
- **检定精简**：只在关键行动上触发检定（撬锁、战斗、抵抗恐惧），普通探索和信息收集直接给结果。

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
- `scene`：**必填字段**。当前玩家所处的具体场景/位置名称，4-8字，用于侧边栏进度追踪。必须是明确的地名或房间名，绝不能是叙事文本片段。例如："乌蘅镇茶馆""沈宅前院枯井""地下密室""禁地深潭""温斯洛普大厅""亨利书房""地下洞穴"。每轮必须输出，场景未变化则保持与上一轮相同。
- `need_check`：需要检定的属性名，或 "None"。
- `difficulty`：检定难度（"普通"/"困难"/"极难"）。
- `hp_damage`：整数。玩家在本轮行动中实际受到的伤害点数。战斗中被击中（按武器类型 1-8）、坠落/陷阱（1-6）、灼烧等。玩家检定成功规避危险、或本轮无危险事件时为 0。
- `san_loss`：整数。玩家在本轮目睹/经历的恐怖造成的理智损失。参考：看到尸体 0-3（轻微）、目睹怪物 1-6（中度）、遭遇神话存在 1-10（严重）、直面宇宙真相 1-100（神话）。普通紧张场景为 0。
- `damage_source`：伤害来源简述（如"邪教徒的匕首"，hp_damage=0 时为空字符串）。
- `san_reason`：理智损失原因简述（如"目睹仪式中的无面尸体"，san_loss=0 时为空字符串）。
- `story_end`：布尔值。在以下两种情况设为 true：
  1. **玩家主动收束**：玩家明确表达"放弃""结束""离开""回归正常生活"等意图时。玩家第二次表达同样意图时必须结束。
  2. **剧本自然结局**：当玩家完成了剧本的最终目标（封印完成、仪式结束、核心真相揭露、最终战斗分出胜负），故事已到达不可逆转的终点时，必须设为 true。此时 narrative 应作为故事的尾声/后日谈。
  日常探索和推进行动设为 false。
- `suggestions`：字符串数组，包含 3 个具体的行动建议。每个建议 6-12 字，必须是玩家在当前场景下最可能采取的合理行动。用第二人称"你"开头或不带主语均可。例如：`["仔细检查桌上的文件", "向酒保打听失踪者的消息", "沿着血迹追踪到地下室"]`。若 story_end=true，此项可为空数组。

```json
{
  "kp_response": "游戏层面回应文本，告知检定类型/结果/可选行动",
  "narrative": "环境渲染文本，第二人称感官描写",
  "scene": "当前场景简短名称",
  "need_check": "力量/敏捷/感知/智力/灵感/意志/None",
  "difficulty": "普通/困难/极难",
  "hp_damage": 0,
  "san_loss": 0,
  "damage_source": "伤害来源简述（hp_damage=0 时为空字符串）",
  "san_reason": "理智损失原因简述（san_loss=0 时为空字符串）",
  "story_end": false,
  "suggestions": ["建议1", "建议2", "建议3"]
}
```

## 五、need_check 判断指南
- 玩家行动有失败风险且结果不确定时，选择最相关的属性检定。
- 战斗/运动 → 力量或敏捷；阅读古籍/分析线索 → 智力；察觉异常/搜索 → 感知；抵抗恐惧/精神压迫 → 意志。
- 闲聊、询问 NPC、无风险的观察 → need_check 设为 "None"。

## 六、伤害与理智裁决指南
- **先裁决检定，再定伤害**：玩家检定成功且规避了危险 → hp_damage 为 0；检定失败或被直接攻击 → 按武器/危险程度给出 1-8 点伤害。
- **理智损失独立于属性检定**：玩家目睹恐怖（尸体、怪物、仪式、超自然现象）时，无论是否触发属性检定，都必须给出 san_loss。这是 COC 的核心机制，不要漏。
- 每轮 hp_damage 与 san_loss 至少一个为 0 时，另一个也必须填写（可为 0）；两者都无 → 均填 0。
- 不要恶意撕卡：普通小伤 1-4，重创 5-8；SAN 单次损失超过 10 只用于直面神话存在。
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
    import random
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

            # 输出被 max_tokens 截断：翻倍 token 上限重试，避免语句半截
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length" and attempt < API_MAX_RETRIES:
                new_max = min(max_tokens * 2, 4096)
                logger.warning(
                    f"{description} 输出被截断 (finish_reason=length)，"
                    f"以 max_tokens={new_max} 重试..."
                )
                max_tokens = new_max
                time.sleep(1.0)
                continue

            return content.strip()

        except Exception as e:
            last_error = str(e)
            err_type = type(e).__name__

            # 区分连接错误 vs 其他错误
            is_conn = any(kw in str(e).lower() for kw in
                         ['connection', 'timeout', 'reset', 'refused', 'network', 'remote disconnect'])

            logger.warning(
                f"{description} 调用失败 (attempt {attempt + 1}/{API_MAX_RETRIES + 1}): "
                f"{err_type}: {str(e)[:120]}"
            )

            if attempt < API_MAX_RETRIES:
                # 连接错误用更长的退避时间（5s/10s），其他错误用 2s/4s
                base = 5.0 if is_conn else 2.0
                jitter = random.uniform(0.5, 1.5)  # ±50% 抖动避免惊群
                wait = base * (attempt + 1) * jitter
                logger.info(f"等待 {wait:.1f}s 后重试...")
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
            f"\n\n【🎬 当前场景设定：你正处在以下场景中，请以此为基础推进剧情】\n"
            f"{scene_context}"
        )

    # 注入历史记忆（在规则参考之前，位置更靠前表示重要性）
    if memory_summary:
        system_content += (
            f"\n\n【📜 历史剧情摘要：请牢记以下已发生的事件，保持叙述连贯性】\n"
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
    raw_output = None

    for attempt in range(2):  # 最多 2 次（首次 + 格式纠正）
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
                # _parse_kp_json 现在含降级解析（方法4），几乎总能返回有效结果
                parse_method = "JSON" if result.get("need_check") != "None" or result.get("scene") else "heuristic"
                logger.info(f"KP 解析成功 ({parse_method}), need_check={result['need_check']}")
                return _normalize_kp_result(result)

            # JSON 结构解析失败且降级文本太短（<30字），尝试格式纠正
            if attempt == 0:
                logger.warning("KP 输出过短或无有效内容，追加格式提示重试...")
                messages.append({
                    "role": "user",
                    "content": "请严格以 JSON 格式回复，确保可以被 json.loads() 解析。"
                })

        except RuntimeError as e:
            logger.error(f"KP API 调用失败 (attempt {attempt + 1}): {e}")
            # _call_with_retry 已重试过，这里不再重试
            break

    # 兜底返回：优先尝试从已有文本提取，否则用硬编码
    if raw_output and len(raw_output) > 30:
        extracted = _extract_from_narrative(raw_output)
        logger.warning(f"KP 使用降级提取。原始输出前200字: {raw_output[:200]}")
        return _normalize_kp_result(extracted)

    logger.error("KP 调用完全失败，使用硬编码兜底回复。")
    return {
        "kp_response": "你环顾四周，暂时没有发现明显的威胁或线索。你可以继续探索、检查身边的物品，或者尝试其他行动。",
        "narrative": "你环顾四周，黑暗中似乎有什么东西在蠕动……但你看不清它的轮廓。空气沉重而潮湿，每一步都伴随着未知的恐惧。",
        "scene": "",
        "need_check": "None",
        "difficulty": "普通",
        "hp_damage": 0,
        "san_loss": 0,
        "damage_source": "",
        "san_reason": "",
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
        return _normalize_generated_text(rendered)

    except RuntimeError as e:
        logger.error(f"Render 调用最终失败: {e}")
        # 失败时直接返回原始梗概
        return _normalize_generated_text(narrative)


# ===================== JSON 解析辅助 =====================

def _parse_kp_json(raw: str) -> Dict[str, str] | None:
    """
    尝试多种方式从 KP 输出中解析 JSON。

    支持：
        1. 直接 json.loads()
        2. 提取 ```json ... ``` 代码块（含换行变体）
        3. 提取 { ... } 最外层花括号（多起点尝试）
        4. 降级：纯叙事文本时智能提取 narrative + 推断字段
    """
    # 方式 1：直接解析
    try:
        data = json.loads(raw)
        return _validate_kp_output(data)
    except json.JSONDecodeError:
        pass

    # 方式 2：提取 ```json 代码块（兼容换行变体）
    import re
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if fence_match:
        try:
            json_str = fence_match.group(1).strip()
            data = json.loads(json_str)
            return _validate_kp_output(data)
        except json.JSONDecodeError:
            pass

    # 方式 3：提取最外层花括号（从后往前找 }，兼容嵌套和叙事前置）
    if "{" in raw and "}" in raw:
        # 从文本末尾往回找 }，确保是 JSON 的结束花括号
        last_brace = raw.rindex("}")
        # 在 } 之前找到配对的 {（从后往前扫描）
        depth = 0
        json_start = -1
        for i in range(last_brace, -1, -1):
            if raw[i] == "}":
                depth += 1
            elif raw[i] == "{":
                depth -= 1
                if depth == 0:
                    json_start = i
                    break
        if json_start >= 0:
            try:
                json_str = raw[json_start:last_brace + 1]
                data = json.loads(json_str)
                return _validate_kp_output(data)
            except json.JSONDecodeError:
                # 有时 JSON 内字符串含未转义换行，尝试修复
                pass

    # 方式 4：降级 —— KP 输出了纯叙事文本无 JSON 结构
    # 从文本中智能提取 narrative 和 kp_response
    text = raw.strip()
    if len(text) > 30:
        return _extract_from_narrative(text)

    return None


def _extract_from_narrative(text: str) -> Dict[str, str]:
    """
    从纯叙事文本中智能提取字段。
    当 KP 未输出 JSON 结构时使用。

    策略：
    - narrative：取完整文本（截断到 250 字）
    - kp_response：取前 2-3 句作为游戏层面回应
    - scene：尝试从文中提取地点名词
    - need_check：检测检定相关关键词
    - suggestions：从文本推断 3 个合理行动
    """
    import re

    # narrative：完整文本（限制长度）
    narrative = text[:300]

    # kp_response：取前 100 字（通常 KP 会在开头给出游戏层面信息）
    # 按句号/感叹号/问号分句，取前两句
    sentences = re.split(r'[。！？]', text)
    kp_response = ""
    for s in sentences[:3]:
        s = s.strip()
        if len(s) > 5:
            kp_response += s + "。"
        if len(kp_response) > 100:
            break
    if not kp_response:
        kp_response = text[:100]

    # scene：通用地点提取 —— 纯结构匹配，无剧本特化词
    scene = ""

    # ---- 策略1：中文地点后缀匹配（通用，适配任何剧本） ----
    loc_suffixes = r'(?:室|厅|堂|院|楼|间|房|庙|洞|碑|潭|井|馆|店|站|所|处|口|门|阶|层|宅|园|墓|塔|桥|台|窟|阁|亭|廊|殿|庵|观|寺)'
    for m in re.finditer(rf'([\w一-鿿]{{1,4}}){loc_suffixes}', text):
        raw = m.group(0)
        # 逐层剥离前缀，直到剩下有意义的地名
        cleaned = raw
        # 剥离动词/介词前缀（在/到/进/出/向/从/沿/顺/旁/边 + 推开/拉/走/跑/爬/蹲/站/坐/躺）
        cleaned = re.sub(r'^[在到进出从向对沿顺傍靠旁推开拉走跑爬蹲站坐躺]{1,3}', '', cleaned)
        # 剥离"X的"修饰（昏暗的/废弃的/前方的/整个…）
        cleaned = re.sub(r'^[\w一-鿿]{1,3}的', '', cleaned)
        # 剥离纯形容词/数量词前缀
        cleaned = re.sub(r'^(?:废弃|破败|阴暗|潮湿|昏暗|漆黑|狭窄|宽阔|阴森|恐怖|陌生|古老|崭新|整个|某个|那个)', '', cleaned)
        # 剥离剩余的动词/介词（做了前几步后可能还剩残留单字）
        cleaned = re.sub(r'^[在到进出从向对沿顺旁推开拉走跑爬蹲站坐躺的了个]{1,2}', '', cleaned)
        if len(cleaned) >= 2:
            scene = cleaned[:8]
            break

    # ---- 策略2："在/到/进入 + 位置" 模式（兜底） ----
    if not scene:
        m = re.search(r'(?:在|到了?|进入|来到|回到|走进|踏入)([\w一-鿿]{2,6})(?:[，。,\s]|$)', text)
        if m:
            candidate = m.group(1).strip()
            # 排除虚词和代词
            stopwords = {'你', '我', '他', '她', '它', '这', '那', '一', '是', '的', '了', '着', '过',
                        '不', '很', '都', '也', '就', '要', '会', '能', '可', '被', '把', '向', '从',
                        '什么', '怎么', '一个', '这里', '那里', '哪儿', '外面', '里面', '前面', '后面'}
            if candidate not in stopwords and len(candidate) >= 2:
                scene = candidate[:8]

    # need_check：检测检定关键词
    need_check = "None"
    difficulty = "普通"
    check_patterns = [
        (r'力量|力气|撬|推|搬|举|砸', '力量'),
        (r'敏捷|跳跃|攀爬|闪避|躲开|快速', '敏捷'),
        (r'感知|察觉|发现|听到|闻到|注意到|侦查|观察', '感知'),
        (r'智力|分析|解读|辨认|回忆|知识|研究', '智力'),
        (r'意志|抵抗|忍住|坚持|精神|勇气', '意志'),
    ]
    for pattern, attr in check_patterns:
        if re.search(pattern, text):
            need_check = attr
            break

    # story_end：检测结局信号
    story_end = bool(re.search(r'封印.*完成|永远.*沉睡|一切.*结束|故事.*落幕|永远.*消失|回归.*平静', text))

    # suggestions：从文本推断
    suggestions = _infer_suggestions(text)

    logger.info(f"降级解析: 从 {len(text)} 字叙事文本提取字段, scene={scene}, need_check={need_check}")
    return {
        "kp_response": kp_response[:100],
        "narrative": narrative,
        "scene": scene,
        "need_check": need_check,
        "difficulty": difficulty,
        # 降级路径无法可靠推断数值，保守填 0（宁可不扣也不乱扣）
        "hp_damage": 0,
        "san_loss": 0,
        "damage_source": "",
        "san_reason": "",
        "story_end": story_end,
        "suggestions": suggestions,
    }


def _infer_suggestions(text: str) -> list:
    """从叙事文本推断合理的后续行动建议"""
    import re
    suggestions = []
    keyword_map = [
        (r'门|大门|入口|房间', '仔细观察周围的环境'),
        (r'棺|尸体|尸|墓', '检查棺椁的细节和封印'),
        (r'书|文件|信件|账册|族谱|纸', '仔细翻阅附近的文字记录'),
        (r'人说|告诉|说道|低声', '继续向此人追问更多细节'),
        (r'井|地下|密室|暗道|楼梯', '探索通往深处的入口'),
        (r'符|咒|阵|封印|仪式', '研究这些符文或仪式的含义'),
        (r'钥匙|锁|铜|开门', '用钥匙打开对应的锁'),
        (r'潭|水|河|液体|血', '靠近水边仔细观察'),
        (r'逃|跑|恐惧|害怕|危险', '保持警惕，准备应对突发情况'),
    ]
    for pattern, suggestion in keyword_map:
        if re.search(pattern, text) and suggestion not in suggestions:
            suggestions.append(suggestion)
        if len(suggestions) >= 3:
            break

    # 不足 3 条时补充通用建议
    defaults = ["仔细观察周围环境", "寻找附近的可疑线索", "与在场的人交谈"]
    for d in defaults:
        if d not in suggestions and len(suggestions) < 3:
            suggestions.append(d)

    return suggestions[:3]


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
    data.setdefault("scene", "")
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
        "scene": str(data.get("scene", "")),
        "need_check": str(data["need_check"]),
        "difficulty": str(data["difficulty"]),
        "hp_damage": _safe_int(data.get("hp_damage"), hi=30),
        "san_loss": _safe_int(data.get("san_loss"), hi=100),
        "damage_source": str(data.get("damage_source", "") or ""),
        "san_reason": str(data.get("san_reason", "") or ""),
        "story_end": bool(data["story_end"]),
        "suggestions": suggestions,
    }


def _safe_int(value: Any, default: int = 0, hi: int = 999) -> int:
    """
    安全转换为非负整数：非法值返回 default，并钳制在 [0, hi]。
    KP 偶尔会输出字符串数字（如 "3"），需容错。
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(n, hi))


# ===================== 生成文本规范化 =====================

_SENTENCE_END = "。！？…"
_CLOSING_CHARS = "\"”」』）)]"


def _normalize_generated_text(text: str) -> str:
    """
    规范化 LLM 生成文本，修复两类常见输出问题：

    1. 标点错用：AI 生成的 em-dash / en-dash 一律替换为常规标点
       （"——" → "，"、"—" → "，"，数字区间 "–" → "-"）。
    2. 语句不完整：若文本不以句末标点结尾（被 max_tokens 截断），
       裁剪到最后一个完整句子，并追加"……"暗示未尽之意。

    注意：仅用于 KP / Render 的生成内容，不作用于玩家输入。
    """
    if not text:
        return text

    text = text.replace("——", "，").replace("—", "，").replace("–", "-")

    stripped = text.rstrip()
    if not stripped:
        return text

    # 以句末标点或闭合符号结尾 → 视为完整
    if stripped[-1] in _SENTENCE_END or stripped[-1] in _CLOSING_CHARS:
        return text

    # 找到最后一个完整句的结尾位置
    cut = -1
    for ch in _SENTENCE_END:
        idx = stripped.rfind(ch)
        if idx > cut:
            cut = idx

    if cut <= 0:
        return text  # 全文无句界，无法安全裁剪

    # 保留句末标点后紧跟的闭合引号/括号
    end = cut + 1
    while end < len(stripped) and stripped[end] in _CLOSING_CHARS:
        end += 1

    return stripped[:end] + "……"


def _normalize_kp_result(result: Dict[str, str]) -> Dict[str, str]:
    """规范化 KP 输出中的全部文本字段。"""
    result["kp_response"] = _normalize_generated_text(str(result.get("kp_response", "")))
    result["narrative"] = _normalize_generated_text(str(result.get("narrative", "")))
    return result
