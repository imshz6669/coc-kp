"""
记忆管理模块 —— 对话历史概括、持久化与上下文注入。

提供 MemoryManager 类，负责：
    - 轮次计数与触发判断（每 N 轮触发一次概括）
    - 调用 DeepSeek Flash 进行对话概括
    - 概括结果持久化为 JSON 文件
    - 超长会话的元概括（摘要的摘要）
    - 向 KP Agent 注入记忆上下文
"""

import os
import json
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.config import (
    MEMORY_SUMMARIZE_INTERVAL,
    MEMORY_DIR,
    MEMORY_MAX_SUMMARIES_BEFORE_META,
)

logger = get_logger(__name__)

# ===================== 概括提示词 =====================

SUMMARY_SYSTEM_PROMPT = """你是一名COC跑团的记录员。请将以下游戏对话概括为一段简洁的历史摘要。

要求：
1. 仅保留关键事件、重要NPC、地点变更、角色状态变化
2. 忽略无关细节和过渡性对话
3. 输出严格的JSON格式，不要输出任何其他内容

输出格式：
{
  "summary": "200字以内的连贯叙述，概括这段时间内发生的主要事件",
  "key_events": ["事件1", "事件2", "事件3"]
}"""

META_SUMMARY_PROMPT = """你是一名COC跑团的记录员。请将以下多段游戏摘要合并为一段综合摘要。

要求：
1. 保留最重要的情节线索和未完成的任务目标
2. 概括角色的整体状态变化
3. 150字以内
4. 输出严格的JSON格式，不要输出任何其他内容

输出格式：
{"summary": "综合摘要文本"}"""


# ===================== MemoryManager =====================

class MemoryManager:
    """
    管理对话历史记忆的概括、持久化与上下文注入。

    每个会话一个实例，通过 session_id 隔离。

    属性：
        session_id     : 会话 UUID
        memory_dir     : 记忆文件存储目录 (data/memory/{session_id}/)
        round_counter  : 当前轮次累计计数
        summaries      : 已加载的摘要列表 (从磁盘恢复)
        pending_rounds : 待概括的最近 N 轮对话数据
        _memory_context: 缓存的内存上下文字符串
    """

    def __init__(self, session_id: str, base_dir: str = None):
        """
        初始化记忆管理器。

        参数：
            session_id : 会话唯一标识
            base_dir   : 记忆文件根目录，默认使用 config.MEMORY_DIR
        """
        self.session_id = session_id
        base = base_dir or MEMORY_DIR
        self.memory_dir = os.path.join(base, session_id)
        self.round_counter = 0
        self.summaries: List[Dict[str, Any]] = []
        self.pending_rounds: List[Dict[str, Any]] = []
        self._memory_context = ""

        # 确保目录存在
        os.makedirs(self.memory_dir, exist_ok=True)

        # 从磁盘加载已有摘要
        self._load_summaries()
        # 恢复上次概括失败但已落盘的轮次，并入待概括队列
        self._load_failed_backup()
        self._rebuild_context()

        logger.info(
            f"MemoryManager 初始化: session={session_id[:8]}..., "
            f"已有 {len(self.summaries)} 条摘要, "
            f"轮次计数={self.round_counter}"
        )

    # ---------- 公开方法 ----------

    def add_round(self, user_msg: str, assistant_msg: str) -> bool:
        """
        记录一轮对话完成。

        参数：
            user_msg      : 玩家输入文本
            assistant_msg : KP 回复文本（润色后的最终回复）

        返回：
            True 表示已累积足够轮次，需要触发概括。
        """
        self.round_counter += 1

        self.pending_rounds.append({
            "round_num": self.round_counter,
            "user_action": user_msg,
            "assistant_response": assistant_msg[:500],  # 截断过长回复
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(
            f"Memory: 记录第 {self.round_counter} 轮, "
            f"pending={len(self.pending_rounds)}/{MEMORY_SUMMARIZE_INTERVAL}"
        )

        if len(self.pending_rounds) >= MEMORY_SUMMARIZE_INTERVAL:
            return True
        return False

    def summarize(self, client) -> bool:
        """
        调用 DeepSeek Flash 概括 pending_rounds，写入磁盘。

        失败容错策略：
        - 失败时**不清空缓冲区**，并落盘备份 pending_rounds_XXXX_XXXX.json
          （进程重启后由 _load_failed_backup 恢复，避免数据永久丢失）
        - 下一轮 add_round 达到阈值时会自动重试，本轮对话自然并入
          下一次概括的更大范围（rounds_range 连续，round_counter 不错位）
        - 成功后清空缓冲区并删除备份文件

        参数：
            client : OpenAI 兼容客户端

        返回：
            True 表示概括成功，False 表示失败（调用方可忽略，不影响游戏主流程）。
        """
        if not self.pending_rounds:
            logger.warning("Memory: 无待概括轮次，跳过。")
            return False

        rounds_range = self._get_pending_range()
        logger.info(
            f"Memory: 开始概括第 {rounds_range} 轮对话 "
            f"({len(self.pending_rounds)} 轮)..."
        )

        try:
            summary_data = self._call_summary_llm(client)
            if summary_data is None:
                raise RuntimeError("概括 LLM 返回空结果")

            summary_data["rounds_range"] = rounds_range
            self._save_summary(summary_data)

            # 重新加载摘要并重建上下文
            self._load_summaries()
            self._rebuild_context()

            # 检查是否需要元概括
            self._maybe_meta_summarize(client)

            logger.info(f"Memory: 概括完成 → {self.memory_dir}")

            # 成功后才清空缓冲区，并清理历史失败备份
            self.pending_rounds.clear()
            self._clear_failed_backups()
            return True

        except Exception as e:
            logger.error(f"Memory: 概括失败 ({rounds_range}): {e}")
            # 保留缓冲区（下一轮重试）+ 落盘备份（防进程重启丢数据）
            self._save_failed_backup()
            return False

    def get_context(self) -> str:
        """
        返回拼接后的记忆上下文，供注入 KP 提示词。

        若尚无摘要，返回空字符串。
        """
        return self._memory_context

    def clear(self) -> None:
        """
        删除该会话的所有记忆文件，重置管理器状态。
        """
        try:
            if os.path.exists(self.memory_dir):
                shutil.rmtree(self.memory_dir)
                logger.info(f"Memory: 已清除会话记忆 ({self.session_id[:8]}...)")
        except Exception as e:
            logger.error(f"Memory: 清除目录失败: {e}")

        self.summaries.clear()
        self.pending_rounds.clear()
        self.round_counter = 0
        self._memory_context = ""

    # ---------- 内部方法 ----------

    def _load_summaries(self) -> None:
        """
        扫描 data/memory/{session_id}/ 目录，加载所有摘要 JSON 文件。
        恢复 round_counter（从摘要推算）。
        """
        self.summaries = []

        if not os.path.isdir(self.memory_dir):
            return

        try:
            files = sorted(os.listdir(self.memory_dir))
        except Exception as e:
            logger.error(f"Memory: 列出目录失败: {e}")
            return

        meta_summaries = []
        individual_summaries = []

        for filename in files:
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.memory_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Memory: 跳过损坏文件 {filename}: {e}")
                continue

            summary_type = data.get("type", "summary")
            if summary_type == "meta_summary":
                meta_summaries.append(data)
            elif summary_type == "summary":
                individual_summaries.append(data)
            # 其他类型（如 failed_backup 备份文件）不参与摘要加载

        # 按 index 排序
        individual_summaries.sort(key=lambda s: s.get("index", 0))
        meta_summaries.sort(key=lambda s: s.get("index", 0))

        self.summaries = individual_summaries + meta_summaries

        # 恢复轮次计数
        if individual_summaries:
            last = individual_summaries[-1]
            rounds_range = last.get("rounds_range", "")
            try:
                # rounds_range 格式如 "7-12"，取最大值
                end = rounds_range.split("-")[-1]
                self.round_counter = int(end)
            except (ValueError, IndexError):
                self.round_counter = len(individual_summaries) * MEMORY_SUMMARIZE_INTERVAL

        logger.info(
            f"Memory: 从磁盘加载 {len(individual_summaries)} 条摘要, "
            f"{len(meta_summaries)} 条元摘要, "
            f"恢复轮次={self.round_counter}"
        )

    def _rebuild_context(self) -> None:
        """
        根据当前加载的摘要重建 memory_context 字符串。

        构建层次化的文本：
            - 先列出元摘要（如果存在）
            - 再列出最近的个体摘要
        限制总长度避免 token 溢出（默认 ~2000 字）。
        """
        if not self.summaries:
            self._memory_context = ""
            return

        parts = []

        # 先加元摘要（高层概括）
        meta_summaries = [s for s in self.summaries if s.get("type") == "meta_summary"]
        for ms in meta_summaries:
            summary_text = ms.get("summary", "")
            if summary_text:
                parts.append(f"【综合摘要】{summary_text}")

        # 再加最近的个体摘要（最多保留 3 条）
        individual = [s for s in self.summaries if s.get("type") != "meta_summary"]
        recent = individual[-3:]  # 只保留最近 3 条
        for s in recent:
            rounds_range = s.get("rounds_range", "?")
            summary_text = s.get("summary", "")
            if summary_text:
                parts.append(f"【第{rounds_range}轮摘要】{summary_text}")

        self._memory_context = "\n\n".join(parts)

        # 长度保护：截断至 ~2000 字
        if len(self._memory_context) > 2500:
            self._memory_context = self._memory_context[:2500] + "…"
            logger.info("Memory: 上下文过长，已截断至 2500 字符")

    def _save_failed_backup(self) -> None:
        """
        概括失败时将待概括轮次落盘备份，防止进程重启导致记忆永久丢失。

        文件名：pending_rounds_{首轮:04d}_{末轮:04d}.json
        同一范围重复失败时覆盖写入，范围扩大时生成新文件，
        成功后由 _clear_failed_backups 统一清理。
        """
        if not self.pending_rounds:
            return

        first = self.pending_rounds[0]["round_num"]
        last = self.pending_rounds[-1]["round_num"]
        filename = f"pending_rounds_{first:04d}_{last:04d}.json"
        filepath = os.path.join(self.memory_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "type": "failed_backup",
                    "version": 1,
                    "failed_at": datetime.now().isoformat(),
                    "rounds": self.pending_rounds,
                }, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Memory: 概括失败，已备份 {len(self.pending_rounds)} 轮到 {filename}"
            )
        except Exception as e:
            logger.error(f"Memory: 失败备份写入失败: {e}")

    def _load_failed_backup(self) -> None:
        """
        启动时恢复上次概括失败但已落盘的轮次，并入待概括队列。

        按 round_num 去重合并后删除备份文件；round_counter 对齐到
        备份中的最大轮号，保证后续轮次编号连续、不错位。
        """
        try:
            files = sorted(os.listdir(self.memory_dir))
        except Exception as e:
            logger.error(f"Memory: 列出目录失败: {e}")
            return

        for filename in files:
            if not filename.startswith("pending_rounds_") or not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.memory_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Memory: 读取失败备份异常 {filename}: {e}")
                continue

            if data.get("type") != "failed_backup":
                continue

            rounds = data.get("rounds", [])
            if not rounds:
                os.remove(filepath)
                continue

            existing = {r.get("round_num") for r in self.pending_rounds}
            restored = 0
            for rd in rounds:
                rn = rd.get("round_num")
                if rn is not None and rn not in existing:
                    self.pending_rounds.append(rd)
                    existing.add(rn)
                    restored += 1
            self.pending_rounds.sort(key=lambda r: r.get("round_num", 0))

            try:
                os.remove(filepath)
            except OSError:
                pass

            max_round = max((r.get("round_num", 0) for r in rounds), default=0)
            self.round_counter = max(self.round_counter, max_round)
            logger.info(
                f"Memory: 恢复 {restored} 轮失败概括 → 待概括队列，"
                f"round_counter 对齐到 {self.round_counter}"
            )

    def _clear_failed_backups(self) -> None:
        """概括成功后清理全部失败备份文件。"""
        try:
            files = os.listdir(self.memory_dir)
        except Exception:
            return
        for filename in files:
            if not filename.startswith("pending_rounds_") or not filename.endswith(".json"):
                continue
            try:
                os.remove(os.path.join(self.memory_dir, filename))
            except OSError:
                pass

    def _save_summary(self, summary_data: Dict[str, Any]) -> str:
        """
        将单条摘要写入 JSON 文件。

        返回：
            写入的文件路径。
        """
        next_idx = self._get_next_index()
        summary_data["index"] = next_idx
        summary_data["type"] = "summary"
        summary_data["version"] = 1
        summary_data["created_at"] = datetime.now().isoformat()

        filename = f"summary_{next_idx:04d}.json"
        filepath = os.path.join(self.memory_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Memory: 摘要已保存 → {filename}")
        return filepath

    def _get_next_index(self) -> int:
        """计算下一条摘要的序号。"""
        individual = [s for s in self.summaries if s.get("type") != "meta_summary"]
        if not individual:
            return 1
        return max(s.get("index", 0) for s in individual) + 1

    def _get_pending_range(self) -> str:
        """返回待概括轮次的范围字符串，如 '1-6'。"""
        if not self.pending_rounds:
            return "?"
        first = self.pending_rounds[0]["round_num"]
        last = self.pending_rounds[-1]["round_num"]
        return f"{first}-{last}"

    def _call_summary_llm(self, client) -> Optional[Dict[str, Any]]:
        """
        调用 DeepSeek Flash 进行概括。

        参数：
            client : OpenAI 兼容客户端

        返回：
            解析成功的摘要 dict，失败返回 None。
        """
        # 构建用户消息
        rounds_text = self._format_pending_rounds()
        user_content = f"以下是{len(self.pending_rounds)}轮COC跑团对话记录，请按JSON格式输出摘要：\n\n{rounds_text}"

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                temperature=0.4,
                max_tokens=512,
            )

            raw_output = response.choices[0].message.content.strip()
            logger.info(f"Memory: 概括 LLM 原始输出 ({len(raw_output)} 字符)")

            return _parse_summary_json(raw_output)

        except Exception as e:
            logger.error(f"Memory: 概括 LLM 调用失败: {e}")
            return None

    def _format_pending_rounds(self) -> str:
        """将待概括轮次格式化为 LLM 输入文本。"""
        lines = []
        for rd in self.pending_rounds:
            rn = rd.get("round_num", "?")
            ua = rd.get("user_action", "")
            ar = rd.get("assistant_response", "")
            lines.append(f"第{rn}轮:")
            lines.append(f"  玩家: {ua}")
            lines.append(f"  KP: {ar[:300]}")
            lines.append("")
        return "\n".join(lines)

    def _maybe_meta_summarize(self, client) -> bool:
        """
        当个体摘要数 >= MEMORY_MAX_SUMMARIES_BEFORE_META 时触发元概括。

        将最近一批个体摘要合并为一条综合摘要。

        返回：
            True 表示触发并成功，False 表示无需触发或失败。
        """
        individual = [s for s in self.summaries if s.get("type") != "meta_summary"]
        meta_count = sum(1 for s in self.summaries if s.get("type") == "meta_summary")

        if len(individual) < MEMORY_MAX_SUMMARIES_BEFORE_META:
            return False

        logger.info(
            f"Memory: 触发元概括 "
            f"(个体摘要={len(individual)}, 阈值={MEMORY_MAX_SUMMARIES_BEFORE_META})"
        )

        try:
            # 取最近的 N 条个体摘要进行元概括
            to_compress = individual[-MEMORY_MAX_SUMMARIES_BEFORE_META:]
            summaries_text = "\n---\n".join(
                f"[{s.get('rounds_range', '?')}]: {s.get('summary', '')}"
                for s in to_compress
            )

            user_content = (
                f"请将以下{MEMORY_MAX_SUMMARIES_BEFORE_META}段游戏摘要合并为一段综合摘要。\n\n"
                f"{summaries_text}"
            )

            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": META_SUMMARY_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=256,
            )

            raw_output = response.choices[0].message.content.strip()
            parsed = _parse_summary_json(raw_output)

            if parsed is None:
                raise RuntimeError("元概括 LLM 返回空结果")

            # 保存元摘要
            meta_idx = meta_count + 1
            meta_data = {
                "index": meta_idx,
                "type": "meta_summary",
                "version": 1,
                "created_at": datetime.now().isoformat(),
                "summaries_covered": [s.get("index") for s in to_compress],
                "summary": parsed.get("summary", ""),
            }

            filename = f"meta_{meta_idx:04d}.json"
            filepath = os.path.join(self.memory_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Memory: 元摘要已保存 → {filename}")
            self._load_summaries()
            self._rebuild_context()
            return True

        except Exception as e:
            logger.error(f"Memory: 元概括失败: {e}")
            return False


# ===================== JSON 解析辅助 =====================

def _parse_summary_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    尝试多种方式从 LLM 输出中解析摘要 JSON。

    策略：
        1. 直接 json.loads()
        2. 提取 ```json ... ``` 代码块
        3. 提取 { ... } 最外层花括号
    """
    # 方式 1：直接解析
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "summary" in data:
            return data
    except json.JSONDecodeError:
        pass

    # 方式 2：提取 ```json 代码块
    if "```json" in raw:
        try:
            start = raw.index("```json") + 7
            end = raw.index("```", start)
            json_str = raw[start:end].strip()
            data = json.loads(json_str)
            if isinstance(data, dict) and "summary" in data:
                return data
        except (ValueError, json.JSONDecodeError):
            pass

    # 方式 3：提取最外层花括号
    if "{" in raw and "}" in raw:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
            if isinstance(data, dict) and "summary" in data:
                return data
        except (ValueError, json.JSONDecodeError):
            pass

    logger.warning(f"Memory: 无法解析摘要 JSON，原始输出: {raw[:300]}")
    return None
