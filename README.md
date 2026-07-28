# 🐙 COC 克苏鲁的呼唤 · AI 守秘人引擎

基于 **LangGraph** + **DeepSeek** + **Streamlit** 构建的 COC（Call of Cthulhu）第七版 AI KP（守秘人）引擎。玩家可在浏览器中体验由 AI 主持的沉浸式克苏鲁跑团冒险。

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────┐
│                  Streamlit 前端                   │
│  侧边栏（角色面板 + 文件上传）│ 主区域（对话流）     │
└────────────────────┬────────────────────────────┘
                     │ 每轮用户输入
                     ▼
┌─────────────────────────────────────────────────┐
│              LangGraph 工作流引擎                 │
│                                                  │
│  START → kp_node → [路由] → tool_node → render → END
│              │                    ↑              │
│              │ (需要检定)          │              │
│              └────────────────────┘              │
│              │ (无需检定)                         │
│              └──────────────────────────────────→ │
└──────┬──────────────────────────┬────────────────┘
       │                          │
       ▼                          ▼
┌──────────────┐        ┌──────────────────┐
│  DeepSeek V4 │        │   RAG 知识库      │
│   (KP + 渲染) │        │ ChromaDB +       │
│              │        │ MiniLM-L6-v2      │
└──────────────┘        └──────────────────┘
```

### 工作流说明

| 节点 | 功能 | 调用模型 |
|------|------|---------|
| `kp_node` | 接收玩家输入 + 角色状态 + RAG 上下文，输出剧情梗概与检定需求（JSON） | DeepSeek V4 Pro |
| `tool_node` | 执行骰子检定、理智扣减、战斗伤害计算 | 无（纯逻辑） |
| `render_node` | 将剧情梗概 + 检定结果润色为沉浸式叙述 | DeepSeek Flash |

---

## 📁 项目结构

```
coc-keeper-agent/
├── app/
│   └── main.py              # Streamlit 前端入口
├── backend/
│   ├── __init__.py
│   ├── graph.py             # LangGraph 主控工作流
│   ├── agents.py            # KP / Render Agent 定义 + LLM 调用
│   ├── tools.py             # 统一工具入口（属性检定/理智/战斗）
│   ├── character.py         # 角色生成 / 更新 / 验证
│   ├── dice.py              # 骰子引擎（D100/D6/D10 + 难度判定）
│   ├── combat.py            # 攻击 / 闪避 / 伤害计算
│   ├── sanity.py            # 理智损失与疯狂判定
│   └── state_manager.py     # 对话截断 / 状态存取 / 临时状态
├── rag/
│   ├── __init__.py
│   ├── loader.py            # 文件解析 + 分块 + 向量化入库
│   └── retriever.py         # RAG 检索器
├── utils/
│   ├── __init__.py
│   ├── config.py            # 配置管理 + OpenAI 客户端封装
│   └── logger.py            # 日志工具
├── data/
│   └── default_rule.txt     # 默认 COC 规则知识库
├── .env.example             # 环境变量模板
├── requirements.txt         # Python 依赖
├── .gitignore
└── README.md
```

---

## 🚀 本地运行步骤

### 1. 克隆项目

```bash
cd coc-keeper-agent
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# 或
venv\Scripts\activate         # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
# 复制 .env.example 为 .env
cp .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5. 下载 Embedding 模型（可选，用于 RAG 知识库）

```bash
# 自动从 HuggingFace 下载（需联网）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 6. 启动应用

```bash
streamlit run app/main.py
```

浏览器打开 `http://localhost:8501` 即可开始冒险。

---

## ☁️ 部署到 Streamlit Cloud

1. 将项目推送到 **GitHub 公开仓库**。
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)。
3. 点击「New app」，选择你的仓库，主文件路径设为 `app/main.py`。
4. 在「Advanced settings」中添加环境变量：
   - `DEEPSEEK_API_KEY` — 你的 API 密钥
   - `DEEPSEEK_BASE_URL` — `https://api.deepseek.com`
5. 点击「Deploy!」等待部署完成。

---

## 🎮 使用指南

1. **启动后自动生成角色**：侧边栏会显示随机生成的调查员属性。
2. **输入行动**：在底部输入框描述你的行动（如"我推开门"、"我翻阅桌上的日记"）。
3. **检定过程**：AI KP 会判断是否需要属性检定，骰子结果将自动计算并展示。
4. **上传规则书**（可选）：在侧边栏上传 TXT/PDF 格式的剧本或扩展规则，AI KP 会自动检索相关内容。
5. **游戏结束**：HP ≤ 0（死亡）或 SAN ≤ 0（永久疯狂）时游戏结束。
6. **重置**：点击侧边栏「重置游戏」按钮重新开始。

---

## ⚙️ 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 必填 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `EMBEDDING_MODEL_PATH` | Embedding 模型路径 | `./models/all-MiniLM-L6-v2` |

| 常量 | 说明 | 值 |
|------|------|-----|
| `MAX_CONTEXT_ROUNDS` | 保留对话轮数 | 5 |
| `RAG_TOP_K` | RAG 检索段落数 | 3 |
| `RAG_SIMILARITY_THRESHOLD` | RAG 相似度阈值 | 0.5 |

---

## 🛠️ 技术栈

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — 工作流编排引擎
- **[DeepSeek](https://deepseek.com)** — 大语言模型（KP 叙事 + 文案润色）
- **[Streamlit](https://streamlit.io)** — Web 前端框架
- **[ChromaDB](https://www.trychroma.com)** — 向量数据库
- **[sentence-transformers](https://www.sbert.net)** — 文本向量化
- **[PyPDF2](https://pypi.org/project/PyPDF2)** — PDF 解析

---

## 📝 License

MIT License — 仅供学习研究与娱乐使用。
