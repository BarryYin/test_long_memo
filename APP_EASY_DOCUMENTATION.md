# Collection Agent (Easy Mode) - 技术文档

## 📋 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [核心组件](#核心组件)
4. [数据流分析](#数据流分析)
5. [关键功能](#关键功能)
6. [配置说明](#配置说明)
7. [使用指南](#使用指南)
8. [扩展开发](#扩展开发)

---

## 项目概述

### 简介
`app_easy.py` 是一个基于 Streamlit 和 OpenAI GPT-4o-mini 的智能催收对话系统。该系统采用三层架构设计，通过记忆层（Memory Layer）追踪客户状态，通过策略层（Layer 1）制定催收计划，通过执行层（Layer 2）与客户交互，并通过评估层（Layer 3）优化策略。

### 核心特性
- **智能记忆系统**：实时跟踪客户意图、还款能力、原因分类等关键信息
- **三层智能架构**：策略制定、执行、评估的完整闭环
- **信息收敛机制**：系统化收集5个关键还款信息（能力、时间、金额、方式、展期）
- **历史分析能力**：自动解析过往催收记录，提取关键模式
- **动态策略调整**：根据实时对话效果自动优化催收策略
- **可视化界面**：清晰展示策略、分析、记忆状态

### 技术栈
- **前端框架**：Streamlit
- **AI 模型**：OpenAI GPT-4o-mini
- **配置管理**：YAML
- **数据处理**：JSON, Python 标准库

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                      │
│  ┌─────────────┐              ┌──────────────────────────┐  │
│  │  Chat UI    │              │  Analysis Dashboard      │  │
│  │  (60%)      │              │  - Strategy              │  │
│  │             │              │  - Memory State          │  │
│  │             │              │  - Thinking Process      │  │
│  └─────────────┘              └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               ↓↑
┌─────────────────────────────────────────────────────────────┐
│                      Memory Layer (核心)                     │
│  - 意图判断 (LLM-based: 0/1)                                 │
│  - 能力评估 (full/partial/zero)                             │
│  - 原因分类 (unemployment/illness/forgot...)                │
│  - 信息收敛追踪 (5个关键信息的完成度)                         │
│  - 历史分析结果缓存                                          │
└─────────────────────────────────────────────────────────────┘
                               ↓↑
┌─────────────────────────────────────────────────────────────┐
│                    Three-Layer Architecture                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer 1: Strategy Manager (策略层)                   │  │
│  │  - 根据客户档案、历史记录、配置生成催收策略             │  │
│  │  - 定义"多步收敛路径" (Step1-7)                       │  │
│  │  - 根据 Layer 3 反馈动态调整策略                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                               ↓                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer 2: Executor (执行层)                           │  │
│  │  - 接收策略、记忆上下文、历史记录                      │  │
│  │  - 与客户进行自然对话                                 │  │
│  │  - 按策略逐步收集关键信息                             │  │
│  │  - 输出 JSON 格式响应（分析+回复）                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                               ↓                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Layer 3: Evaluator (评估层)                          │  │
│  │  - 评估回款可能性 (HIGH/MEDIUM/LOW)                   │  │
│  │  - 分析信息收敛进度                                   │  │
│  │  - 生成策略优化建议                                   │  │
│  │  - 触发 Layer 1 策略更新（当评估为 LOW 时）           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               ↓↑
┌─────────────────────────────────────────────────────────────┐
│                     OpenAI GPT-4o-mini API                   │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **初始化阶段**
   ```
   历史记录 → Memory Layer (parse_history_summary) → 历史分析结果
   客户档案 + 历史记录 → Layer 1 → 初始策略 → Layer 2 → 开场白
   ```

2. **对话循环**
   ```
   用户输入 → Memory Layer (extract_from_dialogue) → 更新记忆状态
                ↓
   记忆上下文 + 对话历史 → Layer 3 → 评估报告
                ↓
   [如果 LOW] → Layer 1 (update_strategy) → 新策略
                ↓
   策略 + 记忆 + 历史 → Layer 2 → AI 回复 + 思考过程
                ↓
   显示在 UI → 等待下一轮用户输入
   ```

---

## 核心组件

### 1. MemoryLayer（记忆层）

#### 核心职责
- **意图识别**：使用 LLM 判断用户今天是否有还款意愿（1=有，0=无）
- **信息提取**：从对话中提取能力、原因、日期、金额等关键信息
- **历史解析**：自动分析过往催收记录，提取失约次数、原因分类、能力评估
- **状态追踪**：维护客户的完整画像和信息收敛进度

#### 数据结构

```python
self.memory = {
    # === 当前会话追踪 ===
    "intent_to_pay_today": int,        # 1=有意愿，0=无意愿
    "payment_refusals": int,            # 拒付次数计数器
    "broken_promises": int,             # 失约次数计数器
    "reason_category": str,             # 原因分类
    "ability_score": str,               # 能力评估
    "reason_detail": str,               # 具体理由文本
    "unresolved_obstacles": list,       # 未解决的障碍列表
    
    # === 历史分析结果 ===
    "history_summary": str,             # 历史摘要（100-200字）
    "history_broken_promises": int,     # 历史失约次数
    "history_reason_category": str,     # 历史原因分类
    "history_ability_score": str,       # 历史能力评估
    
    # === 信息收敛追踪 ===
    "has_ability_confirmed": bool,      # 是否确认有还款能力
    "payment_date_confirmed": str,      # 具体还款日期
    "payment_amount_confirmed": str,    # 具体还款金额
    "payment_type_confirmed": str,      # 付款类型（full/partial）
    "extension_requested": bool         # 是否请求展期
}
```

#### 关键方法

**1. detect_payment_intent(user_msg) → int**
```python
# 使用 LLM 判断用户意图
# 输入：用户消息
# 输出：1（今天会还）或 0（今天不会还）
# 示例：
#   "我今天下午3点给你还" → 1
#   "现在没钱，明天再说" → 0
```

**2. extract_from_dialogue(user_msg, conversation_history)**
```python
# 从对话中提取关键信息，分6步：
# Step 1: LLM 意图判断（调用 detect_payment_intent）
# Step 2: 能力评估（关键词匹配：全额/部分/没钱）
# Step 3: 原因分类（失业/生病/忘记/恶意拖延/其他）
# Step 4: 具体理由（保存原始文本）
# Step 5: 未解决障碍（开车/忙/会议/睡觉等）
# Step 6: 收敛性信息（日期/金额/类型/展期）
```

**3. parse_history_summary(history_text)**
```python
# 使用 LLM 解析历史记录
# 输入：历史文本（可以是多天的催收记录）
# 输出：JSON格式的分析结果
# {
#   "summary": "100-200字的中文摘要",
#   "broken_promises": 失约次数,
#   "reason_category": 主要原因,
#   "ability_score": 能力评估
# }
```

**4. get_memory_context() → str**
```python
# 生成格式化的记忆摘要文本
# 包含：
#   - 客户当前画像
#   - 关键信息收敛进度（✓ 已确认 / ⏳ 未确认）
#   - 历史分析（如果存在）
# 用于传递给 Layer 1/2/3
```

#### 信息收敛机制详解

系统需要收集的 **5个关键信息**：

| 信息项 | 字段名 | 示例值 | 重要性 |
|--------|--------|--------|--------|
| 还款能力 | `has_ability_confirmed` | True/False | ⭐⭐⭐⭐⭐ |
| 还款时间 | `payment_date_confirmed` | "2025-12-30" | ⭐⭐⭐⭐⭐ |
| 还款金额 | `payment_amount_confirmed` | "2000" / "全额" | ⭐⭐⭐⭐ |
| 付款方式 | `payment_type_confirmed` | "full" / "partial" | ⭐⭐⭐ |
| 展期请求 | `extension_requested` | True/False | ⭐⭐⭐ |

**收敛逻辑**：
- 每轮对话后，Memory Layer 更新收敛状态
- Layer 3 分析收敛进度（如"3轮对话仅收集到1个信息"）
- Layer 1 根据收敛进度调整策略优先级

---

### 2. Layer 1: StrategyManager（策略层）

#### 核心职责
- 根据客户档案、历史记录、配置规则生成初始催收策略
- 定义"多步收敛路径" (Step0-7)，指导 Layer 2 的执行
- 接收 Layer 3 的评估反馈，动态调整策略

#### 关键方法

**1. generate_initial_strategy(customer_profile) → str**

输入：
- `customer_profile`：客户基本信息（姓名、金额、逾期天数、性别、年龄、借款频率等）
- `self.history_logs`：历史催收记录
- `self.config`：配置文件规则

输出格式：
```
【历史分析】
（分析客户过往态度、承诺、能力、意愿）

【今日临时催收策略】
1. 沟通基调：...
2. 重点强调的内容：...

【多步收敛路径】
Step0: 查阅聊天历史，延续上次对话...
Step1: 确认还款能力 - 追问客户当前是否有钱可还
Step2: 确认还款时间 - 具体哪天还
Step3: 确认还款金额 - 能还多少（全额/部分）
Step4: 确认付款方式 - 如果是部分，剩余如何处理
Step5: 锁定承诺 - 记录为正式还款计划
Step6: 如果不能还，问清楚原因
Step7: 如果坚持不还，开始施压（信用分、黑名单、紧急联系人...）

⚠️ 每步等客户明确回答后再进入下一步
```

**2. update_strategy(current_strategy, feedback, chat_history, customer_profile, layer3_advice) → str**

触发条件：
- Layer 3 评估回款可能性为 `LOW`
- 信息收敛效率低

调整内容：
- **沟通基调**：根据评估结果调整压力等级
- **收敛路径优先级**：优先追问缺失的关键信息
- **话术和施压手段**：根据客户类型（高频/低频借款者）调整策略

---

### 3. Layer 2: Executor（执行层）

#### 核心职责
- 将 Layer 1 的策略转化为自然的对话
- 根据记忆上下文判断应该收集哪个信息
- 按"多步收敛路径"逐步推进对话
- 根据客户意图（0/1）调整语气和压力

#### 系统提示词结构

```python
combined_system_prompt = f"""
{base_prompt}  # 基础角色定义

# KEY CONTEXT
1. **HISTORY**: {history_logs}  # 历史记录
2. **CLIENT STATE**: {memory_context}  # 记忆状态（包含收敛进度）
3. **TODAY'S STRATEGY**: {strategy}  # Layer 1 的策略
4. **CONFIG RULES**: {config}  # 配置规则

# INSTRUCTIONS
- 执行策略中的"多步收敛路径"
- 检查记忆状态，优先收集缺失信息（⏳）
- 根据意图调整语气：
  * intent=1（有意愿）→ 协作、支持
  * intent=0（无意愿）→ 施压、警告
- 每次只问一个问题，保持自然

# OUTPUT (必须是有效 JSON)
{{
  "user_analysis": "客户态度分析",
  "strategy_check": "引用策略中适用的步骤",
  "tactical_plan": "根据收敛进度，本轮收集哪个信息",
  "response": "最终中文回复（自然、专业）"
}}
"""
```

#### 执行流程

1. **接收输入**
   - 策略（strategy）
   - 对话历史（chat_history）
   - 用户消息（user_input）
   - 历史记录（history_logs）
   - 记忆上下文（memory_context）← **关键**

2. **构建消息链**
   ```python
   messages = [
       {"role": "system", "content": combined_system_prompt},
       ...chat_history,
       {"role": "user", "content": user_input}
   ]
   ```

3. **调用 OpenAI API**
   - 启用 JSON 模式（`response_format: json_object`）
   - 温度设为 0.7（保持自然性）

4. **解析响应**
   - 提取 `response`（给客户的回复）
   - 提取 `thought`（内部思考过程，显示在 UI）

---

### 4. Layer 3: Evaluator（评估层）

#### 核心职责
- 评估当前策略的有效性
- 分析回款可能性（HIGH/MEDIUM/LOW）
- 跟踪信息收敛进度
- 向 Layer 1 提供优化建议

#### 评估维度

**1. 回款可能性分级**
- **HIGH**：客户明确承诺，能力充足，时间确定
- **MEDIUM**：客户有意愿但存在障碍，或信息不完整
- **LOW**：客户拒绝、拖延、失约多次、无能力

**2. 信息收敛进度分析**
```
已收集：[能力(有钱)、时间(2025-12-30)]
未收集：[金额、方式、展期]
收敛效率：3轮对话仅收集到2个信息 → 效率偏低
```

**3. 策略有效性评估**
- 当前策略是否有效推进信息收集？
- 客户的抗拒点或困难是什么？
- 是否需要调整沟通基调或施压手段？

#### 输出格式

```
【分析】
客户在本轮对话中表现出...抗拒点是...当前策略的...

【回款可能性】LOW

【信息收敛进度】
已收集：[能力(未确认)]
未收集：[时间、金额、方式、展期]

【收敛效率评估】
收敛速度慢，3轮对话仅收集到0-1个明确信息

【给 Layer 1 的建议】
建议调整策略：
1. 在"多步收敛路径"中优先追问还款能力和时间
2. 增加压力等级，提及信用影响
3. 如果客户持续拖延，启动 Step7 施压措施
```

---

## 数据流分析

### 完整对话流程

```
┌─────────────┐
│  用户输入    │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────────┐
│  Memory Layer.extract_from_dialogue()   │
│  ├─ detect_payment_intent() → 0/1      │
│  ├─ 能力评估 (full/partial/zero)        │
│  ├─ 原因分类                            │
│  ├─ 日期/金额提取                       │
│  └─ 更新收敛状态                        │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│  Memory Layer.get_memory_context()      │
│  生成格式化的记忆摘要                    │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│  Layer 3.evaluate()                     │
│  输入：对话历史 + 记忆上下文 + 策略      │
│  输出：评估报告（可能性 + 收敛分析）     │
└──────┬──────────────────────────────────┘
       │
       ↓
    ┌──────┐
    │ LOW? │ ────Yes──→ ┌─────────────────────────┐
    └──┬───┘            │ Layer 1.update_strategy()│
       │                │ 根据 Layer 3 建议调整策略 │
       No               └──────────┬──────────────┘
       │                           │
       └───────────┬───────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│  Layer 2.execute()                      │
│  输入：策略 + 记忆上下文 + 历史          │
│  输出：AI 回复 + 思考过程               │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────┐
│  显示在 UI   │
│  - 聊天内容  │
│  - 思考过程  │
│  - 评估报告  │
│  - 记忆状态  │
└─────────────┘
```

### 关键数据传递

| 数据项 | 来源 | 传递给 | 用途 |
|--------|------|--------|------|
| `history_logs` | 用户输入（sidebar） | Layer 1, 2, 3 | 提供历史背景 |
| `customer_profile` | 用户输入（JSON） | Layer 1, 3 | 客户基本信息 |
| `memory_context` | Memory Layer | Layer 1, 2, 3 | 当前客户画像 + 收敛进度 |
| `strategy` | Layer 1 | Layer 2, 3 | 催收策略和路径 |
| `chat_history` | Session State | Layer 2, 3 | 完整对话上下文 |
| `evaluation_output` | Layer 3 | Layer 1, UI | 评估结果和建议 |

---

## 关键功能

### 1. 智能意图识别

**问题**：传统规则引擎难以准确判断客户是否有还款意愿。

**解决方案**：使用 LLM 进行意图分类

```python
def detect_payment_intent(self, user_msg: str) -> int:
    system_prompt = """判断用户对"今天还钱"的意图。
    输出：
    - 1：用户表示今天会还钱
    - 0：用户明确表示今天不会还钱
    """
    result = self.llm_caller(user_msg, system_prompt, json_mode=False)
    return int(result.strip())
```

**优势**：
- 理解隐含意图（"我在忙" → 0）
- 处理模糊回答（"我会尽快" → 1）
- 动态适应不同表达方式

### 2. 历史记录智能解析

**问题**：历史记录是非结构化文本，难以提取关键信息。

**解决方案**：LLM 自动分析 + JSON 结构化输出

```python
def parse_history_summary(self, history_text: str):
    system_prompt = """分析历史记录，提取：
    - summary: 100-200字摘要
    - broken_promises: 失约次数
    - reason_category: 原因分类
    - ability_score: 能力评估
    
    输出严格 JSON 格式。"""
    
    result = self.llm_caller(history_text, system_prompt, json_mode=True)
    data = json.loads(result)
    # 更新到 memory
```

**优势**：
- 自动提取失约次数、原因
- 生成可读性强的摘要
- 一次性解析多天记录

### 3. 信息收敛追踪

**核心思想**：将催收目标拆解为5个可量化的信息点，逐步收集。

```python
# 收敛状态示例
{
    "has_ability_confirmed": True,      # ✅
    "payment_date_confirmed": "2025-12-30",  # ✅
    "payment_amount_confirmed": "",     # ⏳ 未确认
    "payment_type_confirmed": "",       # ⏳ 未确认
    "extension_requested": False        # ✅
}
```

**UI 展示**：
```
🎯 关键信息收敛进度
✅ 还款能力: 已确认 (partial)
✅ 还款时间: 2025-12-30
⏳ 还款金额: 未确认
⏳ 付款方式: 未确认
✅ 展期请求: 否
```

**Layer 3 分析**：
```
【信息收敛进度】
已收集：[能力(部分)、时间(2025-12-30)、展期(否)]
未收集：[金额、方式]

【收敛效率评估】
收敛速度中等，3轮对话收集到3个信息
```

**Layer 1 响应**：
```
【多步收敛路径】（根据收敛进度调整）
Step1: ✅ 已完成 - 能力确认
Step2: ✅ 已完成 - 时间确认
Step3: ⚠️ 优先 - 追问还款金额（全额 ¥1,250,000 还是部分？）
Step4: 待收集 - 付款方式
```

### 4. 动态策略调整

**触发机制**：
```python
is_low_prob = "LOW" in evaluation_output
if is_low_prob:
    new_strategy = layer1.update_strategy(
        current_strategy,
        user_input,
        chat_history,
        customer_profile,
        evaluation_output  # 包含收敛分析
    )
    st.session_state.strategy = new_strategy
```

**调整维度**：
1. **沟通基调**：协作型 ↔ 施压型
2. **收敛路径优先级**：重新排序 Step1-7
3. **话术策略**：针对高频/低频借款者差异化
4. **施压手段**：信用分 → 黑名单 → 紧急联系人 → 上门

### 5. 可视化分析面板

**布局**：
- **左侧（60%）**：聊天界面
- **右侧（40%）**：分析面板
  - 📋 今日策略（Layer 1）
  - 👤 客户记忆（Memory Layer）
  - 💭 思考过程（Layer 2）
  - 🛡️ 评估报告（Layer 3）

**实时更新**：
- 每轮对话后，记忆状态自动更新
- 收敛进度用 ✅/⏳ 图标标识
- 策略更新时显示 🔄 标记

---

## 配置说明

### 配置文件格式（YAML）

```yaml
# configs/T0.yaml 示例
system_prompt: |
  你是一个专业的债务催收顾问...

collection_strategy:
  initial_approach: "礼貌提醒"
  escalation_levels:
    - level: 1
      tone: "友好"
      actions: ["提醒逾期", "询问困难"]
    - level: 2
      tone: "严肃"
      actions: ["强调后果", "提及信用影响"]
    - level: 3
      tone: "强硬"
      actions: ["联系紧急联系人", "法律警告"]

constraints:
  - "遵守法律法规"
  - "不得辱骂威胁"
  - "保护客户隐私"
```

### 环境变量

```bash
# .env 或 .streamlit/secrets.toml
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
```

### 客户档案格式

```json
{
  "name": "LESTARI",
  "amount": "Rp 1.250.000",
  "due_date": "2025-12-17",
  "current_time": "2025-12-17",
  "gender": "Male",
  "age": "30",
  "frenquency_borrow": "often",          // 借款频率
  "payment_timelyness": "sometimes late",  // 还款及时性
  "meaber_level": "high"                  // 会员等级
}
```

### 历史记录格式

```
【YYYY-MM-DD 第X次催收】
- HH:MM 发生了什么
- HH:MM 客户回应
- HH:MM 结果

【分析】
总结性的分析文字...
```

---

## 使用指南

### 快速启动

1. **安装依赖**
```bash
pip install streamlit openai pyyaml
```

2. **配置 API Key**
```bash
export OPENAI_API_KEY=sk-...
# 或在 .streamlit/secrets.toml 中配置
```

3. **运行应用**
```bash
streamlit run app_easy.py
```

### 操作流程

**Step 1: 配置客户信息**
1. 在左侧边栏选择配置文件（如 `T0.yaml`）
2. 编辑客户档案 JSON（姓名、金额、逾期日期等）
3. 编辑历史记录（过往催收情况）

**Step 2: 系统自动初始化**
- 系统自动解析历史记录（`parse_history_summary`）
- Layer 1 生成初始策略
- Layer 2 生成开场白
- 右侧面板显示策略和记忆状态

**Step 3: 对话交互**
1. 在聊天框输入客户回复（模拟场景）
2. 系统自动：
   - 更新记忆状态（意图、能力、收敛进度）
   - Layer 3 评估
   - Layer 2 生成回复
3. 查看右侧分析面板：
   - 👤 记忆状态变化
   - 🎯 收敛进度更新
   - 💭 AI 思考过程
   - 🛡️ 评估报告

**Step 4: 策略调整**
- 当 Layer 3 评估为 `LOW` 时，系统自动更新策略
- 右侧显示 🔄 策略更新标记
- 可手动点击"Regenerate Strategy"重新生成

**Step 5: 重置会话**
- 点击左侧边栏"Reset Session"按钮
- 清空对话历史和记忆状态
- 重新开始新的催收场景

### 使用技巧

1. **充分利用历史记录**
   - 提供详细的过往催收记录
   - 包含时间、客户承诺、失约情况
   - 系统会自动提取关键信息

2. **观察收敛进度**
   - 重点关注 ⏳ 标记的未确认信息
   - Layer 2 会优先收集这些信息
   - 收敛越快，回款可能性越高

3. **策略调整时机**
   - 当连续多轮无进展时，考虑手动重新生成策略
   - 当客户态度突然转变时，观察系统是否自动调整

4. **日志分析**
   - 终端日志会显示每步操作（时间戳 + 描述）
   - 可用于调试和性能分析

---

## 扩展开发

### 添加新的记忆字段

```python
# 1. 在 MemoryLayer.__init__ 中添加字段
self.memory = {
    # ... 现有字段
    "new_field": default_value
}

# 2. 在 extract_from_dialogue 中提取逻辑
if "某关键词" in user_msg:
    self.memory["new_field"] = extracted_value

# 3. 在 get_memory_context 中显示
summary += f"\n- 新字段: {self.memory.get('new_field', '未知')}"

# 4. 在 UI 中展示（main() 函数）
st.write(f"**新字段**: {memory_dict.get('new_field', '未知')}")
```

### 自定义评估维度

```python
class Layer3Evaluator:
    def evaluate(self, ...):
        # 在 system_prompt 中添加新的评估维度
        system_prompt = """
        ...
        【新维度】评估客户的XXX情况
        """
        
        # 在输出格式中添加对应字段
        """
        【新维度评估】...
        """
```

### 扩展收敛信息

```python
# 1. 在 Memory Layer 中添加新的收敛字段
"new_convergence_field": "",

# 2. 在 extract_from_dialogue 中提取
if "某条件" in user_msg:
    self.memory["new_convergence_field"] = value

# 3. 在 get_memory_context 的收敛进度中显示
convergence_status += f"\n✓ 新信息: {self.memory.get('new_convergence_field') or '未确认'}"

# 4. 更新 Layer 1 策略模板，添加新的 Step
【多步收敛路径】
...
Step6: 确认新信息 - 追问...
```

### 集成外部数据源

```python
# 示例：集成数据库查询客户历史
class MemoryLayer:
    def __init__(self, llm_caller, db_connection=None):
        self.db = db_connection
        # ...
    
    def load_from_database(self, customer_id):
        # 查询数据库
        history = self.db.query(f"SELECT * FROM collection_history WHERE customer_id={customer_id}")
        # 调用 parse_history_summary
        self.parse_history_summary(history.to_text())
```

### 多语言支持

```python
# 1. 在配置文件中添加语言字段
language: "id"  # 印尼语

# 2. 在 Layer 2 的 system_prompt 中指定
"""
Respond in {config['language']} language.
"""

# 3. 在 Memory Layer 的提取逻辑中适配关键词
KEYWORDS = {
    "zh": {"没钱": "zero", "有钱": "full"},
    "id": {"tidak ada uang": "zero", "ada uang": "full"}
}
```

---

## 技术细节

### LLM 调用策略

**1. JSON 模式 vs 普通模式**
- **JSON 模式**：Layer 2, Memory.parse_history_summary
  - 确保输出格式严格
  - 便于解析和错误处理
- **普通模式**：Layer 1, Layer 3, Memory.detect_payment_intent
  - 需要更灵活的文本输出
  - 格式不太严格的场景

**2. 温度参数**
```python
temperature = 0.7  # 所有调用统一使用
# 0.7 是平衡性和创造性的最佳值
# 太低（0.1）→ 回复重复、机械
# 太高（0.9）→ 回复不稳定、跑题
```

**3. 错误处理**
```python
try:
    response = client.chat.completions.create(...)
except json.JSONDecodeError as e:
    log(f"JSON 解析失败: {e}")
    return fallback_value
except Exception as e:
    log(f"LLM 调用失败: {e}")
    st.error(f"错误: {str(e)}")
```

### 性能优化

**1. 缓存策略**
- 历史解析结果缓存在 `session_state.memory`
- 策略缓存在 `session_state.strategy`
- 避免重复调用 LLM

**2. 并发控制**
- Layer 3 评估和 Memory 更新可以并行
- 但 Layer 1 更新必须在 Layer 3 之后

**3. Token 优化**
- 历史记录截断（仅保留最近 N 条）
- 策略文本长度限制
- 记忆摘要控制在 500 字以内

### 安全性考虑

**1. API Key 保护**
```python
# 优先级：环境变量 > Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("请配置 API Key")
    st.stop()
```

**2. 输入验证**
```python
try:
    customer_profile = json.loads(profile_str)
except json.JSONDecodeError:
    st.sidebar.error("客户档案 JSON 格式错误")
    customer_profile = default_profile
```

**3. 日志脱敏**
```python
def log(msg):
    # 避免记录敏感信息（姓名、金额）
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
```

---

## 最佳实践

### 1. 策略设计

**DO**：
- 明确定义每个 Step 的目标
- 保持策略灵活性（"根据情况调整"）
- 包含多种施压等级（温和 → 严厉）

**DON'T**：
- 策略过于死板（"必须按顺序执行"）
- 一次性问太多问题
- 忽略客户的个体差异

### 2. 记忆维护

**DO**：
- 每轮对话后立即更新记忆
- 记录客户的原话（reason_detail）
- 追踪失约次数和拒付次数

**DON'T**：
- 覆盖历史信息
- 记录过多无关信息
- 忽略客户态度变化

### 3. 评估标准

**DO**：
- 综合考虑意图、能力、历史
- 关注收敛进度（信息完整度）
- 及时触发策略调整

**DON'T**：
- 仅凭单次对话判断
- 忽略收敛效率
- 过度依赖自动调整

### 4. UI/UX

**DO**：
- 清晰展示关键信息（收敛进度、意图）
- 提供完整的思考过程透明度
- 支持手动重置和重新生成

**DON'T**：
- 隐藏重要的分析结果
- 过度自动化（无法干预）
- UI 过于复杂

---

## 常见问题

### Q1: 系统一直评估为 LOW 怎么办？

**原因**：
- 客户确实无还款意愿
- 历史记录显示多次失约
- 信息收敛进度过慢

**解决方案**：
1. 检查历史记录是否准确
2. 手动重新生成策略
3. 调整配置文件的施压等级
4. 考虑切换到更激进的配置（如 `aggressive_collection.yaml`）

### Q2: 收敛进度一直是 ⏳ 怎么办？

**原因**：
- 客户回复模糊或避而不答
- Layer 2 没有紧扣策略执行
- 提取逻辑未能识别关键词

**解决方案**：
1. 检查 `extract_from_dialogue` 的关键词匹配
2. 在 Layer 1 策略中更明确地要求追问
3. 增加 Memory Layer 的提取规则

### Q3: Layer 2 输出的是英文怎么办？

**原因**：
- 系统提示词中未明确要求中文
- 模型默认语言是英文

**解决方案**：
```python
# 在 Layer2Executor.execute() 的 system_prompt 中添加
"""
IMPORTANT: 你必须用中文回复客户。
"""
```

### Q4: 历史解析失败（JSON 错误）

**原因**：
- LLM 返回格式不符合 JSON 规范
- 历史文本过长或格式特殊

**解决方案**：
```python
try:
    data = json.loads(result)
except json.JSONDecodeError:
    log(f"JSON 解析失败，原始内容: {result[:200]}")
    # 使用默认值
    data = {"summary": "", "broken_promises": 0, ...}
```

### Q5: 如何处理多天历史记录？

**最佳实践**：
```
【2025-12-26】
...

【2025-12-27】
...

【2025-12-28】
...

【综合分析】
客户累计失约3次，主要原因是...
```

系统会自动提取所有日期的信息，并生成综合分析。

---

## 更新日志

### Version 1.0（当前版本）

**核心功能**：
- ✅ 三层架构（策略、执行、评估）
- ✅ 记忆层（意图识别、信息提取、历史解析）
- ✅ 信息收敛机制（5个关键信息追踪）
- ✅ 动态策略调整
- ✅ 可视化分析面板

**已知限制**：
- 仅支持单客户对话（不支持多客户并发）
- 历史记录需手动输入（未集成数据库）
- 配置文件功能未完全利用

**未来计划**：
- 集成数据库（客户档案、历史记录）
- 多客户管理
- 策略模板库
- A/B 测试框架
- 回款效果统计

---

## 附录

### A. 关键词库

**能力评估**：
- `full`: "全额", "所有", "全部", "一次性"
- `partial`: "部分", "一点", "一些", "先还"
- `zero`: "没钱", "无力", "没办法", "钱不够"

**原因分类**：
- `unemployment`: "失业", "没工作", "裁员", "收入"
- `illness`: "生病", "医疗", "健康", "住院"
- `forgot`: "忘记", "忘了", "没想起"
- `malicious_delay`: "拒绝", "不想", "拖延", "不配合"

**障碍识别**：
- "开车", "忙", "会议", "睡觉", "孩子", "病", "手机", "网络"

**日期模式**：
- "明天", "后天", "X号", "X月X日", "YYYY-MM-DD"

### B. 系统日志示例

```
[09:15:32] Analyzing user intent and building memory...
[09:15:34] Calling LLM... Model: gpt-4o-mini, JSON_Mode: False
[09:15:36] LLM Response received.
[09:15:36] Memory updated - Intent:0, Date:, Amount:, Type:
[09:15:36] Layer 3 Evaluating...
[09:15:38] Low probability detected, updating strategy...
[09:15:42] Strategy updated successfully
[09:15:42] Layer 2 Thinking...
[09:15:45] Layer 2: Response received.
```

### C. 参考资源

- [Streamlit 文档](https://docs.streamlit.io/)
- [OpenAI API 文档](https://platform.openai.com/docs/)
- [YAML 语法指南](https://yaml.org/)

---

## 联系与支持

- **项目路径**：`/Users/mac/Documents/GitHub/longmemo/app_easy.py`
- **配置目录**：`/Users/mac/Documents/GitHub/longmemo/configs/`
- **相关文档**：
  - `ARCHITECTURE.md` - 系统架构设计
  - `TESTING_GUIDE.md` - 测试指南
  - `UPGRADE_PLAN.md` - 升级计划

---

**文档版本**：1.0  
**最后更新**：2025-12-29  
**维护者**：GitHub Copilot
