# 📋 长期记忆催收代理项目 - 完整文档

## 项目概述

这是一个**智能债务催收 AI 代理系统**，采用**三层架构**设计，集成 Streamlit Web UI + OpenAI/百度 ERNIE 大模型，用于模拟真实的债权催收场景，并通过多轮对话自动调整催收策略。

**项目名称**: test_long_memo  
**技术栈**: Python + Streamlit + OpenAI API / Baidu ERNIE API + Pydantic  
**核心文件**: `test_gpt52_2_optimized.py` (1124 行)

---

## 🏗️ 三层架构核心设计

该项目实现了一套完整的 **催收智能体协调系统**：

### 层次结构流程图

```
用户输入
    ↓
┌─────────────────────────────────────┐
│ Layer 1: Critic (质检与决策评估)     │
│ - 评估当前策略是否有效               │
│ - 检测用户风险行为 (拒付、失约)      │
│ - 决定是否触发 Meta 重写策略         │
└─────────────────────────────────────┘
    ↓
    ├─ 决策: CONTINUE → 继续当前策略
    ├─ 决策: ADAPT_WITHIN_STRATEGY → 微调话术
    └─ 决策: ESCALATE_TO_META ↓
        │
        ┌─────────────────────────────────────┐
        │ Layer 2: Meta (元策略生成器)         │
        │ - 根据最新对话生成新的 StrategyCard │
        │ - 选择合适的施压等级                 │
        │ - 设计完整的对话流程                 │
        └─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Layer 3: Executor (话术执行器)       │
│ - 基于策略生成自然的用户回复         │
│ - 遵守合规硬约束                     │
│ - 输出最终的催收话术                 │
└─────────────────────────────────────┘
    ↓
返回给用户
```

---

## 📂 核心模块详解

### 1️⃣ **模型配置模块** (Line 13-23)

```python
MODEL_PROVIDERS = {
    "OpenAI": {
        "base_url": None,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "models": ["gpt-4o", "gpt-4o-mini", "o1-mini"]
    },
    "Baidu ERNIE": {
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_key": "...",
        "models": ["ernie-4.5-turbo-32k"]
    }
}
```

**功能**：支持多个 LLM 供应商的动态切换，用户可在 UI 中选择不同模型。

---

### 2️⃣ **业务规则模块** (Line 28-75)

#### 2.1 阶段计算函数
```python
def calculate_stage(dpd, broken_promises, payment_refusals) -> str
```

**输入参数**:
- `DPD`: Days Past Due（逾期天数）
- `broken_promises`: 失约次数（承诺还款但未履行）
- `payment_refusals`: 拒付次数（明确拒绝当天还款）

**计算公式**:
$$\text{总分} = \text{DPD} \times 10 + \text{broken\_promises} \times 15 + \text{payment\_refusals} \times 20$$

**阶段映射**:
| 总分 | 阶段 | 风险级别 | 施压策略 |
|------|------|----------|----------|
| < 0 | Stage0 | 提前期 | 正向激励 |
| 0 | Stage1 | 到期日 | 温和提醒 |
| 0-30 | Stage2 | 轻度逾期 | 温和+施压 |
| 30-60 | Stage3 | 中度逾期 | 强施压 |
| ≥ 60 | Stage4 | 严重逾期 | 最强施压+流程告知 |

#### 2.2 SOP 触发条件
```python
def sop_trigger_named_escalation(dpd, broken_promises) -> bool:
    return (broken_promises >= 1 and dpd > 3)
```

当**失约次数 ≥ 1 且逾期 > 3 天**时，触发升级（可联系紧急联系人、工作单位等）。

---

### 3️⃣ **数据模型（Pydantic Schemas）** (Line 82-165)

#### 3.1 MicroEdits（微调指令）
```python
class MicroEdits(BaseModel):
    ask_style: AskStyle                    # "open" / "forced_choice" / "binary"
    confirmation_format: ConfirmFmt        # 确认格式
    tone: Tone                             # "polite" / "polite_firm" / "firm"
    language: Lang                         # "zh" / "id"
```

由 Critic 生成，指导 Executor 的话术微调。

#### 3.2 CriticResult（质检结果）
```python
class CriticResult(BaseModel):
    decision: Decision                     # CONTINUE / ADAPT / ESCALATE / HANDOFF
    decision_reason: str                   # 详细理由
    reason_codes: List[str]               # 风险代码
    progress_events: List[str]            # 进度事件
    missing_slots: List[str]              # 缺失信息
    micro_edits_for_executor: MicroEdits  # 话术微调
    memory_write: Dict[str, Any]          # 内存更新
    risk_flags: List[str]                 # 风险信号
```

#### 3.3 StrategyCard（策略卡）
```python
class StrategyCard(BaseModel):
    strategy_id: str                       # 策略标识
    stage: str                             # 当前阶段 (Stage0-4)
    today_kpi: List[str]                  # 今日目标/步骤序列
    pressure_level: PressureLevel          # 施压等级
    allowed_actions: List[str]            # 允许执行的动作
    guardrails: List[str]                 # 合规红线
    escalation_actions_allowed: Dict      # 升级动作权限
    params: Dict[str, Any]                # 流程参数
    notes: Optional[str]                  # 备注
```

---

### 4️⃣ **提示词模板** (Line 172-547)

#### 4.1 合规硬约束 (COMPLIANCE_GUARDRAILS)
定义了不可违反的业务规则：
- ✅ 不虚构后果（只能告知真实流程）
- ✅ 不羞辱、恐吓、夸大法律后果
- ✅ DPD≥0 时遵守"当天闭环"（今天必须完成还款或时间点确认）
- ✅ 遇到反复推脱必须执行"二元收敛"：能还/不能还

#### 4.2 Critic 系统提示词
负责：
1. 评估对话**收敛性**（是否逼近"今天、金额、时间"）
2. 评估对话**有效性**（施压是否改变用户态度）
3. 评估对话**进阶性**（是否有新信息产出）
4. **行为检测**：
   - 拒付行为 (payment_refusals +1)
   - 失约行为 (broken_promises +1)
5. **增强检测**：
   - 死循环检测（重复借口）
   - 意图跳变检测（态度突变）
   - 失约触发检测（承诺时间已过）
   - 阶段不匹配检测

#### 4.3 Meta 系统提示词
负责：
1. 根据新的 memory_state 和 critic_result 生成新策略
2. **完整对话流程设计**（4步骤）：
   - Step 1: 尝试全额还款（第一优先）
   - Step 2: 根据用户回答动态分支
   - Step 3: 协商部分还款（今天）
   - Step 4: 如果全部拒绝则触发合规流程
3. **10级渐进式压力策略**：
   ```
   等级1-3: 正向激励（会员提升、额度提升、还款折扣）
   等级4-6: 信用警告（信用分影响、合作终止、黑名单）
   等级7-8: 第三方介入警告（紧急联系人、工作单位）
   等级9-10: 强制措施（社交媒体、上门催收）
   ```

#### 4.4 Executor 系统提示词
负责：
1. 生成自然流畅的催收话术
2. **必须遵守的要求**：
   - ✅ 不要重复已尝试的步骤
   - ✅ 承接上下文，基于用户回答进行回应
   - ✅ 优先回应用户最后说的那句话
   - ❌ 绝对禁止机械化强制（如"请只回答是或否"）
   - ✅ 通过自然问句达成确认（如"那咱们就定在今天下午两点，可以吗？"）

---

### 5️⃣ **LLM 调用模块** (Line 556-598)

#### 5.1 call_llm_text()
```python
def call_llm_text(system: str, user: str, temperature: float = 0.2) -> str
```
- 动态创建 OpenAI 客户端
- 支持不同供应商和模型的切换
- 返回 LLM 的文本输出

#### 5.2 build_history_summary()
```python
def build_history_summary(raw_text: str) -> Dict[str, Any]
```
- 接收原始聊天记录
- 调用"历史记录专家"系统提示词
- 返回结构化摘要：
  ```json
  {
    "summary": "100-200字摘要",
    "broken_promises": 2,
    "reason_category": "unemployment|illness|forgot|malicious_delay|other",
    "ability_score": "full|partial|zero",
    "reason_detail": "一句话主要借口"
  }
  ```

#### 5.3 clean_json_str()
```python
def clean_json_str(text: str) -> str
```
- 从 LLM 输出中提取 JSON
- 支持 ```json { } ``` 格式和原始 JSON 格式
- 处理各种垃圾字符

---

### 6️⃣ **会话管理** (Line 603-660)

#### 6.1 init_new_session()
初始化一个新的客户会话：

```python
{
    "customer_id": "C-001",
    "debt_amount": 10000.0,
    "dpd": 1,
    "broken_promises": 0,
    "payment_refusals": 0,
    "stage": "Stage2",
    "extension_eligible": False,
    "approval_id": "APR-001",
    
    # 客户画像记忆
    "reason_category": "",               # 原因分类
    "ability_score": "",                 # 能力评估
    "reason_detail": "",                 # 具体理由
    "unresolved_obstacles": [],          # 未解决的具体障碍
    
    # 对话与策略
    "dialogue": [],                      # 对话历史
    "strategy_card": None,               # 当前策略
    "last_critic": None,                 # 最后一次质检结果
    "history_summary": "",               # 历史记录摘要
}
```

---

### 7️⃣ **记忆管理** (Line 663-718)

#### 7.1 apply_memory_write()
```python
def apply_memory_write(memory: Dict, memory_write: Dict) -> Dict
```

**智能合并逻辑**：
- 列表字段（如 `unresolved_obstacles`）：去重累积
- 细节字段（如 `reason_detail`）：追加而非替换（用 `|` 分隔）
- 字典字段：深合并
- 状态字段（如能力、分类、阶段）：最新替换

#### 7.2 ensure_strategy_card()
```python
def ensure_strategy_card(memory_state: Dict, strategy_card: Optional[Dict]) -> Dict
```

**功能**：
1. 验证现有策略是否还有效
2. 检查策略的 stage 是否与内存中的 stage 匹配
3. 如果不匹配或无效，根据当前 Stage 生成默认策略

**按 Stage 生成的默认策略**：

| Stage | 策略ID | 目标 | 施压 |
|-------|--------|------|------|
| Stage0 | `{stage}_relationship_building` | 建立关系+正向激励 | polite |
| Stage1 | `{stage}_gentle_reminder` | 温和提醒+摸底 | polite |
| Stage2 | `{stage}_light_pressure` | 施压+收敛 | polite_firm |
| Stage3 | `{stage}_firm_pressure` | 强施压+二元收敛 | firm |
| Stage4 | `{stage}_maximum_pressure` | 最强施压+流程告知 | firm |

---

### 8️⃣ **三层协调函数** (Line 574-650)

#### 8.1 call_critic()
```python
def call_critic(strategy_card, memory_state, dialogue, history_summary) -> CriticResult
```

**工作流**：
1. 构建 Critic 系统提示词
2. 打包 (strategy_card, memory_state, recent_dialogue) 成 JSON payload
3. 调用 LLM 获得结构化决策
4. 异常时回退为 ESCALATE_TO_META

#### 8.2 call_meta()
```python
def call_meta(memory_state, critic, dialogue, history_summary) -> Dict
```

**工作流**：
1. 构建 Meta 系统提示词
2. 打包 (memory_state, critic_result, recent_dialogue) 成 JSON payload
3. 调用 LLM 生成新的 StrategyCard
4. 强制对齐 Stage（确保策略的 stage 与 memory_state.stage 一致）
5. 异常时回退为 fallback 策略

#### 8.3 call_executor()
```python
def call_executor(strategy_card, memory_state, dialogue, micro, history_summary) -> str
```

**工作流**：
1. 构建 Executor 系统提示词（包含完整的策略信息）
2. 打包 (strategy_card, memory_state, history_summary, micro_edits, recent_dialogue)
3. 调用 LLM 生成自然话术
4. 返回最终的用户回复

---

### 9️⃣ **主协调器** (Line 720-790)

#### 9.1 handle_turn()
```python
def handle_turn(user_msg: str)
```

**完整工作流**：

```
1. 将用户消息追加到对话历史
   state["dialogue"].append({"role": "user", "content": user_msg})

2. 刷新业务参数
   - 根据 (dpd, broken_promises, payment_refusals) 重算 Stage
   - 计算 SOP 触发条件

3. 确保策略有效
   state["strategy_card"] = ensure_strategy_card(state, state["strategy_card"])

4. ========== Critic 评估阶段 ==========
   critic = call_critic(strategy_card, memory_state, dialogue, history_summary)
   
5. ========== 记忆更新 + Stage 深度联动 ==========
   apply_memory_write(state, critic.memory_write)  # 更新记忆
   
   # 关键：根据 Critic 刚记下的新行为，立刻重算 Stage
   new_stage = calculate_stage(dpd_current, bp_current, pr_current)
   if new_stage != old_stage:
       # Stage 变了，强制叫醒 Meta（旧策略的压力等级可能不匹配）
       critic.decision = "ESCALATE_TO_META"

6. ========== Meta 策略生成（如需要）==========
   if critic.decision == "ESCALATE_TO_META":
       new_strategy = call_meta(state, critic, dialogue, history_summary)
       state["strategy_card"] = new_strategy

7. ========== Executor 话术生成 ==========
   reply = call_executor(strategy_card, memory_state, dialogue, micro, history_summary)
   state["dialogue"].append({"role": "assistant", "content": reply})

8. 存储本轮遥测数据
   state["last_turn_telemetry"] = telemetry
```

**关键创新**：Stage 深度联动
- Critic 检测到新的失约/拒付后，立刻重算 Stage
- 如果 Stage 升级了，强制触发 Meta 重写策略
- 确保压力等级与新 Stage 同步

---

## 🖥️ Streamlit UI 模块 (Line 795-1124)

### 页面布局

```
┌─────────────────────────────────────────────────────┐
│  三层架构催收大师 - 多会话 & 智能记忆版              │
└─────────────────────────────────────────────────────┘

┌─ 边栏 ─────────────────────────────────────────────┐
│ 🤖 模型引擎配置                                      │
│   - 供应商: OpenAI / Baidu ERNIE                    │
│   - 模型: gpt-4o-mini / ernie-4.5 等                │
│                                                     │
│ 📂 会话管理                                         │
│   - ➕ 创建新会话                                   │
│   - 🔄 选择活跃客户                                 │
│                                                     │
│ ⚙️ 客户基础配置                                     │
│   - 机构名称, 产品名称                              │
│   - 欠款金额, 货币单位                              │
└─────────────────────────────────────────────────────┘

┌─ 主区域 (2列布局) ────────────────────────────────┐
│                                                     │
│ 左列 (66%)                  │  右列 (34%)          │
│ ┌─────────────────────┐    │ ┌──────────────────┐ │
│ │ 💬 对话窗口         │    │ │ 👤 客户画像记忆  │ │
│ │ - 用户消息          │    │ │  - 原因分类      │ │
│ │ - 机构回复          │    │ │  - 能力评估      │ │
│ │                     │    │ │  - 具体理由      │ │
│ │ [发送] [清空] [删除] │    │ │                  │ │
│ │                     │    │ │ 📥 导入历史记录  │ │
│ │                     │    │ │  - 粘贴原文      │ │
│ │                     │    │ │  - 智能解析      │ │
│ │                     │    │ │  - 填充画像      │ │
│ │                     │    │ │                  │ │
│ │                     │    │ │ 📊 业务参数      │ │
│ │                     │    │ │  - DPD           │ │
│ │                     │    │ │  - 失约次数      │ │
│ │                     │    │ │  - 拒付次数      │ │
│ │                     │    │ │  - 可展期        │ │
│ │                     │    │ │                  │ │
│ │                     │    │ │ 🧠 策略核心      │ │
│ │                     │    │ │  - 策略ID        │ │
│ │                     │    │ │  - 今日目标(KPI) │ │
│ │                     │    │ │  - 完整策略视角  │ │
│ │                     │    │ │                  │ │
│ │                     │    │ │ ⛓️ Turn Pipeline │ │
│ │                     │    │ │  - Critic 观察   │ │
│ │                     │    │ │  - Meta 引擎     │ │
│ │                     │    │ │  - Executor动作  │ │
│ │                     │    │ │                  │ │
│ │                     │    │ │ 🧐 Critic质检    │ │
│ │                     │    │ │  - 决策          │ │
│ │                     │    │ │  - 理由          │ │
│ │                     │    │ │  - 风险信号      │ │
│ └─────────────────────┘    │ └──────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 右侧面板详解

#### 1. 👤 客户画像与记忆
- **原因分类** (reason_category)
  - 🚫 失业/收入
  - 🏥 疾病/健康
  - ❓ 忘记/疏忽
  - 👿 恶意拖延
  - ⚙️ 其他

- **能力评估** (ability_score)
  - ✅ 有能力全额
  - ⚠️ 仅能部分
  - ❌ 无力还款

#### 2. 📥 导入过往记录
```
粘贴聊天记录 → AI智能解析 → 填充客户画像
   ↓
提取: summary, broken_promises, reason_category,
      ability_score, reason_detail
```

#### 3. 📊 业务实时参数
- DPD (逾期天数)
- 历史失约次数
- 本次拒付次数
- 可展期 (checkbox)
- ➜ 实时显示当前阶段 & 风险分

#### 4. 🧠 策略核心
- 显示当前策略 ID 和施压等级
- 展开 "🎯 今日目标" (KPI 列表)
- 展开 "📜 Executor完整策略视角"

#### 5. ⛓️ System Turn Pipeline
可视化显示本轮执行流水线：
1. **Critic Observation**: 质检决策
2. **Meta Strategy Engine**: 是否需要重写策略
3. **Executor Action**: 话术生成状态

#### 6. 🧐 Critic质检记录
- 显示最后一次 Critic 的决策
- 显示决策理由
- 显示风险信号

---

## 💾 完整数据流示例

### 场景：用户说 "我没钱"

```
用户输入: "我现在没钱，你能等等吗？"
   ↓
state["dialogue"].append({
    "role": "user",
    "content": "我现在没钱，你能等等吗？"
})
   ↓
Critic 分析:
  - 检测到拒付行为 (DPD >= 0 时拒绝今天还款)
  - memory_write: {"payment_refusals": 1}
  - 触发行为检测: reason_codes = ["payment_refusal_detected"]
  - decision = "ESCALATE_TO_META" (因为是拒付行为)
   ↓
apply_memory_write 更新内存:
  - payment_refusals: 0 → 1
   ↓
Stage 深度联动重算:
  - dpd=1, broken_promises=0, payment_refusals=1
  - 新评分 = 1*10 + 0*15 + 1*20 = 30
  - 新 Stage = "Stage2" (保持)
   ↓
Meta 生成新策略:
  - decision == "ESCALATE_TO_META"
  - 生成新的 StrategyCard:
    {
      "strategy_id": "stage2_escalation_due_to_refusal",
      "stage": "Stage2",
      "today_kpi": [
        "step1_explore_underlying_reasons",
        "step2_assess_extension_or_partial",
        "step3_binary_convergence"
      ],
      "pressure_level": "polite_firm",
      "allowed_actions": [
        "ask_specific_reason",
        "offer_extension_if_eligible",
        "force_choice_today"
      ],
      "guardrails": [
        "today_only_for_dpd_ge_0",
        "no_fake_threats"
      ]
    }
   ↓
Executor 生成话术:
  系统提示词包含:
  - 策略: 要求在今天达成闭环 (全额/部分/展期)
  - 微调: tone="polite_firm", ask_style="binary"
  - 历史: 用户之前的借口信息
  - 指令: 探索具体原因而非接受"没钱"
   ↓
LLM 生成输出例:
  "我理解您现在面临困难。能否告诉我具体是什么原因导致
   您现在没有可用的资金？比如是失业、医疗支出还是其他？
   
   同时，让我们看看有没有其他办法可以解决今天的问题：
   如果您今天能凑出一部分（比如3000元），剩余的部分我们可以考虑
   展期处理。您觉得这样可以吗？"
   ↓
state["dialogue"].append({
    "role": "assistant",
    "content": "... (上述话术)"
})
   ↓
UI 更新:
  ✅ Critic 决策: ESCALATE_TO_META
  ✅ Meta 重写: stage2_light_pressure → stage2_escalation_due_to_refusal
  ✅ Executor 回复: 生成的话术
  ✅ Critic 质检: 检测到 payment_refusal，risk_flags = ["payment_refusal_detected"]
```

---

## 🔑 关键特性

### 1. **多模型支持**
- 支持 OpenAI GPT-4o / GPT-4o-mini / o1-mini
- 支持百度 ERNIE 4.5 Turbo 32K
- 实时切换，无需重启

### 2. **智能记忆系统**
- 自动解析历史聊天记录
- 提取原因分类、能力评估、具体理由
- 累积记录未解决的障碍 (unresolved_obstacles)
- 支持多轮对话的长期记忆

### 3. **行为检测**
- ✅ 拒付行为检测 (payment_refusals +1)
- ✅ 失约行为检测 (broken_promises +1)
- ✅ 死循环检测 (重复借口)
- ✅ 意图跳变检测 (态度突变)
- ✅ 失约触发检测 (承诺时间已过)

### 4. **动态策略调整**
- 根据对话进展自动升级/降级策略
- Stage 变化时自动重写策略
- 支持 5 阶段逐步升级的压力等级
- 支持 10 级细粒度压力控制

### 5. **合规约束**
- 当天闭环原则（DPD≥0 时必须完成）
- 禁止虚构后果
- 禁止恐吓、夸大、羞辱
- 二元收敛（能还/不能还）

### 6. **多会话支持**
- 支持同时管理多个客户会话
- 会话间独立的对话历史和策略
- 支持创建、切换、删除会话

### 7. **可视化流水线**
- 实时显示三层协调过程
- 显示策略变化和更新原因
- 显示每轮的关键决策点

---

## 📊 项目文件结构

```
longmemo/
├── test_gpt52_2_optimized.py    ← 核心文件（当前）
│                                   ├── 配置模块
│                                   ├── 业务规则
│                                   ├── 数据模型
│                                   ├── 提示词
│                                   ├── LLM 调用
│                                   ├── 会话管理
│                                   ├── 记忆管理
│                                   ├── 三层协调
│                                   └── Streamlit UI
│
├── test_gpt52_2.py              ← 之前版本
├── test_gpt52_2_copy.py         ← 备份
├── test_gpt52.py                ← 早期版本
├── app_easy.py                  ← 简化版应用
├── test_baidu.py                ← 百度 API 测试
│
├── configs/                      ← 催收配置文件库
│   ├── default_collection.yaml
│   ├── aggressive_collection.yaml
│   ├── flexible_collection.yaml
│   ├── gambler_followup_collection.yaml
│   ├── T0_free_collection.yaml
│   ├── T0.yaml
│   └── T1_dynamic_collection.yaml
│
├── requirements.txt              ← 依赖清单
├── README.md                     ← 项目说明
└── 优化总结.md                   ← 优化记录
```

---

## 🚀 使用指南

### 启动应用
```bash
streamlit run test_gpt52_2_optimized.py
```

### 初始配置
1. **设置 API 密钥**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **选择模型**
   - 在侧边栏选择 "OpenAI" 或 "Baidu ERNIE"
   - 选择具体模型

3. **创建客户会话**
   - 点击 "➕ 创建新会话"
   - 输入客户 ID（如 "C-001"）
   - 点击"创建会话"

### 使用流程
1. **配置客户信息**
   - 设置欠款金额、机构名称等
   - 设置 DPD、失约次数等业务参数

2. **导入历史记录** (可选)
   - 在 "📥 导入过往记录" 部分粘贴原文
   - 点击 "开始智能解析"
   - 系统自动填充客户画像

3. **开始对话**
   - 在 "💬 对话" 区域输入用户回复
   - 点击 "发送"
   - 系统自动执行三层协调
   - 返回机构话术

4. **监控策略变化**
   - 在右侧面板查看客户画像更新
   - 在 "⛓️ Turn Pipeline" 查看本轮执行过程
   - 在 "🧐 Critic 质检" 查看风险信号

---

## 🔧 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| UI 框架 | Streamlit | 快速构建 Web 界面 |
| 大模型 | OpenAI API / Baidu ERNIE | 三层协调 & 话术生成 |
| 数据验证 | Pydantic | 强类型数据模型 |
| 配置管理 | YAML / dotenv | 催收策略配置 |
| 编程语言 | Python 3.8+ | 核心开发语言 |

---

## 🎯 核心创新点

1. **三层协调架构**：Critic + Meta + Executor 形成完整的智能体闭环

2. **Stage 深度联动**：Critic 检测到新行为后，立刻重算 Stage，触发 Meta 重写

3. **智能内存管理**：支持列表累积、细节追加、字典深合并的混合更新策略

4. **完整对话流程设计**：Meta 不是简单地输出指令，而是设计多步骤的对话流程

5. **10 级渐进式压力**：从正向激励逐步升级到强制措施，兼顾有效性与合规性

6. **行为检测引擎**：自动检测拒付、失约、死循环、意图跳变等风险信号

7. **天然合规约束**：当天闭环、禁止虚构、二元收敛等硬约束内置在提示词中

---

## ✨ 总结

这个项目是一套**生产级别的 AI 催收代理系统**，通过三层架构和智能记忆，实现了：

- ✅ 自动化的债权催收话术生成
- ✅ 动态的策略调整（基于用户行为）
- ✅ 智能的行为检测（拒付、失约、死循环）
- ✅ 合规的风险控制（当天闭环、禁止虚构）
- ✅ 可视化的决策过程（Critic → Meta → Executor）

适用于金融科技、催收服务、风险管理等场景。

