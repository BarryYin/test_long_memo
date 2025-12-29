import os
import json
import datetime as dt
from typing import Literal, List, Dict, Any, Optional

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI

# =========================================================
# Model Engine Config (Supports OpenAI & Baidu)
# =========================================================
MODEL_PROVIDERS = {
    "OpenAI": {
        "base_url": None,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "models": ["gpt-4o", "gpt-4o-mini", "o1-mini"]
    },
    "Baidu ERNIE": {
        "base_url": os.getenv("BAIDU_BASE_URL", "https://qianfan.baidubce.com/v2"),
        "api_key": os.getenv("BAIDU_API_KEY"),
        "models": ["ernie-4.5-turbo-32k",]
    }
}

# Client initialization will be handled dynamically in call_llm_text


# =========================================================
# Business Rules (your definitions)
# =========================================================
def dpd_to_stage(dpd: int) -> str:
    """Legacy function for backward compatibility - delegates to calculate_stage"""
    return calculate_stage(dpd, 0, 0)


def calculate_stage(dpd: int, broken_promises: int = 0, payment_refusals: int = 0) -> str:
    """
    综合计算 Stage,考虑三个因素:
    1. DPD (Days Past Due) - 逾期天数
    2. broken_promises - 失约次数(承诺还款但未履行)
    3. payment_refusals - 拒付次数(明确拒绝当天还款)
    
    评分规则:
    - DPD < 0: 直接返回 Stage0(提前期,正向激励)
    - DPD * 10 + broken_promises * 15 + payment_refusals * 20 = 总分
    - 总分映射到 Stage1-4
    """
    # 特殊处理:DPD < 0(未到期,提前期)
    if dpd < 0:
        return "Stage0"
    
    # 计算风险评分
    dpd_score = dpd * 10
    promise_score = broken_promises * 15
    refusal_score = payment_refusals * 20
    total_score = dpd_score + promise_score + refusal_score
    
    # Stage 映射
    if total_score == 0:
        return "Stage1"  # 到期日,无不良记录
    elif total_score < 30:
        return "Stage2"  # 轻度风险
    elif total_score < 60:
        return "Stage3"  # 中度风险
    else:
        return "Stage4"  # 高风险


def sop_trigger_named_escalation(dpd: int, broken_promises: int) -> bool:
    # Your SOP trigger: broken_promises>=1 and dpd>3
    return (broken_promises >= 1 and dpd > 3)


# =========================================================
# Schemas (Critic + Meta) - Strongly Structured
# =========================================================
Decision = Literal["CONTINUE", "ADAPT_WITHIN_STRATEGY", "ESCALATE_TO_META", "HANDOFF"]
AskStyle = Literal["open", "forced_choice", "binary"]
ConfirmFmt = Literal["none", "amount_time_today", "reply_yes_no"]
Tone = Literal["polite", "polite_firm", "firm"]
Lang = Literal["zh", "id"]
PressureLevel = Literal["polite", "polite_firm", "firm"]


class MicroEdits(BaseModel):
    ask_style: AskStyle = "open"
    confirmation_format: ConfirmFmt = "none"
    tone: Tone = "polite"
    language: Lang = "zh"


class CriticResult(BaseModel):
    decision: Decision
    decision_reason: str
    reason_codes: List[str] = Field(default_factory=list)
    progress_events: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    micro_edits_for_executor: MicroEdits = Field(default_factory=MicroEdits)
    memory_write: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)


class StrategyCard(BaseModel):
    strategy_id: str
    stage: str  # "Stage0"..."Stage4"
    today_kpi: List[str] = Field(default_factory=list)
    pressure_level: PressureLevel = "polite"

    allowed_actions: List[str] = Field(default_factory=list)
    guardrails: List[str] = Field(default_factory=list)

    # If Stage4 + sop_trigger + approval_id exists, Meta can turn these on
    escalation_actions_allowed: Dict[str, Any] = Field(default_factory=dict)

    # For future extension without breaking strict schemas
    params: Dict[str, Any] = Field(default_factory=dict)

    notes: Optional[str] = None


def _forbid_additional_props(schema_node):
    """Recursively set additionalProperties=false for all object nodes."""
    if isinstance(schema_node, dict):
        if schema_node.get("type") == "object":
            schema_node.setdefault("additionalProperties", False)
        for v in schema_node.values():
            _forbid_additional_props(v)
    elif isinstance(schema_node, list):
        for x in schema_node:
            _forbid_additional_props(x)


def critic_json_schema() -> dict:
    schema = CriticResult.model_json_schema()
    _forbid_additional_props(schema)
    return {"name": "critic_result", "strict": True, "schema": schema}


def meta_strategy_json_schema() -> dict:
    schema = StrategyCard.model_json_schema()
    _forbid_additional_props(schema)
    return {"name": "strategy_card", "strict": True, "schema": schema}


# =========================================================
# Prompt Templates - OPTIMIZED
# =========================================================
COMPLIANCE_GUARDRAILS = """
合规硬约束(不可违反):
- 不得虚构不会执行的后果;只能告知真实流程(可被审计)。
- 不得羞辱、恐吓、夸大法律后果、公开曝光式表达。
- 涉及第三方/外访等动作,只能在sop_trigger=true且approval_id存在时,以流程告知口吻提及,且不得向第三方披露债务细节。
- DPD>=0时遵守"当天闭环":必须促成(今天全额)或(今天部分+今天时间点)或(若资格允许)今天完成展期。严禁接受"明天还"等任何跨天承诺。
- 遇到反复推脱/拒付,必须执行"二元收敛":先确认"今天能否支付(能/不能)"。若不能,则触发合规流程告知;若能,则强制二选一(金额/时间)。
- 每次只问1个主问题(最多附带1句极短确认)。
"""


def build_critic_system_prompt() -> str:
    return f"""你是催收对话的质检与策略适配评估器(Critic + Strategy-Fit Gate)。
你必须对齐【当前策略卡】与【对话进程】做门控决策:
- CONTINUE:继续当前策略
- ADAPT_WITHIN_STRATEGY:策略对但话术/问法需微调(不触发元策略)
- ESCALATE_TO_META:策略不适配/无进展/阶段需要切换 (触发元策略改写)
- HANDOFF:高风险合规/投诉/停止联系等

【策略质检标准 - 决定你是否要触发 ESCALATE_TO_META】

你必须根据以下三个维度评估当前策略执行情况：

1. **收敛性 (Convergence)**: 
   - 目标：对话是否在不断逼近“今天、由于什么原因、还多少钱、几点还”？
   - 风险信号：用户在兜圈子、扯皮,而 AI 被带偏,没有在回收有效信息。
   - 动作：若不收敛,必须 ESCALATE_TO_META 要求更强力的“二元收敛”策略。

2. **有效性 (Effectiveness)**:
   - 目标：当前的施压等级是否成功改变了用户的态度？
   - 风险信号：用户对当前的信用/流程警告“免疫”,态度依然敷衍或强硬。
   - 动作：若无效,必须 ESCALATE_TO_META 要求提高压力等级或切换策略维度。

3. **对话进阶 (Progress)**:
   - 目标：每一轮是否都有新的事实被确认为存储到 memory 中？
   - 风险信号：连续2轮对话没有产出任何关于“原因细节”或“还款意愿”的新信息。
   - 动作：若停滞,必须 ESCALATE_TO_META 重新设计切入点。

{COMPLIANCE_GUARDRAILS}

【关键行为检测 - 必须在 memory_write 中更新】
你必须检测用户的以下行为,并通过 memory_write 更新计数器:

1. **拒付行为检测** (payment_refusals):
   当用户在 DPD>=0 时以任何理由拒绝或无法"今天还款",包括但不限于:
   
   **明确拒绝类**:
   - "今天没钱" / "今天不能付" / "今天付不了"
   - "明天还" / "下周还" / "等工资发了再说"
   - "现在没办法" / "暂时还不了"
   - "我拒绝还" / "我不想还"
   
   **障碍类但实质拒绝**（如用户反复强调同一个借口，阻止今天还款）:
   - "银行卡掉了，无法转账" → 表明今天无法还款
   - "手机掉了，无法进行还款" → 表明今天无法还款
   - "在开会，等下再说" → 延迟还款（拒绝今天闭环）
   - "正在忙，稍后处理" → 延迟还款（拒绝今天闭环）
   
   **重复借口**（用户在2轮以上对话中使用同一个理由拒绝还款）:
   - 第一轮："银行卡掉了"
   - 第二轮："是的，卡还没找到，没办法转账"
   - 第三轮："对，还是因为卡掉了，无法还"
   → 这是重复拒绝，计数器 +1
   
   检测到后,在 memory_write 中设置:
   {{"payment_refusals": memory_state.payment_refusals + 1}}
   
   **重要**：只要用户在DPD>=0时无法/拒绝"今天完成还款"，无论理由是什么，都应增加payment_refusals。

2. **失约行为检测** (broken_promises):
   当用户之前承诺了还款(在对话历史或 history_summary 中),但本轮对话中:
   - 承认没有履行承诺
   - 再次推迟还款时间
   - 找新的借口拖延
   
   检测到后,在 memory_write 中设置:
   {{"broken_promises": memory_state.broken_promises + 1}}

3. **正向行为检测**:
   如果用户表示"现在就还" / "马上处理" / "正在转账",可以在 progress_events 中记录。

【增强检测逻辑 - 触发 ESCALATE_TO_META 的条件】

4. **死循环检测 + 自动增加拒付次数**:
   如果用户在最近2-3轮对话中重复使用同一个借口(如连续说"银行卡掉了"、"手机掉了"、"没钱"等),且没有提供新的信息:
   - 设置 reason_codes: ["dead_loop_detected"]
   - 在 memory_write 中增加 payment_refusals: {{"payment_refusals": memory_state.payment_refusals + 1}}
   - decision 应为 ESCALATE_TO_META
   - decision_reason 中说明检测到的重复模式，以及自动增加了拒付次数
   
   例：用户在第1、2、3、4轮都说"银行卡掉了"
   → 识别到死循环
   → payment_refusals += 1
   → Stage 自动提升（例如 Stage2 → Stage3）
   → 策略变为更强硬的催收

5. **意图跳变检测**:
   如果用户的意图突然改变(如从"没钱"突然变成"想办延期",或从"明天还"变成"申请减免"):
   - 设置 reason_codes: ["intent_shift_detected"]
   - decision 应为 ESCALATE_TO_META
   - 在 decision_reason 中说明意图变化

6. **失约触发检测**:
   如果用户之前承诺了具体时间点(如"今天14:00还款"),但从对话上下文判断该时间已过且用户未履行:
   - 设置 reason_codes: ["promise_broken_time_passed"]
   - decision 应为 ESCALATE_TO_META
   - broken_promises 计数器 +1

7. **阶段不匹配检测**:
   如果用户给出的理由明确表明当前策略目标无法达成(如策略要求"全额还款",但用户说"只能还一半"):
   - 设置 reason_codes: ["strategy_goal_unachievable"]
   - decision 应为 ESCALATE_TO_META
   - 在 decision_reason 中说明不匹配的原因

【增量记忆写入 - 提取关键事实】

你必须从对话中提取以下关键信息,并写入 memory_write:

**✅ 必须执行的步骤：**
- 每次用户回复时，检查是否包含以下任何信息：
  1. 用户提到了不能或有困难还钱的原因？→ 写入 `reason_category` 和 `reason_detail`
  2. 用户提到了具体的障碍（如工具缺失、时间冲突等）？→ 写入 `unresolved_obstacles`
  3. 用户表达了还钱的能力程度？→ 写入 `ability_score`
  
**⚠️ 常见错误（必须避免）：**
- ❌ 错误：Critic 在 `decision_reason` 中提到了信息，但没有写入 `memory_write`
- ✅ 正确：Critic 同时在 `decision_reason` 中说明发现了什么，并在 `memory_write` 中记录数据

**例子：**
用户说："我的银行卡掉了，所以现在没法转账，但我后天工资会发，那时候就能还你。"

你应该输出：
```json
{{
  "decision": "ADAPT_WITHIN_STRATEGY",
  "decision_reason": "用户提到具体障碍（银行卡丢失）和未来还款时间。需要：1) 确认卡片是否能补办 2) 确认后天是否确实能还 3) 询问是否有替代还款方式",
  "reason_codes": [],
  "progress_events": ["obstacle_detected: 银行卡丢失", "future_promise_detected: 后天工资"],
  "missing_slots": ["补办卡时间", "后天具体还款时间", "替代还款方式"],
  "micro_edits_for_executor": {{"ask_style": "open", "tone": "polite"}},
  "memory_write": {{
    "reason_category": "other",
    "reason_detail": "银行卡丢失，无法现在转账，但后天工资发放后能还",
    "unresolved_obstacles": ["银行卡丢失"],
    "ability_score": "partial",
    "payment_refusals": 1
  }},
  "risk_flags": []
}}
```

**例子2：用户反复说同一个借口（死循环）**
用户在第1轮说"银行卡掉了"
用户在第2轮说"是的，卡还没找到"
用户在第3轮说"还是没有卡，无法转账"

你应该输出：
```json
{{
  "decision": "ESCALATE_TO_META",
  "decision_reason": "⚠️ 死循环检测：用户在3轮对话中反复强调'银行卡掉了'这同一个理由，无法提供解决方案或替代方式，实质是在拒绝今天还款。自动增加拒付次数 payment_refusals += 1。建议提升催收强度。",
  "reason_codes": ["dead_loop_detected"],
  "progress_events": ["repeating_excuse_detected: 银行卡丢失"],
  "missing_slots": ["替代还款方式", "卡片补办进度"],
  "micro_edits_for_executor": {{"ask_style": "open", "tone": "polite_firm"}},
  "memory_write": {{
    "reason_category": "other",
    "reason_detail": "银行卡丢失，无法现在转账，但后天工资发放后能还",
    "unresolved_obstacles": ["银行卡丢失"],
    "ability_score": "zero",
    "payment_refusals": 1
  }},
  "risk_flags": ["repeating_excuse"]
}}
```

8. **reason_category** (用户不还款的原因分类):
   - "unemployment", "illness", "forgot", "malicious_delay", "other"
   
9. **ability_score** (用户的还款能力评估):
   - "full", "partial", "zero"

10. **reason_detail** (具体理由摘要):
    - 用一句话摘要用户本次给出的具体借口细节。
    - 示例: "正在带孩子,双手没空处理转账"

11. **unresolved_obstacles** (具体障碍列表):
    - 提取用户提到的阻碍还款的**具体行为/场景动作**。
    - 示例: ["正在带孩子", "正在开车", "手机没电了"]

【关键：memory_write 必须包含以下内容】
你必须在每次有新信息时将其写入 memory_write。示例：
{{
  "memory_write": {{
    "reason_category": "forgot",  # 如果用户提到原因，必须分类
    "reason_detail": "银行卡掉了，无法转账",  # 用户提到的具体理由
    "unresolved_obstacles": ["银行卡丢失"],  # 用户提到的具体障碍
    "ability_score": "full"  # 如果能判断能力，也要写
  }}
}}

输出必须是严格JSON,且只输出JSON。格式如下:
{{
  "decision": "CONTINUE" | "ADAPT_WITHIN_STRATEGY" | "ESCALATE_TO_META" | "HANDOFF",
  "decision_reason": "详细的决策理由",
  "reason_codes": ["code1", "code2"],
  "progress_events": ["event1"],
  "missing_slots": ["slot1"],
  "micro_edits_for_executor": {{
    "ask_style": "open" | "forced_choice" | "binary",
    "confirmation_format": "none" | "amount_time_today" | "reply_yes_no",
    "tone": "polite" | "polite_firm" | "firm",
    "language": "zh" | "id"
  }},
  "memory_write": {{"reason_category": "", "reason_detail": "", "unresolved_obstacles": [], "ability_score": ""}},
  "risk_flags": ["flag1"]
}}
"""


def build_meta_system_prompt() -> str:
        # 注意: 此处不要使用 f-string, 避免 JSON 花括号被格式化错误
        return """你是元策略生成器(Meta / Controller)。
输入:memory_state, critic_result, recent_dialogue, history_summary。
输出:更新后的strategy_card(严格JSON)。

【核心原则】
- 策略是"活的",包含多个步骤的对话流程,而非单一指令
- **场景优先**: 检查 memory_state.unresolved_obstacles。如果存在未解决障碍,第一KPI必须是"确认障碍是否消除"。
- 必须遵守合规硬约束
- Stage必须与memory_state.stage一致
- DPD>=0时遵守"当天闭环"
- 参考history_summary避免重复被同一借口拖延

【对话流程设计 - 多步骤策略】

你设计的策略应该是一个**完整的对话流程**,包含多个步骤:

**Step 1: 尝试全额还款(永远是第一步)**
- 如果是刚开始对话，请总结下当前的情况，延续之前的聊天
- 先问用户能否全额还款
- 如果用户说"没钱全还",不要立即放弃
- 追问原因:为什么不能全额?具体差多少?
- 评估原因真实性(参考history_summary,看是否重复借口)
- 探索资金来源:能否向亲友借款?能否用信用卡?
- 强调全额还款的好处:避免罚息、保护信用

**Step 2: 根据用户回答,动态分支**
- 如果用户给出合理原因且确实无法全额:
    * 评估是否符合展期条件(extension_eligible=true, broken_promises<2, payment_refusals<2)
    * 如果符合 → 引导展期(今天完成申请)
    * 如果不符合 → 跳到Step 3
- 如果用户态度恶劣或重复借口:
    * 提高压力等级
    * 直接进入二元收敛:"今天能还还是不能还?"

**Step 3: 协商部分还款(今天)**
- 问用户今天能还多少
- 强调必须是"今天",不接受"明天"
- 金额必须是能力范围内的最大值
- 同步确认剩余金额的处理方案
- 确认具体时间点

**Step 4: 如果连部分都拒绝今天完成**
- 触发合规流程告知
- 告知后续处理流程
- 记录拒付行为

【根据Stage调整压力方式 - 10级渐进式压力策略】

不同Stage使用不同等级的压力策略,从正向激励逐步升级到强制措施:

**压力等级1-3 (正向激励) - 适用于Stage0-1**:
1. **会员等级提升**: "按时还款会获得良好的信用记录,提升您的会员等级"
2. **额度提升**: "会提升您的贷款额度和笔数,未来借款更方便"
3. **还款折扣**: "今天还款可以获得还款折扣优惠"

**压力等级4-6 (信用与限制警告) - 适用于Stage2**:
4. **信用分影响**: "逾期会影响您的信用分,降低贷款额度"
5. **合作终止**: "后续贷款会很困难,我们可能会停止与您的合作"
6. **黑名单警告**: "可能被拉入黑名单,不只是在我们这里,在其他平台也无法借款"

**压力等级7-8 (第三方介入警告) - 适用于Stage3**:
7. **紧急联系人**: "我们可能需要与您的紧急联系人沟通,让他们协助您还款"
8. **工作单位联系**: "可能需要联系您工作的单位,与您的领导沟通了解您的经济状况"

**压力等级9-10 (强制措施) - 适用于Stage4**:
9. **社交媒体联系**: "我们会在社交媒体上与您沟通,需要让您知道我们可以通过多种渠道找到您"
10. **第三方上门**: "将安排第三方上门进行催收"

**使用规则**:
- Stage0-1: 只使用等级1-3(正向激励)
- Stage2: 可使用等级1-6(正向+信用警告)
- Stage3: 可使用等级4-8(信用+第三方介入警告)
- Stage4: 可使用等级7-10(第三方介入+强制措施),但9-10需要sop_trigger_named_escalation=true且approval_id存在
- 必须遵守合规:所有压力必须是真实可执行的,不得虚构或夸大

【策略输出要求】

在strategy_card中体现完整流程和具体压力措施:

1. **today_kpi**: 列出完整的步骤序列和具体压力措施,如:
     - Stage2示例: ["step1_try_full_payment", "step2_explore_reasons", "step3_warning_credit_impact", "step4_negotiate_partial_today"]
     - Stage3示例: ["step1_acknowledge_obstacle", "step2_force_binary_decision", "step3_escalate_warning_notice", "step4_mention_emergency_contact"]
     - 注意：每一步都要包含具体的压力措施名称，不只是流程步骤

2. **allowed_actions**: 包含所有可能用到的具体动作,并按照压力强度排序:
     - Stage2示例: ["ask_full_payment", "ask_reasons", "explore_fund_sources", "mention_credit_impact", "offer_extension_if_eligible", "negotiate_partial_today", "confirm_time"]
     - Stage3示例: ["binary_can_pay_today", "mention_credit_blacklist", "mention_emergency_contact_involvement", "process_notice", "explore_alternative_payment"]
     - Stage4示例: ["final_notice", "mention_third_party_collection", "mention_social_media_contact", "mention_workplace_contact"]

3. **pressure_level**: 根据Stage设置:
     - Stage0-1: "polite"
     - Stage2: "polite_firm"
     - Stage3: "firm"
     - Stage4: "firm"

4. **params**: 设置流程控制和压力参数:
     {{
         "conversation_flow": "multi_step",
         "pressure_tactics": ["specific_tactic_1", "specific_tactic_2"],  # 具体的压力手段
         "allow_extension": true/false,
         "allow_partial": true/false,
         "current_stage_pressure_level": 1-10  # 10级压力策略的具体等级
     }}

5. **notes**: 说明当前对话进展、下一步重点和具体压力措施:
     - 例："用户多次拒绝，已升级到Stage3。建议：1)明确告知信用黑名单后果 2)提及紧急联系人 3)给出最后48小时期限"

【关键：pressure_level 与 10级压力策略对应关系】

stage_to_pressure_tactics 映射表：

**Stage0-1 (polite - 压力等级1-3)**:
- 等级1：会员等级提升 → 话术："按时还款会获得良好的信用记录,提升您的会员等级"
- 等级2：额度提升 → 话术："会提升您的贷款额度和笔数"
- 等级3：还款折扣 → 话术："今天还款可以获得还款折扣优惠"

**Stage2 (polite_firm - 压力等级4-6)**:
- 等级4：信用分影响 → 话术："逾期会影响您的信用分,降低贷款额度"
- 等级5：合作终止警告 → 话术："后续贷款会很困难,我们可能会停止与您的合作"
- 等级6：黑名单警告 → 话术："可能被拉入黑名单,不只是在我们这里,在其他平台也无法借款"

**Stage3 (firm - 压力等级7-8)**:
- 等级7：紧急联系人警告 → 话术："我们可能需要与您的紧急联系人沟通,让他们协助您还款"
- 等级8：工作单位联系警告 → 话术："可能需要联系您工作的单位,与您的领导沟通了解您的经济状况"

**Stage4 (firm - 压力等级9-10)**:
- 等级9：社交媒体接触警告 → 话术："我们会在社交媒体上与您沟通,需要让您知道我们可以通过多种渠道找到您"
- 等级10：第三方上门通知 → 话术："将安排第三方上门进行催收"

【具体例子】

**Stage3 的完整strategy_card示例**:
```json
{
    "strategy_id": "Stage3_firm_escalation_with_contact_warning",
    "stage": "Stage3",
    "today_kpi": [
        "step1_acknowledge_obstacle_银行卡掉了_confirm_replacement_timeline",
        "step2_force_binary_decision_今天能还还是完全不能还",
        "step3_escalate_credit_blacklist_warning_explain_consequences",
        "step4_mention_emergency_contact_involvement_as_alternative_solution",
        "step5_set_final_deadline_next_24_hours"
    ],
    "pressure_level": "firm",
    "allowed_actions": [
        "acknowledge_user_obstacle",
        "force_binary_decision",
        "mention_credit_blacklist",
        "mention_emergency_contact",
        "mention_workplace_contact_possibility",
        "set_hard_deadline",
        "process_notice"
    ],
    "guardrails": [
        "today_only_for_dpd_ge_0",
        "no_fake_threats",
        "compliance_notice_only",
        "factual_consequences_only"
    ],
    "escalation_actions_allowed": {
        "mention_emergency_contact": true,
        "mention_workplace_contact": false
    },
    "params": {
        "conversation_flow": "binary_convergence",
        "pressure_tactics": [
            "7_emergency_contact_warning",
            "6_blacklist_warning",
            "acknowledge_obstacle_but_no_excuse"
        ],
        "current_stage_pressure_level": 7,
        "allow_extension": false,
        "allow_partial": false,
        "final_deadline_hours": 24,
        "escalation_ready": true
    },
    "notes": "用户银行卡掉了（障碍已确认），但多次提及同一理由。已升级到Stage3。建议话术：1)承认卡掉的问题真实存在 2)但告知必须今天找到替代方案（亲友借、其他卡等） 3)强调信用黑名单后果 4)提及可能需要联系紧急联系人 5)给出24小时最后期限"
}
```

只输出strategy_card JSON,不要任何解释文本。格式必须包含上述所有字段,特别是**具体的pressure_tactics和step名称要包含实际措施**。
"""


def build_executor_system_prompt(org_name: str, memory_state: Dict[str, Any], strategy_card: Dict[str, Any], history_summary: str) -> str:
    # 提取策略关键信息
    stage = strategy_card.get("stage", "Unknown")
    pressure = strategy_card.get("pressure_level", "polite")
    kpis = strategy_card.get("today_kpi", [])
    allowed_acts = strategy_card.get("allowed_actions", [])
    guardrails = strategy_card.get("guardrails", [])
    notes = strategy_card.get("notes", "")
    params = strategy_card.get("params", {})

    # 提取10级压力策略的具体等级
    pressure_tactics = params.get("pressure_tactics", [])
    pressure_level_num = params.get("current_stage_pressure_level", 1)

    # 建立压力等级与具体措施的映射
    pressure_tactics_mapping = {
        "1_membership_upgrade": "会员等级提升 → '按时还款会获得良好的信用记录，提升您的会员等级'",
        "2_amount_increase": "额度提升 → '会提升您的贷款额度和笔数'",
        "3_payment_discount": "还款折扣 → '今天还款可以获得还款折扣优惠'",
        "4_credit_score_impact": "信用分影响 → '逾期会影响您的信用分，降低贷款额度'",
        "5_cooperation_termination_warning": "合作终止警告 → '后续贷款会很困难，我们可能会停止与您的合作'",
        "6_blacklist_warning": "黑名单警告 → '可能被拉入黑名单，在其他平台也无法借款'",
        "7_emergency_contact_warning": "紧急联系人警告 → '需要与您的紧急联系人沟通'",
        "8_workplace_contact_warning": "工作单位联系警告 → '可能需要联系您的工作单位'",
        "9_social_media_contact": "社交媒体接触警告 → '我们可以通过多种渠道与您沟通'",
        "10_third_party_onsite_collection": "第三方上门通知 → '将安排第三方上门进行催收'",
    }

    # 生成具体的压力措施说明
    pressure_tactics_display = "\n".join([
        f"  - {pressure_tactics_mapping.get(tactic, tactic)}"
        for tactic in pressure_tactics
    ]) if pressure_tactics else "  - 暂无额外压力措施"

    # 将策略转换为可读的指令
    strategy_display = f"""
【当前执行策略】
策略ID: {strategy_card.get('strategy_id', 'Unknown')}
阶段: {stage}
施压等级: {pressure} (10级评分: {pressure_level_num}/10)

【本次应采用的具体压力措施】:
{pressure_tactics_display}

今日目标(按顺序执行):
{chr(10).join([f"  {i+1}. {kpi}" for i, kpi in enumerate(kpis)])}

允许的动作:
{chr(10).join([f"  - {act}" for act in allowed_acts])}

必须遵守的红线:
{chr(10).join([f"  - {rule}" for rule in guardrails])}

策略备注:
{notes or "无"}
"""

    return f"""你是{org_name}负责该账户的还款跟进人员,通过WhatsApp沟通(此处用中文输出)。

{strategy_display}

【用户档案】
- 客户ID: {memory_state.get('customer_id')}
- 逾期天数(DPD): {memory_state.get('dpd')}
- 欠款金额: {memory_state.get('debt_amount')}
- 失约/拒付次数: {memory_state.get('broken_promises')} / {memory_state.get('payment_refusals')}

【历史摘要】
{history_summary or "暂无历史记录"}

【关键指令：动态执行策略】
注意：你将收到 **recent_dialogue (最近对话记录)**。
你必须先**检查对话历史**，判断【今日目标】中的步骤执行到了哪里：

1. **不要重复**：如果历史记录显示 Step 1 已经尝试过且被用户拒绝，请查看策略备注或自动进入 Step 2/Step 3。
2. **承接上下文**：如果用户刚才回答了原因，请基于该原因进行回应（如评估展期或施压），不要机械地跳回 Step 1。
3. **完成状态**：如果今日目标是"尝试全额还款"，而用户明确说"没钱"，则你的任务转变为"挖掘原因"或"协商部分还款"（即进入下一环节）。

【执行要求（第一响应准则）】
1. **优先回应**: 你现在的首要任务是针对用户最后说的那句话给出回应,在策略指引下。
2. **严禁机械化强制**: **绝对禁止**要求用户"请回复能或不能"、"请回复1或2"。应通过自然问句如"那咱们就定在今天下午两点，可以吗？"来达成确认。
3. **策略融合**: 在回应完客户后，顺势将对话引向【今日目标】中的下一步。当需要采取压力措施时，自然地融入话术，不能生硬。
4. **简洁**: 每次只说一件事。
5. **不暴露身份**: 不提"我是AI"或"我是系统"。

【关键例子】
当需要"等级7：紧急联系人警告"时：
✅ 正确："我们希望不打扰您的紧急联系人，但如果今天无法解决这件事，我们就需要与他们沟通。您能理解吗？"

当需要"等级6：黑名单警告"时：
✅ 正确："如果这笔债继续拖欠，我们就需要将您的信息报送到金融黑名单，这会影响您在所有平台的贷款申请。您现在能帮我们避免这种结果吗？"

现在，基于策略和最新消息，给出你的回复。
"""


def build_history_summarizer_system_prompt() -> str:
    return """你是催收历史记录专家。请将粘贴的过往追款记录压缩成结构化摘要。
要求:
1. **输出格式**: 你必须且只能输出严格的 JSON 格式,不允许包含任何解释性文字或开场白。
2. **分析内容**:
   - `summary`: 100-200字的中文摘要(包含核心阻碍、关键词、节点)。
   - `broken_promises`: 统计历史失约总次数。
   - `reason_category`: 从 [unemployment, illness, forgot, malicious_delay, other] 中选一个最佳匹配。
   - `ability_score`: 从 [full, partial, zero] 中选一个最佳评估。
   - `reason_detail`: 一句话总结历史上的主要借口。

示例:
{
  "summary": "客户历史上多次表示收入不稳定...",
  "broken_promises": 2,
  "reason_category": "unemployment",
  "ability_score": "partial",
  "reason_detail": "长期失业且家中有病人"
}
"""


# =========================================================
# LLM helpers
# =========================================================
def get_current_client_info():
    provider = st.session_state.get("selected_provider", "OpenAI")
    model = st.session_state.get("selected_model", "gpt-4o-mini")
    config = MODEL_PROVIDERS.get(provider, MODEL_PROVIDERS["OpenAI"])

    # 优先使用环境变量，其次使用 Streamlit secrets
    api_key = config.get("api_key")
    base_url = config.get("base_url")

    try:
        secrets = st.secrets if hasattr(st, "secrets") else {}
    except Exception:
        secrets = {}

    if not api_key:
        if provider == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY") or secrets.get("OPENAI_API_KEY")
        else:  # Baidu ERNIE
            api_key = os.getenv("BAIDU_API_KEY") or secrets.get("BAIDU_API_KEY")

    if not base_url and provider != "OpenAI":
        base_url = os.getenv("BAIDU_BASE_URL") or (secrets.get("BAIDU_BASE_URL") if secrets else None) or "https://qianfan.baidubce.com/v2"

    return api_key, base_url, model

def call_llm_text(system: str, user: str, temperature: float = 0.2) -> str:
    api_key, base_url, model = get_current_client_info()
    
    # 动态创建客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM 调用失败 ({model}): {str(e)}")
        return f"Error: {str(e)}"


def build_history_summary(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}
    system = build_history_summarizer_system_prompt()
    txt = call_llm_text(system, raw_text, temperature=0.0)
    try:
        clean_txt = clean_json_str(txt)
        return json.loads(clean_txt)
    except Exception as e:
        # 如果解析失败,返回一个包含 raw_error 的结构,供 UI 诊断
        return {
            "summary": txt, 
            "broken_promises": 0, 
            "parse_error": str(e),
            "raw_llm_output": txt 
        }


# =========================================================
# Session Management
# =========================================================
def init_new_session(customer_id: str):
    return {
        "customer_id": customer_id,
        "organization_name": "信贷中心",
        "product_name": "信用贷款",
        "debt_amount": 10000.0,
        "currency": "元",
        "dpd": 1,
        "broken_promises": 0,
        "payment_refusals": 0,
        "extension_eligible": False,
        "approval_id": "APR-001",
        "allowed_contact_hours": "08:00-20:00",
        "stage": "Stage2",
        "no_response_streak": 0,
        "reason_category": "",
        "ability_score": "",
        "reason_detail": "",
        "unresolved_obstacles": [],  # 新增: 待解决具体障碍
        "dialogue": [],
        "strategy_card": None,
        "last_critic": None,
        "history_summary": "",
        "history_events": []
    }


def extract_broken_promises_from_summary(summary: str) -> int:
    """从 history_summary 中提取失约次数"""
    import re
    if not summary:
        return 0
    
    # 匹配 "失约次数: X次" 或 "失约次数: X"
    match = re.search(r'失约次数[：:]\\s*(\\d+)', summary)
    if match:
        return int(match.group(1))
    return 0


def clean_json_str(text: str) -> str:
    """更强力的 JSON 提取逻辑,能从各种垃圾字符中抠出 JSON。"""
    if not text:
        return ""
    import re
    # 尝试匹配 ```json { ... } ``` 格式
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果没有代码块,尝试匹配第一个 { 和最后一个 } 之间的内容
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return text.strip()


def call_critic(strategy_card: Dict[str, Any], memory_state: Dict[str, Any], dialogue: List[Dict[str, str]], history_summary: str) -> CriticResult:
    system = build_critic_system_prompt()
    payload = {
        "strategy_card": strategy_card,
        "memory_state": memory_state,
        "history_summary": history_summary,
        "recent_dialogue": dialogue[-12:],
    }
    user = "请评估并输出JSON:\\n" + json.dumps(payload, ensure_ascii=False)

    # Use simple text generation + manual parse
    try:
        txt = call_llm_text(system, user, temperature=0.0)
        clean_txt = clean_json_str(txt)
        data = json.loads(clean_txt)
        return CriticResult(**data)
    except Exception as e:
        return CriticResult(
            decision="ESCALATE_TO_META",
            decision_reason=f"critic_failed_parse: {str(e)[:150]}",
            reason_codes=["critic_failed"],
            micro_edits_for_executor=MicroEdits(),
        )


def call_meta(memory_state: Dict[str, Any], critic: CriticResult, dialogue: List[Dict[str, str]], history_summary: str) -> Dict[str, Any]:
    system = build_meta_system_prompt()
    payload = {
        "memory_state": memory_state,
        "critic_result": critic.model_dump(),
        "history_summary": history_summary,
        "recent_dialogue": dialogue[-12:],
    }
    user = "请根据最新对话生成 strategy_card JSON:\n" + json.dumps(payload, ensure_ascii=False)

    try:
        txt = call_llm_text(system, user, temperature=0.0)
        
        # 增强型 JSON 提取逻辑
        import re
        json_match = re.search(r'\{.*\}', txt, re.DOTALL)
        if json_match:
            clean_txt = json_match.group(0)
        else:
            clean_txt = clean_json_str(txt)
            
        data = json.loads(clean_txt)
        sc = StrategyCard(**data)

        # 强制对齐 Stage
        forced_stage = memory_state.get("stage")
        if forced_stage and sc.stage != forced_stage:
            sc.stage = forced_stage

        return sc.model_dump()

    except Exception as e:
        import traceback
        print(f"[META ERROR] {traceback.format_exc()}")
        
        # 安全回退
        dpd = int(memory_state.get("dpd", 0))
        return {
            "strategy_id": "fallback_reconfirm",
            "stage": memory_state.get("stage", "Stage2"),
            "today_kpi": ["confirm_tomorrow_commitment", "ask_small_amount_today_as_sincerity"],
            "pressure_level": "polite_firm",
            "allowed_actions": ["confirm_time", "ask_partial_today"],
            "guardrails": ["no_mechanical_logic", "natural_confirmation"],
            "escalation_actions_allowed": {},
            "params": {"tomorrow_promised": True},
            "notes": "用户承诺明天,尝试引导今天处理少量"
        }


def call_executor(strategy_card: Dict[str, Any], memory_state: Dict[str, Any], dialogue: List[Dict[str, str]], micro: MicroEdits, history_summary: str) -> str:
    org_name = memory_state.get("organization_name", "[机构名]")
    # Updated call with new signature
    system = build_executor_system_prompt(org_name, memory_state, strategy_card, history_summary)
    payload = {
        "strategy_card": strategy_card,
        "memory_state": memory_state,
        "history_summary": history_summary,
        "micro_edits": micro.model_dump(),
        "recent_dialogue": dialogue[-12:],
    }
    user = "请基于以下信息生成下一条发给用户的话术:\\n" + json.dumps(payload, ensure_ascii=False)
    return call_llm_text(system, user, temperature=0.2)


# =========================================================
# Memory helpers
# =========================================================
def apply_memory_write(memory: Dict[str, Any], memory_write: Dict[str, Any]) -> Dict[str, Any]:
    if not memory_write:
        return memory
    merged = dict(memory)
    
    # 定义需要“累积”而非“替换”的字段
    cumulative_list_fields = ["unresolved_obstacles", "history_raw_reasons"]
    
    for k, v in memory_write.items():
        # 1. 如果是需要累积的列表字段
        if k in cumulative_list_fields:
            current_list = merged.get(k, [])
            if not isinstance(current_list, list): current_list = [current_list] if current_list else []
            new_items = v if isinstance(v, list) else [v]
            # 去重合并
            for item in new_items:
                if item not in current_list:
                    current_list.append(item)
            merged[k] = current_list
            
        # 2. 如果是理由细节，我们采取“追加”模式而非替换
        elif k == "reason_detail":
            old_val = merged.get(k, "")
            if v and v != old_val:
                merged[k] = f"{old_val} | {v}".strip(" | ") if old_val else v
        
        # 3. 字典类型的深合并
        elif isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
            
        # 4. 其他状态类字段（如能力、分类、Stage）采用最新替换
        else:
            merged[k] = v
            
    return merged


def ensure_strategy_card(memory_state: Dict[str, Any], strategy_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # If it's a non-empty dict, validate it AND check if stage matches
    if strategy_card and isinstance(strategy_card, dict) and len(strategy_card) > 0:
        try:
            validated = StrategyCard(**strategy_card).model_dump()
            # Check if the stage in strategy_card matches the current memory_state stage
            # Use multi-factor calculation
            dpd = int(memory_state.get("dpd", 0))
            bp = int(memory_state.get("broken_promises", 0))
            pr = int(memory_state.get("payment_refusals", 0))
            current_stage = memory_state.get("stage", calculate_stage(dpd, bp, pr))
            if validated.get("stage") == current_stage:
                return validated
            # Stage mismatch detected, fall through to regenerate
        except Exception:
            pass

    dpd = int(memory_state.get("dpd", 0))
    bp = int(memory_state.get("broken_promises", 0))
    pr = int(memory_state.get("payment_refusals", 0))
    stage = memory_state.get("stage", calculate_stage(dpd, bp, pr))

    # Stage-specific strategy design with EXPLICIT pressure tactics
    if stage == "Stage0":
        # 提前期:建立关系,正向激励 (压力等级1-3)
        sc = StrategyCard(
            strategy_id=f"{stage}_relationship_building",
            stage=stage,
            today_kpi=[
                "step1_build_trust_and_introduce_benefits",
                "step2_explain_incentive_programs",
                "step3_confirm_payment_method_and_remind_due_date"
            ],
            pressure_level="polite",
            allowed_actions=["inform_benefits", "offer_discount", "confirm_payment_method", "ask_preferred_contact_time"],
            guardrails=["no_pressure", "positive_tone_only", "focus_on_relationship"],
            escalation_actions_allowed={},
            params={
                "conversation_flow": "relationship_building",
                "pressure_tactics": ["1_membership_upgrade", "2_amount_increase", "3_payment_discount"],
                "current_stage_pressure_level": 1,
                "focus": "relationship",
                "tone": "friendly"
            },
            notes="提前期,客户正常还款。建议话术：强调按时还款的好处(会员等级、额度提升、折扣优惠)。不涉及任何压力措施。"
        )
    elif stage == "Stage1":
        # 到期日:温和提醒 + 摸底 (压力等级3-4)
        sc = StrategyCard(
            strategy_id=f"{stage}_gentle_reminder",
            stage=stage,
            today_kpi=[
                "step1_remind_due_today_with_positivity",
                "step2_ask_payment_plan_and_ability",
                "step3_mention_benefits_of_full_payment_today"
            ],
            pressure_level="polite",
            allowed_actions=["ask_pay_today", "ask_payment_time", "offer_extension_if_eligible", "ask_reasons", "explore_fund_sources"],
            guardrails=["today_only_for_dpd_ge_0", "no_threats", "gentle_tone"],
            escalation_actions_allowed={},
            params={
                "conversation_flow": "information_gathering",
                "pressure_tactics": ["3_payment_discount", "4_credit_impact_mention"],
                "current_stage_pressure_level": 3,
                "probe_ability": True,
                "focus": "information_gathering"
            },
            notes="到期日。建议话术：1)提醒今天是到期日 2)询问是否能今天还款 3)如果有困难,探索原因和资金来源 4)强调按时还款的好处。还未涉及强压力。"
        )
    elif stage == "Stage2":
        # 轻度逾期:施压 + 收敛 (压力等级4-6)
        sc = StrategyCard(
            strategy_id=f"{stage}_light_pressure",
            stage=stage,
            today_kpi=[
                "step1_ask_full_payment_today_firmly",
                "step2_explore_reasons_and_fund_sources",
                "step3_mention_credit_impact_and_blacklist_warning",
                "step4_negotiate_partial_payment_with_deadline"
            ],
            pressure_level="polite_firm",
            allowed_actions=[
                "ask_pay_today", 
                "forced_choice_amount_time", 
                "mention_credit_impact",
                "mention_blacklist_warning",
                "explore_fund_sources",
                "offer_extension_if_eligible",
                "negotiate_partial_today",
                "confirm_time"
            ],
            guardrails=["today_only_for_dpd_ge_0", "no_fake_threats", "no_humiliation", "factual_consequences_only"],
            escalation_actions_allowed={},
            params={
                "conversation_flow": "convergence",
                "pressure_tactics": ["4_credit_score_impact", "5_cooperation_termination_warning", "6_blacklist_warning"],
                "current_stage_pressure_level": 5,
                "allow_partial": True,
                "allow_extension": True if not memory_state.get("broken_promises", 0) >= 2 else False,
                "credit_warning": True
            },
            notes="逾期1-5天,有未履行承诺。建议话术：1)确认能否全额还款,若不能探索原因 2)提及逾期会影响信用分、降低额度 3)警告可能被拉入黑名单 4)如符合条件,可提及展期 5)最后协商今天的部分还款 6)设定明确的还款时间。"
        )
    elif stage == "Stage3":
        # 中度逾期:强施压 + 二元收敛 (压力等级7-8)
        sc = StrategyCard(
            strategy_id=f"{stage}_firm_escalation",
            stage=stage,
            today_kpi=[
                "step1_acknowledge_customer_obstacle_if_exists",
                "step2_force_binary_decision_today_or_never",
                "step3_escalate_blacklist_and_credit_damage_warning",
                "step4_mention_emergency_contact_involvement",
                "step5_set_final_deadline_24_48_hours"
            ],
            pressure_level="firm",
            allowed_actions=[
                "binary_can_pay_today", 
                "mention_credit_blacklist",
                "mention_emergency_contact_warning",
                "mention_workplace_contact_possibility",
                "set_hard_deadline",
                "process_notice",
                "acknowledge_obstacle_but_emphasize_solution"
            ],
            guardrails=[
                "today_only_for_dpd_ge_0", 
                "no_fake_threats", 
                "compliance_notice_only", 
                "factual_consequences_only",
                "no_humiliation"
            ],
            escalation_actions_allowed={"mention_emergency_contact": True},
            params={
                "conversation_flow": "binary_convergence",
                "pressure_tactics": ["6_blacklist_warning", "7_emergency_contact_warning", "8_workplace_contact_warning"],
                "current_stage_pressure_level": 7,
                "allow_partial": False,
                "allow_extension": False,
                "final_deadline_hours": 24,
                "escalation_ready": True
            },
            notes="逾期5-20天或多次失约/拒付。建议话术：1)承认用户的困境(如银行卡掉了)但强调必须找替代方案 2)强制二元选择「今天能还还是完全不能还」 3)详细说明被拉入黑名单的后果(所有平台都无法借款) 4)提及可能需要联系紧急联系人协助 5)给出24小时最后期限 6)如果继续拒绝,告知将启动正式催收流程。"
        )
    else:  # Stage4
        # 严重逾期:最强施压 + 流程告知 (压力等级9-10)
        sc = StrategyCard(
            strategy_id=f"{stage}_maximum_pressure",
            stage=stage,
            today_kpi=[
                "step1_final_notice_about_debt_status",
                "step2_escalate_third_party_collection_warning",
                "step3_mention_social_media_and_workplace_contact",
                "step4_process_formal_escalation_notice"
            ],
            pressure_level="firm",
            allowed_actions=[
                "binary_can_pay_today", 
                "process_notice", 
                "mention_third_party_collection",
                "mention_social_media_contact_warning",
                "mention_workplace_contact_warning",
                "final_notice"
            ],
            guardrails=[
                "compliance_notice_only", 
                "no_humiliation", 
                "factual_consequences_only",
                "record_all_escalation_evidence"
            ],
            escalation_actions_allowed={
                "contact_emergency": True,
                "contact_workplace": True if memory_state.get("sop_trigger_named_escalation") else False,
                "social_media_mention": True if memory_state.get("sop_trigger_named_escalation") else False
            },
            params={
                "conversation_flow": "formal_escalation",
                "pressure_tactics": ["9_social_media_contact", "10_third_party_onsite_collection"],
                "current_stage_pressure_level": 9,
                "allow_partial": False,
                "allow_extension": False,
                "full_compliance_mode": True,
                "escalation_required": True
            },
            notes="严重逾期(>20天)或多次拒付。建议话术：1)明确告知已进入严重违约阶段 2)说明将启动第三方催收机构 3)提及我们有多种方式与其沟通(社交媒体、工作单位等) 4)强调这将严重影响其信用和生活 5)给出最后48小时期限 6)告知将启动法律程序。仅在获得批准(approval_id存在)时才能执行第9-10级措施。"
        )
    
    return sc.model_dump()


# =========================================================
# Orchestrator (single turn)
# =========================================================
def handle_turn(user_msg: str):
    # Retrieve current active state
    state = st.session_state.all_sessions[st.session_state.active_session]
    
    # Initialize turn telemetry
    telemetry = {
        "step1_critic": "Pending",
        "step2_meta": "Skipped",
        "step3_executor": "Pending",
        "strategy_changed": False,
        "old_strategy_id": state["strategy_card"].get("strategy_id") if state["strategy_card"] else None
    }
    
    # Append user message
    state["dialogue"].append({"role": "user", "content": user_msg})

    # Refresh stage
    dpd = int(state.get("dpd", 0))
    bp = int(state.get("broken_promises", 0))
    pr = int(state.get("payment_refusals", 0))
    stage = calculate_stage(dpd, bp, pr)
    state["stage"] = stage

    # Compute SOP trigger
    state["sop_trigger_named_escalation"] = sop_trigger_named_escalation(dpd, bp)

    # Ensure strategy
    state["strategy_card"] = ensure_strategy_card(state, state["strategy_card"])

    # 1) Critic
    critic = call_critic(
        state["strategy_card"],
        state,
        state["dialogue"],
        state["history_summary"]
    )
    state["last_critic"] = critic.model_dump()
    telemetry["step1_critic"] = f"Decision: {critic.decision}"

    # 2) Apply memory writes (Includes dynamic reason detection)
    new_memory = apply_memory_write(state, critic.memory_write)
    
    # 诊断日志
    if critic.memory_write:
        print(f"[DEBUG] Critic memory_write: {critic.memory_write}")
        print(f"[DEBUG] Merged new_memory keys: {new_memory.keys()}")
        print(f"[DEBUG] reason_detail before: {state.get('reason_detail')}")
        print(f"[DEBUG] reason_detail after merge: {new_memory.get('reason_detail')}")
    
    state.update(new_memory)
    
    # --- NEW: Stage 深度联动 (Stage Refresh & Force Meta) ---
    # 根据 Critic 刚记下的新行为,立刻重算 Stage
    dpd_current = int(state.get("dpd", 0))
    bp_current = int(state.get("broken_promises", 0))
    pr_current = int(state.get("payment_refusals", 0))
    new_calculated_stage = calculate_stage(dpd_current, bp_current, pr_current)
    
    if new_calculated_stage != state["stage"]:
        old_stage = state["stage"]
        state["stage"] = new_calculated_stage
        telemetry["step1_critic"] += f" | 🚩 Stage Shift: {old_stage} -> {new_calculated_stage}"
        # 如果 Stage 变了,强制叫醒 Meta,因为旧策略的压力等级可能已经不匹配新 Stage 了
        if critic.decision != "ESCALATE_TO_META":
            critic.decision = "ESCALATE_TO_META"
            critic.decision_reason += f" (System: Stage shifted to {new_calculated_stage}, forcing meta re-alignment)"

    # 3) Meta
    if critic.decision == "ESCALATE_TO_META":
        telemetry["step2_meta"] = "Triggered (Updating Strategy...)"
        new_strategy = call_meta(
            state,
            critic,
            state["dialogue"],
            state["history_summary"]
        )
        if new_strategy.get("strategy_id") != telemetry["old_strategy_id"]:
            telemetry["strategy_changed"] = True
        state["strategy_card"] = new_strategy
        telemetry["step2_meta"] = f"Success (New Strategy: {new_strategy.get('strategy_id')})"
    else:
        telemetry["step2_meta"] = "Skipped (Strategy Fit)"

    # 4) Executor
    telemetry["step3_executor"] = "Generating Speech..."
    reply = call_executor(
        state["strategy_card"],
        state,
        state["dialogue"],
        critic.micro_edits_for_executor,
        state["history_summary"]
    )
    state["dialogue"].append({"role": "assistant", "content": reply})
    telemetry["step3_executor"] = "Reply Sent"
    
    # Store telemetry
    state["last_turn_telemetry"] = telemetry


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(layout="wide")
st.title("三层架构催收大师 - 多会话 & 智能记忆版")

# 1. Initialize Global Session Container
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {
        "C-demo": init_new_session("C-demo")
    }
if "active_session" not in st.session_state:
    st.session_state.active_session = "C-demo"

# Sidebar: Model Engine Select
st.sidebar.header("🤖 模型引擎配置")
provider_list = list(MODEL_PROVIDERS.keys())
selected_provider = st.sidebar.selectbox("供应商", provider_list, index=0)
st.session_state.selected_provider = selected_provider

available_models = MODEL_PROVIDERS[selected_provider]["models"]
selected_model = st.sidebar.selectbox("模型名称", available_models, index=1 if "gpt-4o-mini" in available_models else 0)
st.session_state.selected_model = selected_model

st.sidebar.divider()

# Sidebar: Session Manager
st.sidebar.header("📂 会话管理 (Sessions)")
with st.sidebar.expander("➕ 创建新会话", expanded=False):
    new_cid = st.text_input("客户 ID", value="C-new")
    if st.button("创建会话"):
        if new_cid not in st.session_state.all_sessions:
            st.session_state.all_sessions[new_cid] = init_new_session(new_cid)
            st.session_state.active_session = new_cid
            st.rerun()
        else:
            st.warning("该 ID 已存在")

session_list = list(st.session_state.all_sessions.keys())
selected_session = st.sidebar.selectbox("选择活跃客户", session_list, index=session_list.index(st.session_state.active_session))

if selected_session != st.session_state.active_session:
    st.session_state.active_session = selected_session
    st.rerun()

# Get ACTIVE STATE shortcut
state = st.session_state.all_sessions[st.session_state.active_session]

st.sidebar.divider()
st.sidebar.header("⚙️ 客户基础配置")
state["organization_name"] = st.sidebar.text_input("机构名称", value=state["organization_name"])
state["product_name"] = st.sidebar.text_input("产品名称", value=state["product_name"])
state["debt_amount"] = st.sidebar.number_input("欠款金额", value=float(state["debt_amount"]))
state["currency"] = st.sidebar.text_input("货币单位", value=state["currency"])

left, right = st.columns([2, 1])

# --- LEFT COLUMN: Dialogue ---
with left:
    st.subheader(f"💬 对话: {st.session_state.active_session}")
    for m in state["dialogue"]:
        if m["role"] == "user":
            st.markdown(f"**用户:** {m['content']}")
        else:
            st.markdown(f"**机构:** {m['content']}")

    user_msg = st.text_input("输入回复...", key=f"input_{st.session_state.active_session}")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("发送", type="primary"):
            if not user_msg:
                st.warning("请输入内容")
            else:
                handle_turn(user_msg)
                st.rerun()
    with c2:
        if st.button("清空对话"):
            state["dialogue"] = []
            state["last_critic"] = None
            st.rerun()
    with c3:
        if st.button("🗑️ 删除该会话"):
            if len(st.session_state.all_sessions) > 1:
                del st.session_state.all_sessions[st.session_state.active_session]
                st.session_state.active_session = list(st.session_state.all_sessions.keys())[0]
                st.rerun()

# --- RIGHT COLUMN: Analysis & Controls ---
with right:
    # --- 👤 客户画像与记忆 ---
    st.subheader("👤 客户画像与记忆 (Memory)")
    with st.container(border=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            cat = state.get('reason_category', '') or '未知'
            cat_map = {
                "unemployment": "🚫 失业/收入",
                "illness": "🏥 疾病/健康",
                "forgot": "❓ 忘记/疏忽",
                "malicious_delay": "👿 恶意拖延",
                "other": "⚙️ 其他",
                "": "⏳ 未知"
            }
            display_cat = cat_map.get(cat, cat)
            st.metric("原因分类", display_cat)
        
        with col_m2:
            score = state.get('ability_score', '') or '未知'
            score_map = {
                "full": "✅ 有能力全额",
                "partial": "⚠️ 仅能部分",
                "zero": "❌ 无力还款",
                "": "⏳ 未知"
            }
            display_score = score_map.get(score, score)
            st.metric("能力评估", display_score)

        # 显示具体理由
        reason_detail = state.get('reason_detail', '').strip()
        if reason_detail:
            st.info(f"**具体理由:** {reason_detail}")
        else:
            st.caption("⏳ 尚未确定具体理由")
        
        # 显示待解决障碍
        obstacles = state.get('unresolved_obstacles', [])
        if obstacles and len(obstacles) > 0:
            obstacles_str = " | ".join(obstacles)
            st.warning(f"**待解决障碍:** {obstacles_str}")
        
        # 诊断面板（DEBUG用）
        with st.expander("🔍 诊断：Memory 原始数据", expanded=False):
            st.code(json.dumps({
                "reason_category": state.get('reason_category', ''),
                "reason_detail": state.get('reason_detail', ''),
                "ability_score": state.get('ability_score', ''),
                "unresolved_obstacles": state.get('unresolved_obstacles', [])
            }, ensure_ascii=False, indent=2), language="json")

    # --- 📥 导入历史记录 ---
    with st.expander("📥 导入过往记录 (智能解析)", expanded=False):
        hist_text = st.text_area("粘贴聊天记录原文...", height=150)
        if st.button("开始智能解析并填入画像"):
            if hist_text.strip():
                with st.spinner("AI 正在深度解析历史记录..."):
                    result = build_history_summary(hist_text)
                    
                    if "parse_error" in result:
                        st.error(f"❌ 记忆解析失败: {result['parse_error']}")
                        with st.expander("🔍 查看模型返回的原话 (用于排查)"):
                            st.code(result["raw_llm_output"])
                    
                    state["history_summary"] = result.get("summary", "")
                    state["broken_promises"] = result.get("broken_promises", 0)
                    state["reason_category"] = result.get("reason_category", "")
                    state["ability_score"] = result.get("ability_score", "")
                    state["reason_detail"] = result.get("reason_detail", "")
                    
                    state["history_events"].append({
                        "imported_at": dt.datetime.now().isoformat(),
                        "text": hist_text
                    })
                    st.success("✅ 历史记录已转化为系统记忆！")
                    st.rerun()

    # --- ⚙️ 业务参数可调 ---
    st.divider()
    st.subheader("📊 业务实时参数")
    dpd = st.number_input("DPD (逾期天数)", value=int(state["dpd"]), step=1)
    bp = st.number_input("历史失约次数", value=int(state["broken_promises"]), step=1)
    pr = st.number_input("本次拒付次数", value=int(state["payment_refusals"]), step=1)
    ext = st.checkbox("可展期 (extension_eligible)", value=bool(state["extension_eligible"]))
    
    state["dpd"] = dpd
    state["broken_promises"] = bp
    state["payment_refusals"] = pr
    state["extension_eligible"] = ext
    
    # Auto-refresh stage
    state["stage"] = calculate_stage(dpd, bp, pr)
    risk_score = dpd * 10 + bp * 15 + pr * 20
    
    # 显示风险等级及变化
    st.divider()
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    with col_risk1:
        st.metric("🚩 DPD风险", dpd, f"得分: {dpd * 10}")
    with col_risk2:
        st.metric("🔗 失约风险", bp, f"得分: {bp * 15}")
    with col_risk3:
        st.metric("✋ 拒付风险", pr, f"得分: {pr * 20}")
    
    # 显示总体阶段和风险分
    stage_color_map = {
        "Stage0": "🟢",
        "Stage1": "🟡", 
        "Stage2": "🟠",
        "Stage3": "🔴",
        "Stage4": "⚫"
    }
    stage_emoji = stage_color_map.get(state['stage'], "❓")
    st.markdown(f"{stage_emoji} **当前阶段: {state['stage']}** | 总风险分: **{risk_score}**")

    # --- 🧠 策略展示 ---
    st.divider()
    st.subheader("🧠 策略核心")
    
    # Refresh strategy if stage changed
    state["strategy_card"] = ensure_strategy_card(state, state["strategy_card"])
    sc = state["strategy_card"]
    
    if sc:
        st.markdown(f"**ID:** `{sc.get('strategy_id')}` | **施压:** `{sc.get('pressure_level')}`")
        with st.expander("🎯 今日目标 (KPI)", expanded=True):
            for k in sc.get('today_kpi', []):
                st.markdown(f"- {k}")
        
        with st.expander("📜 Executor 完整策略视角", expanded=False):
            executor_strategy_view = f"""
策略ID: {sc.get('strategy_id')} | 阶段: {sc.get('stage')} | 等级: {sc.get('pressure_level')}
今日目标:
{chr(10).join([f"  - {k}" for k in sc.get('today_kpi', [])])}
允许动作: {", ".join(sc.get('allowed_actions', []))}
红线规则: {", ".join(sc.get('guardrails', []))}
备注: {sc.get('notes', '无')}
"""
            st.code(executor_strategy_view, language="text")

    # --- ⛓️ System Turn Pipeline (可视化流水线) ---
    st.divider()
    st.subheader("⛓️ System Turn Pipeline")
    tele = state.get("last_turn_telemetry")
    if tele:
        # Step 1: Critic
        with st.container(border=True):
            st.markdown(f"**1. Critic Observation**")
            st.code(tele["step1_critic"])
            
            # Step 2: Meta (Show flow)
            st.markdown(f"**2. Meta strategy Engine**")
            meta_status = tele["step2_meta"]
            if "Success" in meta_status:
                st.success(meta_status)
                if tele.get("strategy_changed"):
                    st.toast("🎯 策略已由 Meta 重写！", icon="🔥")
                    st.info(f"🔄 策略变更: {tele['old_strategy_id']} ➔ {state['strategy_card'].get('strategy_id')}")
            elif "Skipped" in meta_status:
                st.caption(meta_status)
            else:
                st.warning(meta_status)

            # Step 3: Executor
            st.markdown(f"**3. Executor Action**")
            st.code(tele["step3_executor"])
    else:
        st.caption("等待第一轮对话流水线数据...")

    # --- 🧐 Critic 质检区 ---
    st.divider()
    st.subheader("🧐 Critic 质检记录")
    critic = state.get("last_critic")
    if critic:
        decision = critic.get("decision")
        color = "green" if decision == "CONTINUE" else "orange" if decision == "ADAPT_WITHIN_STRATEGY" else "red"
        st.markdown(f"决策: :{color}[**{decision}**]")
        st.info(f"理由: {critic.get('decision_reason')}")
        if critic.get("risk_flags"):
            st.warning(f"🚩 风险信号: {', '.join(critic.get('risk_flags', []))}")
        
        # 新增：展示 Critic 提取的 memory_write
        memory_write = critic.get("memory_write", {})
        if memory_write:
            st.divider()
            st.markdown("**📝 Critic 提取的记忆更新:**")
            with st.expander("查看 memory_write 内容", expanded=True):
                st.code(json.dumps(memory_write, ensure_ascii=False, indent=2), language="json")
                
                # 诊断：检查拒付次数是否被正确增加
                if "payment_refusals" in memory_write:
                    new_pr_value = memory_write.get("payment_refusals")
                    st.caption(f"⚠️ Critic 建议将拒付次数 (payment_refusals) 更新为: **{new_pr_value}**")
                    st.caption(f"💡 提示：你可以在上面的'本次拒付次数'中手动调整，或者让系统自动应用 Critic 的建议")
                
                # 检查是否有reason_detail被成功提取
                if "reason_detail" in memory_write and memory_write["reason_detail"]:
                    state_reason = state.get("reason_detail", "")
                    if memory_write["reason_detail"] in state_reason or state_reason in memory_write["reason_detail"] or not state_reason:
                        st.success(f"✅ reason_detail 已应用到 state: {memory_write['reason_detail']}")
                    else:
                        st.warning(f"⚠️ 数据可能不匹配。Critic 提取: {memory_write['reason_detail']} | State: {state_reason}")
    else:
        st.caption("等待下一轮质检结果...")
