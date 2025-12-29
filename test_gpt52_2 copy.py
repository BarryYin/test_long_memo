import os
import json
import datetime as dt
from typing import Literal, List, Dict, Any, Optional

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI

# =========================================================
# Config
# =========================================================
# If your actual model name differs, change this.
MODEL = "gpt-5-mini"

# OpenAI client (expects OPENAI_API_KEY env var)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================================================
# Business Rules (your definitions)
# =========================================================
def dpd_to_stage(dpd: int) -> str:
    """Legacy function for backward compatibility - delegates to calculate_stage"""
    return calculate_stage(dpd, 0, 0)


def calculate_stage(dpd: int, broken_promises: int = 0, payment_refusals: int = 0) -> str:
    """
    综合计算 Stage，考虑三个因素：
    1. DPD (Days Past Due) - 逾期天数
    2. broken_promises - 失约次数（承诺还款但未履行）
    3. payment_refusals - 拒付次数（明确拒绝当天还款）
    
    评分规则：
    - DPD < 0: 直接返回 Stage0（提前期，正向激励）
    - DPD * 10 + broken_promises * 15 + payment_refusals * 20 = 总分
    - 总分映射到 Stage1-4
    """
    # 特殊处理：DPD < 0（未到期，提前期）
    if dpd < 0:
        return "Stage0"
    
    # 计算风险评分
    dpd_score = dpd * 10
    promise_score = broken_promises * 15
    refusal_score = payment_refusals * 20
    total_score = dpd_score + promise_score + refusal_score
    
    # Stage 映射
    if total_score == 0:
        return "Stage1"  # 到期日，无不良记录
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

    # If Stage4 + sop_trigger + approval_id exists, Meta can turn these on (true/false)
    escalation_actions_allowed: Dict[str, bool] = Field(default_factory=dict)

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
# Prompt Templates
# =========================================================
COMPLIANCE_GUARDRAILS = """
合规硬约束（不可违反）：
- 不得虚构不会执行的后果；只能告知真实流程（可被审计）。
- 不得羞辱、恐吓、夸大法律后果、公开曝光式表达。
- 涉及第三方/外访等动作，只能在sop_trigger=true且approval_id存在时，以流程告知口吻提及，且不得向第三方披露债务细节。
- DPD>=0时遵守“当天闭环”：必须促成（今天全额）或（今天部分+今天时间点）或（若资格允许）今天完成展期。严禁接受“明天还”等任何跨天承诺。
- 遇到反复推脱/拒付，必须执行“二元收敛”：先确认“今天能否支付（能/不能）”。若不能，则触发合规流程告知；若能，则强制二选一（金额/时间）。
- 每次只问1个主问题（最多附带1句极短确认）。
"""


def build_critic_system_prompt() -> str:
    return f"""你是催收对话的质检与策略适配评估器（Critic + Strategy-Fit Gate）。
你必须对齐【当前策略卡】与【对话进程】做门控决策：
- CONTINUE：继续当前策略
- ADAPT_WITHIN_STRATEGY：策略对但话术/问法需微调（不触发元策略）
- ESCALATE_TO_META：策略不适配/无进展/阶段需要切换（触发元策略）
- HANDOFF：高风险合规/投诉/停止联系等

{COMPLIANCE_GUARDRAILS}

输出必须是严格JSON，且只输出JSON。格式如下：
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
  "memory_write": {{"key": "value"}},
  "risk_flags": ["flag1"]
}}
"""


def build_meta_system_prompt() -> str:
    return f"""你是元策略生成器（Meta / Controller）。
输入：memory_state, critic_result, recent_dialogue, history_summary。
输出：更新后的strategy_card（严格JSON）。

规则：
- 必须遵守合规硬约束。
- Stage必须与DPD映射一致（memory_state.stage），不要擅自改Stage。
- DPD>=0必须遵守“当天闭环”：不允许给未来承诺空间。今天必须落地：全额、或部分+今天时间点、或（若extension_eligible=true）今天完成展期。拒绝任何“明天/下周”的提议。
- 展期仅在extension_eligible=true时允许作为策略分支，且目标必须是“今天完成展期”。
- Stage4且sop_trigger_named_escalation=true且approval_id存在时，允许以流程告知口吻更明确提及升级处置（不得羞辱/夸大/公开曝光）。
- 必须参考history_summary（历史追款摘要），避免重复被客户用同一借口拖延，优先采用已提供过的替代方案，并在必要时提高收敛强度。
- 若用户持续无效沟通，指示Executor进入“二元收敛”模式（今天能/不能）。


只输出strategy_card JSON，不要任何解释文本。格式如下：
{{
  "strategy_id": "str",
  "stage": "Stage0" | "Stage1" | "Stage2" | "Stage3" | "Stage4",
  "today_kpi": ["kpi1", "kpi2"],
  "pressure_level": "polite" | "polite_firm" | "firm",
  "allowed_actions": ["action1", "action2"],
  "guardrails": ["rule1"],
  "escalation_actions_allowed": {{"action_name": boolean}},
  "params": {{}},
  "notes": "optional notes"
}}
"""




def build_executor_system_prompt(org_name: str, memory_state: Dict[str, Any], strategy_card: Dict[str, Any], history_summary: str) -> str:
    # 提取策略关键信息
    stage = strategy_card.get("stage", "Unknown")
    pressure = strategy_card.get("pressure_level", "polite")
    kpis = strategy_card.get("today_kpi", [])
    allowed_acts = strategy_card.get("allowed_actions", [])
    guardrails = strategy_card.get("guardrails", [])
    
    # 动态构建策略指令
    strategy_instruction = f"""
    【当前策略状态】
    - 阶段: {stage}
    - 施压等级: {pressure} (决定你的语气强硬度)
    - 今日KPI (你的核心目标): {', '.join(kpis)}
    - 允许的动作: {', '.join(allowed_acts)}
    - 必须遵守的红线: {', '.join(guardrails)}
    """

    # 记忆与上下文指令
    context_instruction = f"""
    【用户记忆档案】
    - 逾期天数: {memory_state.get('dpd')} (正数表示已逾期)
    - 失约次数: {memory_state.get('broken_promises')} (次数越多，你越不应轻信新的非即时承诺)
    - 历史摘要: {history_summary or "暂无历史"} (这是用户过去的表现，如果用户重复之前的借口，必须当场揭穿)
    """

    return f"""你是{org_name}负责该账户的还款跟进人员，通过WhatsApp沟通（此处用中文输出）。

{strategy_instruction}

{context_instruction}

【核心交互逻辑 - 必须结合当前策略】

1. **历史连续性 (Critical)**:
   - 在回复前，先看 `recent_dialogue`。
   - 如果上一轮你问了问题，用户回答了，必须针对性反馈，不要机械跳转。
   - 如果上一轮用户处于回避状态，参考【今日KPI】决定是否强制收敛。

2. **未来承诺拦截 (Promise Injection)**:
   - 除非【允许动作】里明确包含 "accept_future_promise" (通常低DPD才允许)，否则：
   - 当用户说“明天还”、“工资还没发”时 -> 必须回应：“理解困难，但按规定今天必须处理（DPD>=0）”。
   - 紧接：给出【今日全额】或【今日部分】的选项（根据KPI决定）。

3. **借口拆解 (Excuse Busting)**:
   - 如果用户理由在 `history_summary` 出现过 -> 直接指出：“记录显示您之前也提过类似情况，我们已提供了替代方案。”
   - 提供替代渠道（网银/亲友/柜台）并强制二选一。

4. **二元收敛 (Binary Converge)**:
   - 当用户推脱时，话术模板：“为了帮您[避免后果]，请二选一：A. 现在处理全额 {memory_state.get('debt_amount')}；B. 今天先付 [建议金额] 并在 [具体时间] 前补齐。请回复 A 或 B。”

5. **二元收敛与合规告知**（Feature: 用户明确拒绝、说“今天没钱”、“不能付”）：
   - 第一步：确认状态。“所以您确认今天无法处理任何金额，是吗？（请回答 是/否）”
   - 第二步（若用户确认不能）：执行【合规流程告知】。
     - 话术模板：“好的，由于您明确表示今天无法处理，系统将自动进入后续处置流程（包含[列举1-2个真实后果，如信用记录更新/持续跟进]）。为确保信息触达，请告知今天后续何时方便在该号码联系您？”
   - 注意：告知流程后，不再乞求还款，只确认联系时间窗口。

每天的聊天开始，请延续之前的对话，以便让客户知道我们一直在跟进他的借款订单。

【要求】
- 不提“我是AI”。
- 语气需符合 `{pressure}` 等级。
- 严禁违背 `guardrails`。
- 每次只输出一条精简回复，不要长篇大论。
"""


# =========================================================
# LLM helpers
# =========================================================
# (omitted previous helpers for brevity if unchanged, but need to be careful with replace)

# Note: I need to replace the call site too in the same file but it's far away.
# Actually replace_file_content replaces a single contiguous block.
# The definition is at ~185, and the call is at ~313. They are not contiguous.
# I should use multi_replace_file_content or just do the definition first.

# Let's fix the definition first.
# Wait, simply changing signature here won't work because I cannot change the call site in the same replace_file_content if they are far apart.
# I will use multi_replace_file_content.



def build_history_summarizer_system_prompt() -> str:
    return """你是催收历史记录摘要器。请把用户粘贴的“过往追款记录（纯文本）”压缩成给催收对话使用的简明摘要。
要求：
- 输出中文，100~220字（尽量短但信息密度高）
- 重点：客户常见借口/障碍、是否拒绝替代方案、关键日期节点、到期/逾期结果、是否失约
- 不要输出列表编号，不要加标题
- 只输出摘要文本
"""


# =========================================================
# LLM helpers
# =========================================================
def call_llm_text(system: str, user: str, temperature: float = 0.2) -> str:
    # temperature arg is kept in signature for compatibility but ignored in call
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # temperature=temperature,  # Unsupported by this model/endpoint
    )
    return resp.output_text.strip()


def build_history_summary(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""
    system = build_history_summarizer_system_prompt()
    return call_llm_text(system, raw_text, temperature=0.0)


def clean_json_str(text: str) -> str:
    """Helper to remove markdown code blocks from JSON string."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (e.g. ```json)
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def call_critic(strategy_card: Dict[str, Any], memory_state: Dict[str, Any], dialogue: List[Dict[str, str]], history_summary: str) -> CriticResult:
    system = build_critic_system_prompt()
    payload = {
        "strategy_card": strategy_card,
        "memory_state": memory_state,
        "history_summary": history_summary,
        "recent_dialogue": dialogue[-12:],
    }
    user = "请评估并输出JSON：\n" + json.dumps(payload, ensure_ascii=False)

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
    user = "请生成新的strategy_card JSON：\n" + json.dumps(payload, ensure_ascii=False)

    try:
        txt = call_llm_text(system, user, temperature=0.0)
        clean_txt = clean_json_str(txt)
        data = json.loads(clean_txt)
        sc = StrategyCard(**data)

        # Hard alignment: stage must match memory_state.stage
        forced_stage = memory_state.get("stage")
        if forced_stage and sc.stage != forced_stage:
            sc.stage = forced_stage
            sc.notes = (sc.notes or "") + " | stage_forced_to_memory_state"

        return sc.model_dump()

    except Exception as e:
        # Log the error for debugging
        import traceback
        error_detail = traceback.format_exc()
        print(f"[META ERROR] {error_detail}")  # This will show in terminal
        
        dpd = int(memory_state.get("dpd", 0))
        stage = memory_state.get("stage", dpd_to_stage(dpd))
        fallback = StrategyCard(
            strategy_id="fallback_strategy",
            stage=stage,
            today_kpi=["payment_today_or_extension_today"] if dpd >= 0 else ["confirm_plan"],
            pressure_level="polite_firm" if dpd >= 1 else "polite",
            allowed_actions=["ask_pay_today", "offer_extension_if_eligible", "process_notice"],
            guardrails=["today_only_for_dpd_ge_0", "no_fake_threats", "no_humiliation"],
            escalation_actions_allowed={},
            params={"meta_error": f"{str(e)[:150]}"},
            notes="meta_fallback"
        )
        return fallback.model_dump()


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
    user = "请基于以下信息生成下一条发给用户的话术：\n" + json.dumps(payload, ensure_ascii=False)
    return call_llm_text(system, user, temperature=0.2)


# =========================================================
# Memory helpers
# =========================================================
def apply_memory_write(memory: Dict[str, Any], memory_write: Dict[str, Any]) -> Dict[str, Any]:
    if not memory_write:
        return memory
    merged = dict(memory)
    for k, v in memory_write.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def ensure_strategy_card(memory_state: Dict[str, Any], strategy_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # If it's a non-empty dict, validate it AND check if stage matches
    if strategy_card and isinstance(strategy_card, dict) and len(strategy_card) > 0:
        try:
            validated = StrategyCard(**strategy_card).model_dump()
            # Check if the stage in strategy_card matches the current memory_state stage
            current_stage = memory_state.get("stage", dpd_to_stage(int(memory_state.get("dpd", 0))))
            if validated.get("stage") == current_stage:
                return validated
            # Stage mismatch detected, fall through to regenerate
        except Exception:
            pass

    dpd = int(memory_state.get("dpd", 0))
    stage = memory_state.get("stage", dpd_to_stage(dpd))

    sc = StrategyCard(
        strategy_id=f"{stage}_default",
        stage=stage,
        today_kpi=["payment_today_or_extension_today"] if dpd >= 0 else ["confirm_plan"],
        pressure_level="polite_firm" if dpd >= 1 else "polite",
        allowed_actions=["ask_pay_today", "offer_extension_if_eligible", "process_notice"],
        guardrails=["today_only_for_dpd_ge_0", "no_fake_threats", "no_humiliation"],
        escalation_actions_allowed={},
        params={}
    )
    return sc.model_dump()


# =========================================================
# Orchestrator (single turn)
# =========================================================
def handle_turn(user_msg: str):
    # Append user message
    st.session_state.dialogue.append({"role": "user", "content": user_msg})

    # Refresh stage by DPD every turn (now using multi-factor calculation)
    dpd = int(st.session_state.memory_state.get("dpd", 0))
    bp = int(st.session_state.memory_state.get("broken_promises", 0))
    pr = int(st.session_state.memory_state.get("payment_refusals", 0))
    stage = calculate_stage(dpd, bp, pr)
    st.session_state.memory_state["stage"] = stage

    # Compute SOP trigger and store
    st.session_state.memory_state["sop_trigger_named_escalation"] = sop_trigger_named_escalation(dpd, bp)

    # Ensure strategy exists and is synced
    st.session_state.strategy_card = ensure_strategy_card(st.session_state.memory_state, st.session_state.strategy_card)

    # NEW: If it's a default strategy, try to get a better one from Meta immediately
    is_default = st.session_state.strategy_card.get("strategy_id", "").endswith("_default")
    
    # 1) Critic (Gate)
    critic = call_critic(
        st.session_state.strategy_card,
        st.session_state.memory_state,
        st.session_state.dialogue,
        st.session_state.history_summary
    )
    
    # If default strategy, force escalation to Meta even if Critic didn't ask (on the first turn)
    if is_default and critic.decision == "CONTINUE":
        critic.decision = "ESCALATE_TO_META"
        critic.decision_reason += " | Initial default strategy detected, forcing Meta-layer activation."
    
    st.session_state.last_critic = critic.model_dump()

    # 2) Apply critic memory writes
    st.session_state.memory_state = apply_memory_write(st.session_state.memory_state, critic.memory_write)

    # 3) Meta rewrite strategy if needed
    if critic.decision == "ESCALATE_TO_META":
        print(f"[DEBUG] Calling Meta layer... Current strategy_id: {st.session_state.strategy_card.get('strategy_id')}")
        new_strategy = call_meta(
            st.session_state.memory_state,
            critic,
            st.session_state.dialogue,
            st.session_state.history_summary
        )
        print(f"[DEBUG] Meta returned strategy_id: {new_strategy.get('strategy_id')}")
        st.session_state.strategy_card = new_strategy

    # 4) Executor response
    reply = call_executor(
        st.session_state.strategy_card,
        st.session_state.memory_state,
        st.session_state.dialogue,
        critic.micro_edits_for_executor,
        st.session_state.history_summary
    )
    st.session_state.dialogue.append({"role": "assistant", "content": reply})


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(layout="wide")
st.title("三层Prompt（Meta/Executor/Critic）+ 历史追款摘要(history_summary)（单文件，可直接跑）")

# Sidebar Configuration
st.sidebar.header("环境/模拟参数配置")
org_name = st.sidebar.text_input("机构名称", value=st.session_state.get("memory_state", {}).get("organization_name", "信贷中心"))
prod_name = st.sidebar.text_input("产品名称", value=st.session_state.get("memory_state", {}).get("product_name", "信用贷款"))
debt_amt = st.sidebar.number_input("欠款金额", value=float(st.session_state.get("memory_state", {}).get("debt_amount", 10000.0)))
curr = st.sidebar.text_input("货币单位", value=st.session_state.get("memory_state", {}).get("currency", "元"))

if "memory_state" in st.session_state:
    st.session_state.memory_state.update({
        "organization_name": org_name,
        "product_name": prod_name,
        "debt_amount": debt_amt,
        "currency": curr
    })

if "dialogue" not in st.session_state:
    st.session_state.dialogue = []
if "memory_state" not in st.session_state:
    st.session_state.memory_state = {
        "customer_id": "C-demo",
        "organization_name": "信贷中心",
        "product_name": "信用贷款",
        "debt_amount": 10000.0,
        "currency": "元",
        "dpd": 1,
        "broken_promises": 0,
        "payment_refusals": 0,  # NEW: 拒付次数
        "extension_eligible": False,  # toggle in UI
        "approval_id": "APR-001",      # needed for Stage4 named escalation mention
        "allowed_contact_hours": "08:00-20:00 WIB",
        "stage": "Stage2",
        "no_response_streak": 0,
    }
if "strategy_card" not in st.session_state or st.session_state.strategy_card is None:
    # Initialize with default immediately, using current memory_state
    st.session_state.strategy_card = ensure_strategy_card(
        st.session_state.memory_state,  # Use actual memory state instead of hardcoded values
        None
    )
if "last_critic" not in st.session_state:
    st.session_state.last_critic = None

# NEW: history storage
if "history_events" not in st.session_state:
    st.session_state.history_events = []  # raw imported text blocks + metadata
if "history_summary" not in st.session_state:
    st.session_state.history_summary = ""  # short summary passed to LLM every turn

left, right = st.columns([2, 1])

with left:
    st.subheader("对话")
    for m in st.session_state.dialogue:
        if m["role"] == "user":
            st.markdown(f"**用户：** {m['content']}")
        else:
            st.markdown(f"**机构：** {m['content']}")

    user_msg = st.text_input("用户输入（模拟WhatsApp）", key="user_input")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("发送"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("缺少 OPENAI_API_KEY 环境变量")
            else:
                handle_turn(user_msg)
                st.rerun()

    with c2:
        if st.button("模拟：用户不回应（仅记录）"):
            st.session_state.dialogue.append({"role": "assistant", "content": "（系统记录：本次触达用户未回应）"})
            st.session_state.memory_state["no_response_streak"] = int(st.session_state.memory_state.get("no_response_streak", 0)) + 1
            st.rerun()

    with c3:
        if st.button("清空对话"):
            st.session_state.dialogue = []
            st.session_state.last_critic = None
            st.rerun()

with right:
    st.subheader("导入过往追款记录（纯文本 → summary）")
    hist_text = st.text_area("粘贴历史记录原文（可多段）", height=200, placeholder="把你们的过往追款记录粘贴到这里…")

    r1, r2 = st.columns([1, 1])
    with r1:
        if st.button("导入并生成summary"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("缺少 OPENAI_API_KEY 环境变量")
            else:
                txt = (hist_text or "").strip()
                if not txt:
                    st.warning("请先粘贴历史文本")
                else:
                    st.session_state.history_events.append({
                        "source": "manual_paste",
                        "text": txt,
                        "imported_at": dt.datetime.now().isoformat()
                    })
                    # regenerate (replace) summary from latest text
                    st.session_state.history_summary = build_history_summary(txt)
                    st.rerun()

    with r2:
        if st.button("清空历史summary"):
            st.session_state.history_summary = ""
            st.rerun()

    st.caption("history_summary 会在每轮都传给 Critic / Meta / Executor，用于跨天记忆与避免重复被同一借口拖延。")
    st.subheader("history_summary（给模型看的）")
    st.write(st.session_state.history_summary or "（空）")

    with st.expander("history_events（原始导入记录，供回放/审计）", expanded=False):
        st.json(st.session_state.history_events)

    st.divider()
    st.subheader("业务参数/记忆（可调）")
    dpd = st.number_input("DPD（可为负）", value=int(st.session_state.memory_state.get("dpd", 0)), step=1)
    bp = st.number_input("broken_promises（失约次数）", value=int(st.session_state.memory_state.get("broken_promises", 0)), step=1)
    pr = st.number_input("payment_refusals（拒付次数）", value=int(st.session_state.memory_state.get("payment_refusals", 0)), step=1)
    ext = st.checkbox("extension_eligible（可展期）", value=bool(st.session_state.memory_state.get("extension_eligible", False)))
    approval_id = st.text_input("approval_id（Stage4点名升级需存在）", value=str(st.session_state.memory_state.get("approval_id", "")))

    st.session_state.memory_state["dpd"] = int(dpd)
    st.session_state.memory_state["broken_promises"] = int(bp)
    st.session_state.memory_state["payment_refusals"] = int(pr)
    st.session_state.memory_state["extension_eligible"] = bool(ext)
    st.session_state.memory_state["approval_id"] = approval_id

    # Auto-refresh stage + SOP trigger displayed (using multi-factor calculation)
    st.session_state.memory_state["stage"] = calculate_stage(int(dpd), int(bp), int(pr))
    st.session_state.memory_state["sop_trigger_named_escalation"] = sop_trigger_named_escalation(int(dpd), int(bp))
    
    # Display risk score for transparency
    risk_score = int(dpd) * 10 + int(bp) * 15 + int(pr) * 20
    st.caption(f"Stage 会根据 DPD、失约次数、拒付次数综合计算；当前风险评分：{risk_score}")
    st.json(st.session_state.memory_state)

    st.divider()
    st.subheader("🧠 策略核心 (Strategy Core)")
    # Ensure UI state matches session state
    st.session_state.strategy_card = ensure_strategy_card(st.session_state.memory_state, st.session_state.strategy_card)
    current_sc = st.session_state.strategy_card
    
    # Strategy Card Visualization
    with st.container(border=True):
        # Debug: Show if strategy_card is empty or invalid
        if not current_sc or not isinstance(current_sc, dict):
            st.error("⚠️ Strategy Card is empty or invalid!")
            st.json({"error": "strategy_card is None or not a dict", "value": str(current_sc)})
        else:
            st.markdown(f"**Strategy ID:** `{current_sc.get('strategy_id', 'Unknown')}`")
            st.markdown(f"**当前阶段 (Stage):** `{current_sc.get('stage', 'Unknown')}`")
            st.markdown(f"**施压等级 (Pressure):** `{current_sc.get('pressure_level', 'Unknown')}`")
        
        st.markdown("**📅 今日KPI (Today's KPI):**")
        for kpi in current_sc.get('today_kpi', []):
            st.markdown(f"- {kpi}")
            
        with st.expander("🛠️ 允许动作 (Allowed Actions)"):
            st.write(", ".join(current_sc.get('allowed_actions', [])))
            
        with st.expander("🛡️ 合规/其他 (Guardrails & Params)"):
            st.write("Guardrails:", current_sc.get('guardrails', []))
            st.write("Params:", current_sc.get('params', {}))
            if current_sc.get('notes'):
                st.info(f"Notes: {current_sc.get('notes')}")

    st.subheader("🧐 门控判断 (Critic Observation)")
    critic_data = st.session_state.last_critic
    if critic_data:
        decision = critic_data.get('decision')
        
        # Color code the decision
        color = "green" if decision == "CONTINUE" else "orange" if decision == "ADAPT_WITHIN_STRATEGY" else "red"
        st.markdown(f":{color}[**Decision:**] **{decision}**")
        
        reason = critic_data.get('decision_reason', '')
        st.info(f"**Reasoning:** {reason}")
        
        risk_flags = critic_data.get('risk_flags', [])
        if risk_flags:
            st.error(f"🚩 Risk Flags: {risk_flags}")
            
        with st.expander("详细 Critic 数据 (Raw)"):
            st.json(critic_data)
    else:
        st.write("（等待第一轮对话...）")