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
# MODEL = "gpt-4o-2024-05-13" # Replaced "gpt-5.2" with a valid model name for now, or keep as user requested if they have access. User had "gpt-5.2" in the snippet. I will stick to "gpt-4o" or similar to ensure it runs, or "gpt-4o-mini". But to be safe, I'll use "gpt-4o" or just respect their code if I can.
# Actually, the user's snippet had "gpt-5.2". I should probably keep it if they have it mapped, or change it to something standard like "gpt-4o".
# Given the user context says "gpt52", maybe they mapped it?
# Let's use "gpt-4o" to be safe, or check if they have a specific mapper.
# Wait, the user file name is test_gpt52.py.
# I will output the code exactly as they had it, but maybe change the model to a known working one like "gpt-4o" to avoid immediate 400 errors, unless "gpt-5.2" checks out.
# I'll stick to "gpt-4o" as a safe bet for a demo, alerting them.
# actually, in the user provided snippet (Step 16), it was `MODEL = "gpt-5.2"`. I'll keep it but add a comment.

MODEL = "gpt-5.2" 

# OpenAI client (expects OPENAI_API_KEY env var)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================================================
# Business Rules (your definitions)
# =========================================================
def dpd_to_stage(dpd: int) -> str:
    # DPD<0 Stage0; DPD=0 Stage1; DPD=1 Stage2; DPD=2-3 Stage3; DPD>3 Stage4
    if dpd < 0:
        return "Stage0"
    if dpd == 0:
        return "Stage1"
    if dpd == 1:
        return "Stage2"
    if 2 <= dpd <= 3:
        return "Stage3"
    return "Stage4"


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
    decision_reason: str = Field(default="（未提供了理由）")
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
- DPD>=0时遵守“当天闭环”：必须促成（今天全额）或（今天部分+今天时间点）或（若资格允许）今天完成展期。
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

输出必须是严格JSON，且只输出JSON。
"""


def build_meta_system_prompt() -> str:
    return f"""你是元策略生成器（Meta / Controller）。
输入：memory_state, critic_result, recent_dialogue。
输出：更新后的strategy_card（严格JSON）。

规则：
- 必须遵守合规硬约束。
- Stage必须与DPD映射一致（memory_state.stage），不要擅自改Stage。
- DPD>=0必须遵守“当天闭环”：不允许给未来承诺空间。今天必须落地：全额、或部分+今天时间点、或（若extension_eligible=true）今天完成展期。
- 展期仅在extension_eligible=true时允许作为策略分支，且目标必须是“今天完成展期”。
- Stage4且sop_trigger_named_escalation=true且approval_id存在时，允许以流程告知口吻更明确提及升级处置（不得羞辱/夸大/公开曝光）。

只输出strategy_card JSON，不要任何解释文本。
"""


def build_executor_system_prompt() -> str:
    return f"""你是[机构名]负责该账户的还款跟进人员，通过WhatsApp沟通（此处用中文输出）。
要求：
- 不提“我是AI”，不提“转人工专员”。
- 每次只输出一条要发给用户的消息。
- 每次只问1个主问题（最多附带1句极短确认）。
- DPD>=0必须遵守“当天闭环”：促成今天全额，或今天部分+今天时间点，或（若允许）今天完成展期。
- 施压方式：仅限事实（DPD/费用/流程）+ 真实可执行的流程告知（可审计），不羞辱、不夸大、不虚构。
- 若Stage4且sop_trigger=true且approval_id存在，可更明确告知将按流程升级处置（仍需克制），避免披露第三方细节。
输出只给“要发给用户的一条消息”，不要解释。
"""


# =========================================================
# LLM helpers
# =========================================================
def call_llm_text(system: str, user: str, temperature: float = 0.2) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {e}"


def call_critic(strategy_card: Dict[str, Any], memory_state: Dict[str, Any], dialogue: List[Dict[str, str]]) -> CriticResult:
    system = build_critic_system_prompt()
    payload = {
        "strategy_card": strategy_card,
        "memory_state": memory_state,
        "recent_dialogue": dialogue[-12:],
    }
    user = "请评估并输出JSON：\n" + json.dumps(payload, ensure_ascii=False)

    # Preferred: JSON schema structured outputs
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={
                "type": "json_object"
            }
        )
        data = json.loads(resp.choices[0].message.content)
        return CriticResult(**data)
    except Exception as e:
        # Fallback: best-effort JSON + validation
        st.error(f"Critic Error: {e}")
        return CriticResult(
            decision="ESCALATE_TO_META",
            decision_reason=f"critic_failed: {str(e)[:90]}",
            reason_codes=["critic_failed"],
            micro_edits_for_executor=MicroEdits(
                ask_style="forced_choice",
                confirmation_format="amount_time_today",
                tone="polite_firm",
                language="zh"
            ),
        )


def call_meta(memory_state: Dict[str, Any], critic: CriticResult, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
    system = build_meta_system_prompt()
    payload = {
        "memory_state": memory_state,
        "critic_result": critic.model_dump(),
        "recent_dialogue": dialogue[-12:],
    }
    user = "请生成新的strategy_card JSON：\n" + json.dumps(payload, ensure_ascii=False)

    # Preferred: JSON schema structured outputs
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={
                "type": "json_object"
            }
        )
        data = json.loads(resp.choices[0].message.content)
        sc = StrategyCard(**data)

        # Hard alignment: stage must match memory_state.stage
        forced_stage = memory_state.get("stage")
        if forced_stage and sc.stage != forced_stage:
            sc.stage = forced_stage
            sc.notes = (sc.notes or "") + " | stage_forced_to_memory_state"

        return sc.model_dump()

    except Exception as e:
        # Fallback
        st.error(f"Meta Error: {e}")
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
            params={"meta_error": f"{str(e)[:80]}"},
            notes="meta_fallback"
        )
        return fallback.model_dump()


def call_executor(strategy_card: Dict[str, Any], memory_state: Dict[str, Any], dialogue: List[Dict[str, str]], micro: MicroEdits) -> str:
    system = build_executor_system_prompt()
    payload = {
        "strategy_card": strategy_card,
        "memory_state": memory_state,
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
    if strategy_card:
        try:
            return StrategyCard(**strategy_card).model_dump()
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

    # Refresh stage by DPD every turn
    dpd = int(st.session_state.memory_state.get("dpd", 0))
    stage = dpd_to_stage(dpd)
    st.session_state.memory_state["stage"] = stage

    # Compute SOP trigger and store
    bp = int(st.session_state.memory_state.get("broken_promises", 0))
    st.session_state.memory_state["sop_trigger_named_escalation"] = sop_trigger_named_escalation(dpd, bp)

    # Ensure strategy exists
    st.session_state.strategy_card = ensure_strategy_card(st.session_state.memory_state, st.session_state.strategy_card)

    # 1) Critic (Gate)
    critic = call_critic(st.session_state.strategy_card, st.session_state.memory_state, st.session_state.dialogue)
    st.session_state.last_critic = critic.model_dump()

    # 2) Apply critic memory writes
    st.session_state.memory_state = apply_memory_write(st.session_state.memory_state, critic.memory_write)

    # 3) Meta rewrite strategy if needed
    if critic.decision == "ESCALATE_TO_META":
        st.session_state.strategy_card = call_meta(st.session_state.memory_state, critic, st.session_state.dialogue)

    # 4) Executor response
    reply = call_executor(
        st.session_state.strategy_card,
        st.session_state.memory_state,
        st.session_state.dialogue,
        critic.micro_edits_for_executor
    )
    st.session_state.dialogue.append({"role": "assistant", "content": reply})


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(layout="wide")
st.title("三层Prompt（Meta/Executor/Critic）+ 增量记忆 门控编排（单文件，可直接跑）")

if "dialogue" not in st.session_state:
    st.session_state.dialogue = []
if "memory_state" not in st.session_state:
    st.session_state.memory_state = {
        "customer_id": "C-demo",
        "dpd": 1,
        "broken_promises": 0,
        "extension_eligible": False,  # toggle in UI
        "approval_id": "APR-001",      # needed for Stage4 named escalation mention
        "allowed_contact_hours": "08:00-20:00 WIB",
        "stage": "Stage2",
        "no_response_streak": 0,
    }
if "strategy_card" not in st.session_state:
    st.session_state.strategy_card = None
if "last_critic" not in st.session_state:
    st.session_state.last_critic = None

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
    st.subheader("业务参数/记忆（可调）")
    dpd = st.number_input("DPD（可为负）", value=int(st.session_state.memory_state.get("dpd", 0)), step=1)
    bp = st.number_input("broken_promises（失约次数）", value=int(st.session_state.memory_state.get("broken_promises", 0)), step=1)
    ext = st.checkbox("extension_eligible（可展期）", value=bool(st.session_state.memory_state.get("extension_eligible", False)))
    approval_id = st.text_input("approval_id（Stage4点名升级需存在）", value=str(st.session_state.memory_state.get("approval_id", "")))

    st.session_state.memory_state["dpd"] = int(dpd)
    st.session_state.memory_state["broken_promises"] = int(bp)
    st.session_state.memory_state["extension_eligible"] = bool(ext)
    st.session_state.memory_state["approval_id"] = approval_id

    # Auto-refresh stage + SOP trigger displayed
    st.session_state.memory_state["stage"] = dpd_to_stage(int(dpd))
    st.session_state.memory_state["sop_trigger_named_escalation"] = sop_trigger_named_escalation(int(dpd), int(bp))

    st.caption("Stage 会根据 DPD 自动刷新；sop_trigger 会自动计算。")
    st.json(st.session_state.memory_state)

    st.subheader("StrategyCard（当前策略卡）")
    current_sc = ensure_strategy_card(st.session_state.memory_state, st.session_state.strategy_card)
    st.json(current_sc)

    st.subheader("Last Critic（门控输出）")
    st.json(st.session_state.last_critic)