import streamlit as st
import json
import os
import yaml
from openai import OpenAI
import datetime
import re

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- Configuration & Setup ---
st.set_page_config(page_title="Collection Agent (Easy Mode)", layout="wide")

# Configure OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key and "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

if not api_key:
    st.error("OpenAI API Key is missing. Please set it in environment variables or .streamlit/secrets.toml")
    st.stop()
base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(api_key=api_key, base_url=base_url)
MODEL_NAME = "gpt-4o-mini"

# --- Memory Layer (NEW) ---
class MemoryLayer:
    """记忆层，用 LLM 做智能意图判断"""
    
    def __init__(self, llm_caller):
        self.llm_caller = llm_caller
        self.memory = {
            "intent_to_pay_today": None,  # 1 = 今天会还，0 = 今天不会还
            "payment_refusals": 0,
            "broken_promises": 0,
            "reason_category": "",
            "ability_score": "",
            "reason_detail": "",
            "unresolved_obstacles": [],
            # 历史分析结果
            "history_summary": "",
            "history_broken_promises": 0,
            "history_reason_category": "",
            "history_ability_score": "",
            # 收敛性追踪字段（多步信息收集）
            "has_ability_confirmed": False,   # 是否确认有钱还
            "payment_date_confirmed": "",     # 具体还款日期（如 "2025-12-30"）
            "payment_amount_confirmed": "",   # 具体金额（如 "2000" 或 "全额"）
            "payment_type_confirmed": "",     # "full" / "partial" / ""
            "extension_requested": False      # 是否请求展期
        }
    
    def detect_payment_intent(self, user_msg: str) -> int:
        """
        用 LLM 判断用户的意图：今天会还钱(1) 还是 今天不会还钱(0)
        返回：1 = 有意愿今天还，0 = 无意愿今天还
        """
        system_prompt = """你是意图判断专家。根据用户的话语，判断用户对"今天还钱"的意图。
只需输出一个数字：
- 1：用户表示今天会还钱（或者至少没有明确拒绝）
- 0：用户明确表示今天不会还钱（没钱、明天再说、有其他障碍等）

例子：
- "我今天下午3点给你还" → 1
- "现在没钱，明天再说" → 0
- "我在忙，稍后处理" → 0
- "我会尽快还给你" → 1
- "这事儿我还没想好" → 0
- "可以，我现在就转账" → 1

用户话语：{user_msg}

直接输出数字（0 或 1）。"""
        
        try:
            result = self.llm_caller(
                user_msg,
                system_prompt=system_prompt,
                json_mode=False
            )
            intent = int(result.strip())
            return 1 if intent == 1 else 0
        except Exception as e:
            log(f"Intent detection error: {e}")
            return 0
    
    def extract_from_dialogue(self, user_msg: str, conversation_history: list):
        """从用户消息中提取关键信息"""
        # ========== 第一步：LLM 意图判断 ==========
        intent = self.detect_payment_intent(user_msg)
        self.memory["intent_to_pay_today"] = intent
        
        # 如果意图是不还，计数拒付
        if intent == 0:
            self.memory["payment_refusals"] += 1
        
        # ========== 第二步：能力评估 ==========
        if "全" in user_msg and ("还" in user_msg or "支付" in user_msg):
            self.memory["ability_score"] = "full"
        elif "部分" in user_msg or "一点" in user_msg or "一些" in user_msg or "先" in user_msg:
            self.memory["ability_score"] = "partial"
        elif "没钱" in user_msg or "无力" in user_msg or "没办法" in user_msg:
            self.memory["ability_score"] = "zero"
        
        # ========== 第三步：原因分类 ==========
        if "失业" in user_msg or "没工作" in user_msg or "收入" in user_msg or "裁员" in user_msg:
            self.memory["reason_category"] = "unemployment"
        elif "生病" in user_msg or "医疗" in user_msg or "健康" in user_msg or "住院" in user_msg:
            self.memory["reason_category"] = "illness"
        elif "忘记" in user_msg or "忘了" in user_msg or "没想起" in user_msg:
            self.memory["reason_category"] = "forgot"
        elif "拒绝" in user_msg or "不想" in user_msg or "拖延" in user_msg or "不配合" in user_msg:
            self.memory["reason_category"] = "malicious_delay"
        else:
            self.memory["reason_category"] = "other"
        
        # ========== 第四步：具体理由（累积新增拒绝/理由片段） ==========
        # 原逻辑仅在 reason_detail 为空时记录一次，导致后续新的拒绝理由未被加入。
        # 调整为：当本轮意图判断为不还（intent == 0）且消息长度足够时，追加最新理由片段（去重，限长）。
        if len(user_msg) > 5:
            snippet = user_msg.strip()[:100]
            if intent == 0:
                existing = self.memory.get("reason_detail", "")
                if existing:
                    if snippet not in existing:
                        # 使用分号分隔并限制总长度，避免无限增长
                        self.memory["reason_detail"] = (existing + "；" + snippet)[:500]
                else:
                    self.memory["reason_detail"] = snippet
        
        # ========== 第五步：未解决障碍 ==========
        obstacle_keywords = {
            "开车": "正在开车",
            "忙": "正在忙碌",
            "会议": "在开会",
            "睡觉": "正在睡觉",
            "孩子": "带孩子",
            "病": "身体不适",
            "手机": "手机问题",
            "网络": "网络问题"
        }
        for kw, obstacle in obstacle_keywords.items():
            if kw in user_msg and obstacle not in self.memory["unresolved_obstacles"]:
                self.memory["unresolved_obstacles"].append(obstacle)
        
        # ========== 第六步：收敛性信息提取（时间、金额、类型、展期）==========
        import re
        from datetime import datetime, timedelta
        
        # 日期识别（明天/后天/12月30日/30号等）
        if "明天" in user_msg:
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            self.memory["payment_date_confirmed"] = tomorrow
        elif "后天" in user_msg:
            day_after = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
            self.memory["payment_date_confirmed"] = day_after
        elif re.search(r'(\d{1,2})[月号日]', user_msg):
            date_match = re.search(r'(\d{1,2})[月号日]', user_msg)
            self.memory["payment_date_confirmed"] = f"2025-12-{date_match.group(1)}"
        elif re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', user_msg):
            # 完整日期格式
            date_match = re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})', user_msg)
            if date_match:
                self.memory["payment_date_confirmed"] = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
        
        # 金额识别
        if "全额" in user_msg or "全部" in user_msg or "所有" in user_msg:
            self.memory["payment_type_confirmed"] = "full"
            self.memory["payment_amount_confirmed"] = "全额"
            self.memory["has_ability_confirmed"] = True
        elif "部分" in user_msg or "一部分" in user_msg or "先还" in user_msg:
            self.memory["payment_type_confirmed"] = "partial"
            # 尝试提取具体数字
            amount_match = re.search(r'(\d+)', user_msg)
            if amount_match:
                self.memory["payment_amount_confirmed"] = amount_match.group(1)
        
        # 能力确认（有钱/可以还）
        if "有钱" in user_msg or "可以还" in user_msg or "能还" in user_msg or "会还" in user_msg:
            self.memory["has_ability_confirmed"] = True
        elif "没钱" in user_msg or "钱不够" in user_msg or "没有钱" in user_msg:
            self.memory["has_ability_confirmed"] = False
        
        # 展期请求
        if "展期" in user_msg or "延期" in user_msg or "推迟" in user_msg or "宽限" in user_msg:
            self.memory["extension_requested"] = True
        
        log(f"Memory updated - Intent:{intent}, Date:{self.memory['payment_date_confirmed']}, Amount:{self.memory['payment_amount_confirmed']}, Type:{self.memory['payment_type_confirmed']}")
    
    def parse_history_summary(self, history_text: str):
        """
        用 LLM 自动解析历史记录，提取关键信息
        """
        if not history_text or len(history_text.strip()) < 10:
            log("History text is empty or too short, skipping parse_history_summary")
            return
        
        system_prompt = """你是历史记录分析专家。请将粘贴的过往催收记录分析并提取关键信息。
        
要求：
1. 输出必须是严格的 JSON 格式
2. 分析内容包括：
   - summary: 100-200字的中文摘要（包含核心阻碍、关键词、节点）
   - broken_promises: 统计历史失约总次数
   - reason_category: 从 [unemployment, illness, forgot, malicious_delay, other] 中选一个最佳匹配
   - ability_score: 从 [full, partial, zero] 中选一个最佳评估

示例输出：
{
  "summary": "客户历史上多次表示收入不稳定...",
  "broken_promises": 2,
  "reason_category": "unemployment",
  "ability_score": "partial"
}

只输出 JSON，不要其他文字。"""
        
        try:
            log(f"Starting to parse history summary, text length: {len(history_text)}")
            result = self.llm_caller(
                history_text,
                system_prompt=system_prompt,
                json_mode=True
            )
            data = json.loads(result)
            
            # 更新历史分析结果到记忆
            self.memory["history_summary"] = data.get("summary", "")
            self.memory["history_broken_promises"] = data.get("broken_promises", 0)
            self.memory["history_reason_category"] = data.get("reason_category", "")
            self.memory["history_ability_score"] = data.get("ability_score", "")
            
            # 如果当前的失约次数还是0，用历史的
            if self.memory["broken_promises"] == 0:
                self.memory["broken_promises"] = self.memory["history_broken_promises"]
            
            log(f"History parsed successfully: summary_length={len(data.get('summary', ''))}, broken_promises={data.get('broken_promises', 0)}")
        except json.JSONDecodeError as e:
            log(f"History parse JSON error: {e}")
        except Exception as e:
            log(f"History parse error: {type(e).__name__}: {e}")
    
    def get_memory_context(self) -> str:
        """生成记忆摘要，用于传给 Layer1 和 Layer2"""
        intent_text = "有意愿今天还" if self.memory.get('intent_to_pay_today') == 1 else "无意愿今天还"
        
        # 收敛性进度
        convergence_status = f"""
【关键信息收敛进度】
✓ 还款能力: {'已确认' if self.memory.get('has_ability_confirmed') else '未确认'} ({self.memory.get('ability_score', '未知')})
✓ 还款时间: {self.memory.get('payment_date_confirmed') or '未确认'}
✓ 还款金额: {self.memory.get('payment_amount_confirmed') or '未确认'}
✓ 付款方式: {self.memory.get('payment_type_confirmed') or '未确认'}
✓ 展期请求: {'是' if self.memory.get('extension_requested') else '否'}
"""
        
        # 构建当前画像
        summary = f"""
【客户当前画像】
- 今日意图: {intent_text} (intent={self.memory.get('intent_to_pay_today')})
- 拒付次数: {self.memory.get('payment_refusals', 0)}
- 失约次数: {self.memory.get('broken_promises', 0)}
- 能力评估: {self.memory.get('ability_score', '未知')}
- 原因分类: {self.memory.get('reason_category', '未知')}
- 具体理由: {self.memory.get('reason_detail', '暂无')}
- 待解决障碍: {', '.join(self.memory.get('unresolved_obstacles', [])) or '无'}

{convergence_status}
"""
        
        # 如果有历史分析，添加进来
        if self.memory.get('history_summary'):
            summary += f"""

【历史分析】
- 摘要: {self.memory.get('history_summary', '暂无')}
- 历史失约: {self.memory.get('history_broken_promises', 0)} 次
- 历史原因: {self.memory.get('history_reason_category', '未知')}
- 历史能力: {self.memory.get('history_ability_score', '未知')}
"""
        
        return summary.strip()
    
    def to_dict(self):
        return self.memory.copy()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def call_llm(prompt, system_prompt="You are a helpful assistant.", json_mode=False):
    try:
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            
        log(f"Calling LLM... Model: {MODEL_NAME}, JSON_Mode: {json_mode}")
        response = client.chat.completions.create(**kwargs)
        log("LLM Response received.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        log(f"LLM Error: {e}")
        st.error(f"LLM Error: {e}")
        return "Error"

# --- Agent Layers (Adapted for Streamlit) ---

class Layer1StrategyManager:
    def __init__(self, config, history_logs):
        self.config = config
        self.history_logs = history_logs

    def generate_initial_strategy(self, customer_profile):
        system_prompt = "你是催收策略经理。根据客户信息、历史记录以及公司的基础配置规则，制定今天的催收策略。"
        user_prompt = f"""
        客户资料：{json.dumps(customer_profile, ensure_ascii=False)}
        客户资料将有助于你分析客户当前的经济能力
        
        历史记录：{json.dumps(self.history_logs, ensure_ascii=False)}
        基于此记忆体，我们能总结出我们每天与客户交流的内，客户的还款意愿与还款能力吗？客户拒绝还款的理由是否合理
        注意，历史记录是非常重要的资料来源，说明我们已经与客户交流过了，但客户可能因为各种原因还没有还钱，现在希望是延续之前的对话，继续给客户施加压力。

        *** 关键约束 ***
        目标：尽快拿回钱。

        请按以下格式输出：
        
        【历史分析】
        （在此处简要分析客户昨天的态度、承诺、还款能力和意愿）
        
        【今日临时催收策略】
        1. 沟通基调：...
        2. 重点强调的内容：...
        
        【多步收敛路径】（核心：逐步收敛关键信息）
        step0: 查阅聊天历史，延续上次的聊天，询问客户之前阻碍还款的因素是否已经消除，今天是否有计划还钱
        Step1: 确认还款能力 - 追问客户当前是否有钱可还（是/否）
        Step2: 确认还款时间 - 如果有能力，追问具体哪天还（明确日期）
        Step3: 确认还款金额 - 追问能还多少（全额 ¥{customer_profile.get('amount_due', 0)} 还是部分）
        Step4: 确认付款方式 - 如果是部分还款，追问剩余如何处理（展期/分期）
        Step5: 锁定承诺 - 让客户明确承诺上述信息，记录为正式还款计划
        Step6: 如果客户说不能还钱，要问清楚客户不还钱的原因是什么？
        step7: 如果客户坚持今天不能还钱，开始施加压力给到客户，逐步提升。
          - 客户还钱，会获得好的信用，提升会员等级；
          - 会提升贷款额度和笔数；会获得还款折扣；
          - 会影响信用分，降低贷款额度；
          - 后续贷款会很困难，我们会停掉与客户的合作；
          - 拉入黑名单，客户的贷款行为受限，不只是在我们这不能借款，在哪里都不能借款。
          - 与紧急联系人沟通，让他协助还款
          - 联系你工作的单位，与你的领导进行沟通，了解你的经济状况
          - 在社交媒体上与你进行沟通，需要让你知道我们可以在社交媒体上找打你
          - 安排第三方上门进行催收
        
        ⚠️ 注意：每一步都要等客户明确回答后再进入下一步，不要一次问太多问题。
        """
        return call_llm(user_prompt, system_prompt)

    def update_strategy(self, current_strategy, feedback, chat_history, customer_profile, layer3_advice):
        system_prompt = """你是催收策略经理。Layer 3 评估层给出了当前策略的效果评估和改进建议，请根据建议调整策略。
        
重要：Layer 3 的建议包含两部分关键信息：
1. 回款可能性评估（HIGH/MEDIUM/LOW）
2. 信息收敛进度分析（哪些信息已收集，哪些还缺失）

你需要根据这些信息，调整策略的：
- 沟通基调和压力等级
- 【多步收敛路径】的优先级（优先收集缺失的关键信息）
- 话术和施压手段
"""
        user_prompt = f"""
        
客户资料：{json.dumps(customer_profile, ensure_ascii=False)}
历史记录：{json.dumps(self.history_logs, ensure_ascii=False)}
当前会话历史：{json.dumps(chat_history, ensure_ascii=False, indent=2)}

Layer 3 评估与建议（包含收敛性分析）：
{layer3_advice}

请根据 Layer 3 的建议方向，特别关注：
1. 如果回款可能性为 LOW，需要调整压力等级或话术
2. 如果信息收敛进度慢（如"3轮对话仅收集到1个信息"），需要在策略中明确强调"追问未确认的信息"
3. 如果某些关键信息（如时间、金额）一直未收集到，在【多步收敛路径】中优先追问

抓住客户的借款需求，制定有说服力的策略，比如客户是长期高频借贷者，它会担心信用缺失。

如果客户是初次借贷，或者低频借贷者，这个时候可以适当的给些压力，比如你如果一直不配合我们，我们会与紧急联系人沟通
        
请直接输出修改后的新策略，必须包含：
【今日临时催收策略】
1. 沟通基调：...（根据 Layer 3 的评估调整）
2. 重点强调的内容：...

【多步收敛路径】（根据 Layer 3 的收敛性分析，调整优先级）
Step1-7: （参考初始策略格式，但根据已收集/未收集信息调整顺序）

⚠️ 重点：根据 Layer 3 的【信息收敛进度】，在策略中明确指出"下一步优先追问哪个未确认的信息"。
        """
        return call_llm(user_prompt, system_prompt)

class Layer2Executor:
    def __init__(self, config):
        self.config = config
        # 如果 config 中没有 system_prompt，设置一个默认的
        if 'system_prompt' not in self.config:
            self.config['system_prompt'] = """你是一个专业的债务催收顾问。
你需要根据客户情况和制定的策略，与客户进行沟通。
保持专业态度，根据策略指导灵活调整沟通方式。
遵守所有法律和道德规范。"""

    def execute(self, strategy, chat_history, user_input, history_logs="", memory_context=""):
        # Clean up the base prompt
        cleaned_base_prompt = self.config.get('system_prompt', 'You are a helpful collection agent.')
        if "You must output a JSON object" in cleaned_base_prompt:
            cleaned_base_prompt = cleaned_base_prompt.replace("You must output a JSON object", "")
        if "Output Format" in cleaned_base_prompt:
            cleaned_base_prompt = cleaned_base_prompt.replace("Output Format", "")
        
        combined_system_prompt = f"""{cleaned_base_prompt}

# KEY CONTEXT (Read Carefully)
1. **HISTORY (Last Interaction)**:
{history_logs}

2. **CLIENT CURRENT STATE (Memory)**:
{memory_context}

3. **TODAY'S STRATEGY (Your Supreme Command)**:
{strategy}

4. **CONFIG RULES (Reference)**:
{json.dumps(self.config, ensure_ascii=False, indent=2)}

# INSTRUCTIONS
You are a professional Collection Agent - focused on **execution**, not analysis.

**YOUR ROLE**: 
- Execute "TODAY'S STRATEGY" from Layer 1 (策略层)
- Layer 1's strategy includes "多步收敛路径" - follow the steps systematically
- Check "CLIENT CURRENT STATE" to see what information is already collected
- Focus on collecting the next piece of missing information according to the strategy

**EXECUTION PRINCIPLES**:
- Follow "TODAY'S STRATEGY" and its "多步收敛路径" (Step1-5)
- Use "CLIENT CURRENT STATE" to know what's collected (✅) vs. missing (⏳)
- Ask for ONE missing information at a time
- Be natural - don't sound like a robot reading a script
- If strategy says "Step2: 确认还款时间" and time is missing (⏳), ask about time

**TONE ADJUSTMENT**:
- If intent=1 (client willing to pay today): be helpful, supportive, collaborative
- If intent=0 (client unwilling): increase pressure, use warnings, be firm
- If payment_refusals is high: escalate pressure, mention consequences

**REQUIRED OUTPUT (JSON FORMAT)**:
You MUST respond with VALID JSON containing these exact fields:
{{
  "user_analysis": "Brief analysis of client's current attitude",
  "strategy_check": "Quote the specific strategy sentence or step that applies here",
  "tactical_plan": "Which missing info to collect based on strategy's 多步收敛路径",
  "response": "Final Chinese message - be natural, professional, vary sentence structure"
}}

IMPORTANT: Output ONLY valid JSON. No markdown code blocks, no explanation text before or after JSON."""
        
        messages = [{"role": "system", "content": combined_system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_input})
        
        try:
            log("Layer 2: Sending request to OpenAI with JSON format...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            log("Layer 2: Response received.")
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            # Combine thoughts for UI display
            full_thought = (
                f"**User Analysis**: {data.get('user_analysis', 'N/A')}\n\n"
                f"**Strategy Check**: {data.get('strategy_check', 'N/A')}\n\n"
                f"**Tactical Plan**: {data.get('tactical_plan', 'N/A')}"
            )
            
            return data.get('response', ''), full_thought
        except json.JSONDecodeError as e:
            log(f"Layer 2 JSON Parse Error: {e}")
            log(f"Raw response: {content}")
            return f"System Error: JSON parse failed - {str(e)[:100]}", ""
        except Exception as e:
            log(f"Layer 2 Error: {e}")
            import traceback
            log(traceback.format_exc())
            return f"System Error: {str(e)[:200]}", ""

class Layer3Evaluator:
    def evaluate(self, chat_history, history_logs, customer_profile, current_strategy, memory_context=""):
        system_prompt = """
        你是催收策略的评估专家和信息收敛分析师。
        你的任务有三个：
        1. 评估当前策略在客户身上的有效性，特别是"客户回款的可能性"
        2. 分析关键信息的收敛进度（5个关键信息：能力、时间、金额、方式、展期）
        3. 给 Layer 1 策略层提供优化建议

        你需要综合分析：
        1. 过去的催收历史记录（历史日志）
        2. 当前的完整对话历史
        3. 客户的资料信息
        4. 当前的催收策略
        5. 客户记忆状态（Memory）- 包含已收集和未收集的信息

        【关键信息收敛分析】
        检查以下 5 个关键信息的收集状态：
        1. 还款能力 (has_ability_confirmed) - 客户是否有钱还
        2. 还款时间 (payment_date_confirmed) - 具体哪天还
        3. 还款金额 (payment_amount_confirmed) - 能还多少
        4. 付款方式 (payment_type_confirmed) - 全额还是部分
        5. 展期请求 (extension_requested) - 是否请求展期
        
        根据 Memory 状态，分析：
        - 哪些信息已经收集到（✅）
        - 哪些信息还缺失（⏳）
        - 信息收集的优先级顺序
        - 当前策略是否有效推进信息收集
        
        【重要】你的建议是给 Layer 1（策略层）的，不是给 Layer 2（执行层）的。
        - 如果收敛进度慢，建议 Layer 1 调整策略（比如增加压力、改变话术）
        - 如果回款可能性低，建议 Layer 1 重新制定策略框架
        
        请按以下格式输出：
        【分析】简要分析客户的抗拒点或困难，以及当前策略的有效性。
        【回款可能性】HIGH / MEDIUM / LOW
        【信息收敛进度】已收集：[列出已确认的信息，如"能力(有钱)、时间(2025-12-30)"] / 未收集：[列出缺失的信息]
        【收敛效率评估】(评价当前策略是否有效推进信息收集，如"收敛速度慢，3轮对话仅收集到1个信息")
        【给 Layer 1 的建议】(如果可能性为LOW 或 收敛效率低，建议调整策略方向；比如"建议在策略中增加明确的时间追问环节")
        """
        user_prompt = f"""
客户记忆状态（Memory）：
{memory_context}

今日完整对话历史：{json.dumps(chat_history, ensure_ascii=False, indent=2)}

客户资料：{json.dumps(customer_profile, ensure_ascii=False)}

当前策略：
{current_strategy}
"""
        return call_llm(user_prompt, system_prompt)

# --- Main App Logic ---

def main():
    # Sidebar: Configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Config File Selection
    config_files = [f for f in os.listdir("configs") if f.endswith(".yaml")]
    selected_config = st.sidebar.selectbox("Select Config", config_files, index=config_files.index("T0.yaml") if "T0.yaml" in config_files else 0)
    
    if selected_config:
        config = load_config(os.path.join("configs", selected_config))
    else:
        st.error("No config file found!")
        return

    # Customer Profile
    st.sidebar.subheader("Customer Profile")
    default_profile = {
        "name": "LESTARI",
        "amount": "Rp 1.250.000",
        "due_date": "2025-12-17",
        "current_time": "2025-12-17",
        "gender": "Male",
        "age": "30",
        "frenquency_borrow":"often",
        "payment_timelyness":"sometimes late",
        "meaber_level": "high"
    }
    profile_str = st.sidebar.text_area("Edit Profile (JSON)", json.dumps(default_profile, indent=2, ensure_ascii=False), height=250)
    try:
        customer_profile = json.loads(profile_str)
    except:
        st.sidebar.error("Invalid JSON in Profile")
        customer_profile = default_profile

    # History Logs
    st.sidebar.subheader("History Logs")
    default_history = '''
【2025年12月26日 第一次催收】
- 11:00 问客户今天是否有钱还款，客户表示"最近生意不太好，但今天下午应该有钱"
- 12:30 客户确认下午3点能还，金额为全额 ¥5000
- 15:00 承诺时间已过，客户未回复，提醒一次，客户已读不回
- 16:00 再次提醒，客户表示"可能晚点，我现在在处理其他事情"

【2025年12月27日 第二次催收】
- 09:00 早上提醒，客户表示"昨天确实没时间，今天晚上一定还"
- 19:00 再次提醒，无回应
- 20:00 多次提醒后，客户才回复"我现在没钱，明天再说吧"
- 结果：又失约一次

【分析】
客户已失约2次，第一次承诺全额未兑现，第二次拖延到要求部分还款。
态度从"肯定能还"变成"没钱"，有推诿迹象。
    '''
    history_logs = st.sidebar.text_area("Edit History Logs", default_history, height=200)
    
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "strategy" not in st.session_state:
        st.session_state.strategy = None
    if "layer1_analysis" not in st.session_state:
        st.session_state.layer1_analysis = None
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryLayer(llm_caller=call_llm)
    
    # 解析历史记录（仅在首次初始化时执行一次）
    if "history_parsed" not in st.session_state:
        with st.spinner("🔍 Analyzing history logs..."):
            st.session_state.memory.parse_history_summary(history_logs)
        st.session_state.history_parsed = True

    # Reset Button
    if st.sidebar.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.strategy = None
        st.session_state.layer1_analysis = None
        st.session_state.memory = MemoryLayer(llm_caller=call_llm)
        st.session_state.history_parsed = False
        st.rerun()

        # --- Main Area ---
    st.title("🤖 Collection Agent (Easy Mode)")

    # Layout: 2 Columns
    # Col 1: Chat (60%)
    # Col 2: Strategy & Analysis (40%)
    col_chat, col_info = st.columns([2, 1.2])

    # --- Right Column: Strategy & Analysis ---
    with col_info:
        st.subheader("🧠 Agent Brain (Strategy & Analysis)")
        
        # NEW: 显示当前记忆状态
        with st.expander("👤 Client Memory (Current)", expanded=True):
            memory_dict = st.session_state.memory.to_dict()
            
            # 核心指标：意图
            intent_emoji = "✅ 有意愿还" if memory_dict.get('intent_to_pay_today') == 1 else "❌ 无意愿还"
            st.write(f"**今日意图**: {intent_emoji}")
            
            # 行为指标
            st.write(f"**拒付次数**: {memory_dict.get('payment_refusals', 0)} 次")
            st.write(f"**失约次数**: {memory_dict.get('broken_promises', 0)} 次")
            
            # 能力和原因
            st.write(f"**能力评估**: {memory_dict.get('ability_score', '未知')}")
            st.write(f"**原因分类**: {memory_dict.get('reason_category', '未知')}")
            
            # 障碍
            if memory_dict.get('unresolved_obstacles'):
                st.write(f"**待解决**: {', '.join(memory_dict['unresolved_obstacles'])}")
            
            # ======== 新增：收敛性进度显示 ========
            st.divider()
            st.markdown("**🎯 关键信息收敛进度**")
            
            conv_col1, conv_col2 = st.columns(2)
            with conv_col1:
                ability_icon = "✅" if memory_dict.get('has_ability_confirmed') else "⏳"
                st.write(f"{ability_icon} **还款能力**: {memory_dict.get('ability_score', '未确认')}")
                
                date_icon = "✅" if memory_dict.get('payment_date_confirmed') else "⏳"
                st.write(f"{date_icon} **还款时间**: {memory_dict.get('payment_date_confirmed') or '未确认'}")
                
                extension_icon = "⚠️" if memory_dict.get('extension_requested') else "✅"
                st.write(f"{extension_icon} **展期请求**: {'是' if memory_dict.get('extension_requested') else '否'}")
            
            with conv_col2:
                amount_icon = "✅" if memory_dict.get('payment_amount_confirmed') else "⏳"
                st.write(f"{amount_icon} **还款金额**: {memory_dict.get('payment_amount_confirmed') or '未确认'}")
                
                type_icon = "✅" if memory_dict.get('payment_type_confirmed') else "⏳"
                payment_type_text = {"full": "全额", "partial": "部分", "": "未确认"}.get(memory_dict.get('payment_type_confirmed', ''), '未确认')
                st.write(f"{type_icon} **付款方式**: {payment_type_text}")
            
            # 历史分析结果
            if memory_dict.get('history_summary'):
                st.divider()
                st.markdown("**📜 历史分析**")
                st.caption(memory_dict.get('history_summary', '暂无'))
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("历史失约", f"{memory_dict.get('history_broken_promises', 0)} 次")
                with col2:
                    st.metric("历史能力", memory_dict.get('history_ability_score', '未知'))
        
        st.divider()
        
        # 1. Global Strategy (Layer 1)
        with st.expander("📋 Daily Strategy (Layer 1)", expanded=True):
            if not st.session_state.strategy:
                # Auto-initialize: Generate strategy and opening message immediately
                with st.spinner("Layer 1 Manager is analyzing history and generating strategy..."):
                    layer1 = Layer1StrategyManager(config, [history_logs])
                    full_strategy_output = layer1.generate_initial_strategy(customer_profile)
                    st.session_state.strategy = full_strategy_output
                    
                    # Generate opening message with memory context
                    memory_context = st.session_state.memory.get_memory_context()
                    layer2 = Layer2Executor(config)
                    # Layer2 now receives memory_context for awareness of history
                    opening_response, thought = layer2.execute(
                        full_strategy_output, 
                        [], 
                        "Start the conversation naturally. If there's history with the customer, acknowledge it and continue. If new customer, introduce yourself and explain your role.",
                        history_logs,
                        memory_context  # NEW: Pass memory context so Layer2 knows the history
                    )
                    
                    st.session_state.messages.append({"role": "assistant", "content": opening_response, "thought": thought})
                    st.rerun()
            else:
                st.info(st.session_state.strategy)
                
                # Add a button to force regenerate strategy if needed
                if st.button("Regenerate Strategy"):
                    st.session_state.strategy = None
                    st.session_state.messages = [] # Also clear messages to restart conversation
                    st.rerun()
        
        st.divider()
        st.markdown("**Thinking Process Log**")

    # --- Left Column: Chat Interface ---
    with col_chat:
        st.subheader("💬 Chat Interface")

    # --- Render History ---
    for i, msg in enumerate(st.session_state.messages):
        # Chat Content (Col 1)
        with col_chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Analysis Content (Col 2)
        if msg["role"] == "assistant":
            with col_info:
                st.markdown(f"**Turn {i+1} Analysis**")
                
                # Layer 3
                if "layer3_evaluation" in msg and msg["layer3_evaluation"]:
                    with st.expander("🛡️ Layer 3 Evaluation", expanded=False):
                        st.caption(msg["layer3_evaluation"])
                
                # Layer 1 Update Event
                if "layer1_update" in msg and msg["layer1_update"]:
                    st.warning(f"🔄 Strategy Updated at Turn {i+1}")
                    with st.expander("View New Strategy"):
                        st.caption(msg["layer1_update"])

                # Layer 2 Thought
                if "thought" in msg and msg["thought"]:
                    with st.expander("💭 Layer 2 Thought", expanded=False):
                        st.caption(msg["thought"])
                
                st.divider()

    # --- User Input Handling ---
    if prompt := st.chat_input("Type your reply here..."):
        # 1. 追踪到记忆（核心改进：调用 LLM 做意图判断 0/1）
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with col_chat:
            with st.chat_message("user"):
                st.write(prompt)
        
        # 2. 初始化 Layers
        layer1 = Layer1StrategyManager(config, [history_logs])
        layer2 = Layer2Executor(config)
        layer3 = Layer3Evaluator()
        
        # 3. 分析记忆（包含 LLM 意图判断）
        with st.spinner("🧠 Analyzing user intent and building memory..."):
            st.session_state.memory.extract_from_dialogue(prompt, st.session_state.messages)
        
        # 4. 生成记忆摘要
        memory_context = st.session_state.memory.get_memory_context()
        
        # 5. Layer 3: Evaluation（新增：传入 memory_context）
        with col_info:
            st.markdown(f"**Current Turn Analysis**")
            with st.spinner("🛡️ Layer 3 Evaluating..."):
                evaluation_output = layer3.evaluate(
                    st.session_state.messages,
                    [history_logs],
                    customer_profile,
                    st.session_state.strategy,
                    memory_context  # NEW：传入记忆上下文用于收敛分析
                )
            
            with st.expander("🛡️ Layer 3 Evaluation", expanded=True):
                st.caption(evaluation_output)
        
        # 6. 检查是否需要更新策略
        is_low_prob = "LOW" in evaluation_output or "可能性】LOW" in evaluation_output
        
        layer1_update_text = None
        if is_low_prob:
            with col_info:
                with st.spinner("⚠️ Low probability! Updating Strategy..."):
                    new_strategy = layer1.update_strategy(
                        st.session_state.strategy, 
                        prompt, 
                        st.session_state.messages,
                        customer_profile,
                        evaluation_output
                    )
                    st.session_state.strategy = new_strategy
                    layer1_update_text = new_strategy
                    st.warning("🔄 Strategy Updated!")
                    with st.expander("View New Strategy"):
                        st.caption(new_strategy)
        
        # 7. Layer 2: Execution（关键改进：传入记忆上下文）
        with col_info:
            with st.spinner("💭 Layer 2 Thinking..."):
                response, thought = layer2.execute(
                    st.session_state.strategy, 
                    st.session_state.messages[:-1], 
                    prompt,
                    history_logs,
                    memory_context  # NEW：传入记忆上下文
                )
                with st.expander("🕵️ Layer 2 Execution Monitor (Thought)", expanded=True):
                    st.write(thought)
                st.divider()
        
        # 8. 输出响应
        with col_chat:
            with st.chat_message("assistant"):
                st.write(response)
        
        # 9. 保存到历史
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "thought": thought,
            "layer3_evaluation": evaluation_output,
            "layer1_update": layer1_update_text
        })
        
        st.rerun()

    



if __name__ == "__main__":
    main()
