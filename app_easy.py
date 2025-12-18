import streamlit as st
import json
import os
import yaml
from openai import OpenAI

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

# --- Helper Functions ---
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
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return "Error"

# --- Agent Layers (Adapted for Streamlit) ---

class Layer1StrategyManager:
    def __init__(self, config, history_logs):
        self.config = config
        self.history_logs = history_logs

    def generate_initial_strategy(self, customer_profile):
        system_prompt = "你是催收策略经理。根据客户信息、历史记录以及公司的基础配置规则，制定今天的临时催收策略。"
        user_prompt = f"""
        客户资料：{json.dumps(customer_profile, ensure_ascii=False)}
        
        请制定一个“今日临时催收策略”，指导催收员如何与该客户沟通。
        注意，历史记录是非常重要的资料来源，说明我们已经与客户交流过了，但客户没有还钱，现在希望是延续之前的对话，继续给客户施加压力。

        *** 关键约束 ***
        你制定的策略必须严格遵守“公司基础配置/规则”中的“HARD RULES”。
        例如：如果规则禁止部分还款，你的策略中绝对不能建议或暗示可以接受部分还款。
        
        请先分析历史记录：
        历史记录：{json.dumps(self.history_logs, ensure_ascii=False)}
        基于此记忆体，我们能总结出客户的还款意愿，与还款能力吗？以及我们接下里催收策略是什么。目标是尽快拿回钱。
        
        请按以下格式输出：
        
        【历史分析】
        （在此处简要分析客户昨天的态度、承诺、还款能力和意愿）
        
        【今日临时催收策略】
        1. 沟通基调：...
        2. 重点强调的内容：...
        """
        return call_llm(user_prompt, system_prompt)

    def update_strategy(self, current_strategy, feedback, chat_history, customer_profile, layer3_advice):
        system_prompt = "你是催收策略经理。Layer 3 评估认为当前策略回款可能性较低，请根据建议调整策略。"
        user_prompt = f"""
        Layer 3 评估与建议：
        {layer3_advice}
        
        客户资料：{json.dumps(customer_profile, ensure_ascii=False)}
        历史记录：{json.dumps(self.history_logs, ensure_ascii=False)}
        当前会话历史：{json.dumps(chat_history, ensure_ascii=False, indent=2)}
        当前策略：{current_strategy}

        请根据 Layer 3 的建议方向，重新制定今天的催收策略。
        策略应具体指导催收员如何使用建议的激励或施压手段（如黑名单、提额、折扣等）来突破客户防线。
        
        *** 关键约束 ***
        修改后的策略依然必须严格遵守“公司基础配置/规则”中的“HARD RULES”。
        例如：如果规则禁止部分还款，即使为了激励客户，也不能违反此规则（除非申请折扣是允许的例外）。
        
        请直接输出修改后的新策略。
        """
        return call_llm(user_prompt, system_prompt)

class Layer2Executor:
    def __init__(self, config):
        self.config = config
        self.base_system_prompt = self.config.get('system_prompt', '')

    def execute(self, strategy, chat_history, user_input):
        # Clean up the base prompt
        cleaned_base_prompt = self.base_system_prompt.replace("You must output a JSON object", "")
        cleaned_base_prompt = cleaned_base_prompt.replace("Output Format", "")
        
        combined_system_prompt = f"""
        {cleaned_base_prompt}
        
        ---
        TODAY'S TEMPORARY STRATEGY (FROM MANAGER):
        {strategy}
        ---

        公司基础配置/规则：
        {json.dumps(self.config, ensure_ascii=False, indent=2)}

        请先依据历史聊天情况，总结下过去客户的还款表现，延续与客户的对话。拉进与客户的关系，并提醒客户今天的借款该还钱了！

        # INSTRUCTIONS
        1. You must follow the "Hard Rules" in the configuration AND the "Temporary Strategy" above.
        2. If there is a conflict, the "Hard Rules" (like NO Partial Payments) take precedence.
        3. **LANGUAGE**: You MUST reply in CHINESE (中文).
        4. **FORMAT**: You MUST output a JSON object with two fields:
           - "thought": Your internal reasoning (analyze customer intent, check rules, decide strategy).
           - "response": The final message to send to the customer (in Chinese).
        5. **CONTEXT**: You have a history with this customer. Do NOT re-introduce yourself (e.g., "Hello, I am Cindy") unless it is the very first message of a new conversation or the customer asks. Treat this as an ongoing professional relationship.
        6. **TONE**: Be CONCISE and PROFESSIONAL. Avoid excessive politeness or flowery language. You are a debt collector, not a customer service representative. Get straight to the point about payment.
        """
        
        messages = [{"role": "system", "content": combined_system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_input})
        
        content = call_llm(user_input, combined_system_prompt, json_mode=True) # Using call_llm wrapper but passing prompt as user_input is tricky because call_llm constructs messages. 
        # Let's use client directly here or adjust call_llm. 
        # Adjusting logic below to use client directly for full control over messages list.
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return data.get('response', ''), data.get('thought', '')
        except Exception as e:
            return f"System Error: {e}", ""

class Layer3Evaluator:
    def evaluate(self, last_user_input, last_agent_response):
        system_prompt = """
        你是催收策略的评估专家。
        你的任务是评估当前策略在客户身上的有效性，特别是“客户回款的可能性”。
        
        请分析客户的最新回复，判断按照当前策略继续下去，客户回款的可能性是 大(HIGH)、中(MEDIUM) 还是 小(LOW)。
        
        如果回款可能性为 小(LOW)，你需要给出策略调整建议。你可以从以下6个维度中选择最适合当前情况的切入点（指导第一层策略修改）：
        1. 客户还钱，会获得好的信用，提升会员等级；
        2. 会提升贷款额度和笔数；
        3. 会获得还款折扣；
        4. 会影响信用分，降低贷款额度；
        5. 后续贷款会很困难，我们会停掉与客户的合作；
        6. 拉入黑名单，客户的贷款行为受限，不只是在我们这不能借款，在哪里都不能借款。

        请按以下格式输出：
        【分析】简要分析客户的抗拒点或困难，以及当前策略为何无效。
        【回款可能性】HIGH / MEDIUM / LOW
        【建议方向】(如果可能性为LOW，请从上述6点中选择1-2点建议；否则留空)
        """
        user_prompt = f"Agent Response: {last_agent_response}\nCustomer Input: {last_user_input}"
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
   【过去聊天总结】
在上一轮催收中已做出明确、具体的还款承诺（如“今天下午3点准时还”），但到达承诺时间点后，不仅未还款，且对后续所有提醒消息已读不回，陷入完全静默。”
    '''
    history_logs = st.sidebar.text_area("Edit History Logs", default_history, height=200)
    
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "strategy" not in st.session_state:
        st.session_state.strategy = None
    if "layer1_analysis" not in st.session_state:
        st.session_state.layer1_analysis = None

    # Reset Button
    if st.sidebar.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.strategy = None
        st.session_state.layer1_analysis = None
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
        
        # 1. Global Strategy (Layer 1)
        with st.expander("📋 Daily Strategy (Layer 1)", expanded=True):
            if not st.session_state.strategy:
                # Auto-initialize: Generate strategy and opening message immediately
                with st.spinner("Layer 1 Manager is analyzing history and generating strategy..."):
                    layer1 = Layer1StrategyManager(config, [history_logs])
                    full_strategy_output = layer1.generate_initial_strategy(customer_profile)
                    st.session_state.strategy = full_strategy_output
                    
                    # Generate opening message
                    layer2 = Layer2Executor(config)
                    opening_instruction = "(System Instruction: Start the conversation now. If this is a new contact, introduce yourself. If there is history, continue naturally based on the strategy.)"
                    opening_response, thought = layer2.execute(full_strategy_output, [], opening_instruction)
                    
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
        # 1. Render User Message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with col_chat:
            with st.chat_message("user"):
                st.write(prompt)

        # 2. Process Assistant Response
        # Initialize Layers
        layer1 = Layer1StrategyManager(config, [history_logs])
        layer2 = Layer2Executor(config)
        layer3 = Layer3Evaluator()

        # --- Layer 3: Evaluation ---
        last_agent_response = st.session_state.messages[-2]['content'] if len(st.session_state.messages) > 1 else ""
        
        with col_info:
            st.markdown(f"**Current Turn Analysis**")
            with st.spinner("🛡️ Layer 3 Evaluating..."):
                evaluation_output = layer3.evaluate(prompt, last_agent_response)
            
            with st.expander("🛡️ Layer 3 Evaluation", expanded=True):
                st.caption(evaluation_output)

        # Check for LOW probability trigger
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

        # --- Layer 2: Execution ---
        with col_info:
            with st.spinner("💭 Layer 2 Thinking..."):
                response, thought = layer2.execute(
                    st.session_state.strategy, 
                    st.session_state.messages[:-1], 
                    prompt
                )
                with st.expander("💭 Layer 2 Thought", expanded=True):
                    st.caption(thought)
                st.divider()

        # --- Output Response ---
        with col_chat:
            with st.chat_message("assistant"):
                st.write(response)
        
        # Save to History
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "thought": thought,
            "layer3_evaluation": evaluation_output,
            "layer1_update": layer1_update_text
        })

    



if __name__ == "__main__":
    main()
