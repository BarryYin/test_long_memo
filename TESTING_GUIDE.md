# 多步收敛优化 - 快速测试指南

## 🎯 本次优化目标

解决两个核心问题：
1. ✅ Memory 是否每轮更新？
2. ✅ 策略是否有多步收敛性？

## 📋 测试清单

### 1. Memory 更新测试

**测试步骤**：
1. 启动应用：`streamlit run app_easy.py`
2. 输入："我没钱"
3. 检查右侧 Memory 面板：
   - `ability_score` 应该变为 "zero"
   - `has_ability_confirmed` 应该变为 False
4. 输入："明天还 2000"
5. 检查 Memory：
   - `payment_date_confirmed` 应该显示 "2025-12-29"
   - `payment_amount_confirmed` 应该显示 "2000"
   - `payment_type_confirmed` 应该显示 "partial"

**预期结果**：每轮对话后，Memory 都会实时更新

---

### 2. 收敛性测试

#### 测试场景 A：逐步收敛

| 轮次 | 输入 | 预期 Memory 更新 | 预期 Agent 行为 |
|------|------|------------------|----------------|
| 1 | "我现在没钱" | ability=False | 追问：什么时候有钱？ |
| 2 | "明天有" | date="2025-12-29" | 追问：能还多少？ |
| 3 | "部分，1000块" | amount="1000", type="partial" | 追问：剩余怎么办？ |
| 4 | "能展期吗" | extension=True | 解释展期政策 |

#### 测试场景 B：快速收敛

| 轮次 | 输入 | 预期 Memory 更新 | 预期 Agent 行为 |
|------|------|------------------|----------------|
| 1 | "明天全额还款" | date="2025-12-29", type="full", ability=True | 锁定承诺，确认方式 |

#### 测试场景 C：多信息一次输入

| 轮次 | 输入 | 预期 Memory 更新 | 预期 Agent 行为 |
|------|------|------------------|----------------|
| 1 | "我30号能还2000，剩下的能展期吗" | date="2025-12-30", amount="2000", type="partial", extension=True | 确认所有信息，讨论展期 |

---

### 3. UI 可视化测试

**检查点**：
- [ ] `🎯 关键信息收敛进度` 模块是否显示
- [ ] 未确认项显示 `⏳` 图标
- [ ] 已确认项显示 `✅` 图标
- [ ] 展期请求显示 `⚠️` 图标（如果为 True）
- [ ] 日期格式正确（YYYY-MM-DD）
- [ ] 付款方式正确翻译（full → 全额，partial → 部分）

---

### 4. Layer1 策略输出测试

**检查点**：
- [ ] 策略中包含 `【多步收敛路径】` 部分
- [ ] 包含 Step1-5 的详细步骤
- [ ] 每步都有明确的追问目标
- [ ] 包含 ⚠️ 注意事项

**查看方法**：
1. 打开右侧 "📋 Daily Strategy (Layer 1)" 面板
2. 查看策略输出内容

---

### 5. Layer2 执行逻辑测试

**检查点**：
- [ ] Agent 根据 Memory 状态决定下一步问什么
- [ ] 一次只问一个问题（不会连续问 3 个问题）
- [ ] 已确认的信息不会重复追问
- [ ] 当所有信息收敛后，开始锁定承诺

**查看方法**：
1. 查看 "💬 Chat Interface" 中 Agent 的回复
2. 查看 "Layer 2 Executor Thinking" 中的 tactical_plan
3. 检查 tactical_plan 是否提到"哪个信息还未确认"

---

## 🐛 常见问题排查

### 问题 1：日期识别不准确

**症状**：输入"明天"，但 `payment_date_confirmed` 仍为空

**排查**：
1. 检查是否包含其他干扰词（如"明天再说"）
2. 检查 log 输出，看是否有 "Memory updated" 日志
3. 验证 datetime 导入正常

**修复**：调整正则表达式，增加容错

---

### 问题 2：金额提取失败

**症状**：输入"2000块"，但 `payment_amount_confirmed` 为空

**排查**：
1. 确认是否同时有 "部分" 关键词
2. 检查正则表达式 `r'(\d+)'` 是否匹配
3. 查看 log 输出

**修复**：独立金额提取逻辑，不依赖 "部分" 关键词

---

### 问题 3：Layer2 不追问未确认信息

**症状**：某些信息未确认，但 Agent 不主动追问

**排查**：
1. 检查 Memory Context 是否正确传递给 Layer2
2. 查看 Layer2 的 System Prompt 是否包含 "CONVERGENCE STRATEGY"
3. 检查 `get_memory_context()` 输出是否包含 "【关键信息收敛进度】"

**修复**：确认 Line 386-432 的 Layer2 Prompt 更新正确

---

## 📊 测试报告模板

```markdown
## 测试结果

**测试时间**：YYYY-MM-DD HH:MM

**测试场景**：[场景名称]

| 项目 | 预期 | 实际 | 状态 |
|------|------|------|------|
| Memory 更新 | 每轮更新 | ✅/❌ | Pass/Fail |
| 日期识别 | 正确解析 | ✅/❌ | Pass/Fail |
| 金额识别 | 正确提取 | ✅/❌ | Pass/Fail |
| 收敛路径显示 | 包含 Step1-5 | ✅/❌ | Pass/Fail |
| Layer2 智能追问 | 只问未确认项 | ✅/❌ | Pass/Fail |
| UI 进度显示 | 正确显示 ✅/⏳ | ✅/❌ | Pass/Fail |

**问题记录**：
- [如有问题，详细描述]

**建议改进**：
- [如有建议，详细描述]
```

---

## 🔧 快速修复命令

```bash
# 重启应用
pkill -f streamlit
streamlit run app_easy.py

# 查看最近 50 行日志
tail -f -n 50 ~/.streamlit/logs/streamlit.log

# 检查 Python 语法错误
python -m py_compile app_easy.py

# 清除 Streamlit 缓存
rm -rf ~/.streamlit/cache
```

---

**文档版本**：v1.0  
**创建时间**：2025-12-28  
**适用版本**：app_easy.py v2.1
