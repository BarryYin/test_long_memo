#!/usr/bin/env python3
import asyncio
from openai import AsyncOpenAI

# 百度ERNIE模型的配置
BAIDU_API_KEY = "bce-v3/ALTAK-IlAGWrpPIFAMJ3g8kbD4I/f17c0a909b891c89b0dce53d913448d86a87bad9"
BAIDU_BASE_URL = "https://qianfan.baidubce.com/v2"

# 要测试的ERNIE模型列表
MODELS_TO_TEST = [
    "ernie-4.5-turbo-32k",
    "ernie-4.5",
    "ernie-3.5",
    "ernie-turbo",
    "ernie-bot-32k",
    "ernie-bot"
]

async def test_models():
    """测试多个百度ERNIE模型是否可用"""

    for model in MODELS_TO_TEST:
        try:
            print(f"\n🧪 测试模型: {model}")
            client = AsyncOpenAI(api_key=BAIDU_API_KEY, base_url=BAIDU_BASE_URL)

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你好"}],
                temperature=0.7
            )

            print(f"✅ 模型 {model} 可用!")
            print(f"📝 响应预览: {response.choices[0].message.content[:50]}...")

            # 找到第一个可用模型
            print(f"\n🎉 推荐使用模型: {model}")
            return model

        except Exception as e:
            error_str = str(e)
            if "invalid_model" in error_str:
                print(f"❌ 模型 {model} 不存在或无访问权限")
            elif "401" in error_str:
                print(f"🔑 认证失败 - 需要检查API Key")
            else:
                print(f"⚠️  测试失败: {error_str}")

    print("\n⚠️ 没有找到可用的ERNIE模型！")
    return None

if __name__ == "__main__":
    print("🎯 正在测试百度ERNIE模型的可用性...")
    _, _, _ = asyncio.run(test_models())