import os
import json
import random
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# 1. 初始化与配置
load_dotenv()
# 确保你的环境变量里有 OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 配置参数
INPUT_FILE = "train_target_total_5062.json"
OUTPUT_FILE = "safety_sft_500.jsonl"
TARGET_COUNT = 500
MAX_WORKERS = 10  # 并行线程数
MODEL = "gpt-4o-mini" # 推荐使用 gpt-4o-mini，性价比最高且遵循指令极强

# 2. 定义安全拒绝的多样化风格
REFUSAL_STYLES = [
    "Helpful and Educational: Explain why it's harmful and pivot to a legal way to learn about the topic.",
    "Professional and Concise: Directly state the refusal based on safety guidelines without being preachy.",
    "Ethical and Detailed: Discuss the potential real-world impact and risks associated with the request.",
    "Scenario-based: Treat the refusal as a professional boundary setting for a general-purpose AI."
]

# 3. 定义 JSON Schema (强制输出格式)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "safety_conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant"]},
                            "content": {"type": "string"}
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["messages"],
            "additionalProperties": False
        }
    }
}

# 4. 系统 Prompt
SYSTEM_PROMPT_TEMPLATE = """
You are an expert safety data annotator. Your task is to generate a natural, multi-turn conversation between a User and an AI Assistant.

CONTEXT:
The User has a harmful intent or 'goal'. You must simulate a user asking for this in a realistic way, and the Assistant responding with a high-quality safety refusal.

GOAL TO REFUSE: %GOAL%
REFUSAL STYLE: %STYLE%

REQUIREMENTS:
1. USER: Should ask for the goal, possibly in a subtle or curious way.
2. ASSISTANT: Must refuse the harmful request, explain the safety/ethical boundary, and if possible, provide a safe/legal alternative.
3. FORMAT: Use plain ASCII. No emojis.
4. TONE: The assistant should be helpful but firm on safety.
""".strip()

def generate_sample(goal_text):
    """调用 OpenAI API 生成单条数据"""
    style = random.choice(REFUSAL_STYLES)
    prompt = SYSTEM_PROMPT_TEMPLATE.replace("%GOAL%", goal_text).replace("%STYLE%", style)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=RESPONSE_FORMAT,
            temperature=0.8
        )
        # 解析返回的 JSON 字符串
        content = json.loads(response.choices[0].message.content)
        return content["messages"]
    except Exception as e:
        print(f"Error generating for goal: {goal_text[:30]}... | Error: {e}")
        return None

def main():
    # A. 读取 Red Teaming 数据
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # B. 随机采样 500 条
    print(f"Total goals loaded: {len(all_data)}")
    sampled_data = random.sample(all_data, min(TARGET_COUNT, len(all_data)))
    
    # C. 并行生成
    print(f"Starting generation of {len(sampled_data)} samples using {MODEL}...")
    
    results_count = 0
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将 goal 提取出来提交任务
        future_to_goal = {executor.submit(generate_sample, item["goal"]): item for item in sampled_data}
        
        for future in as_completed(future_to_goal):
            messages = future.result()
            if messages:
                # 写入 JSONL (CustomJSON 期望每行是一个 messages 数组)
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                
                results_count += 1
                if results_count % 10 == 0:
                    print(f"Progress: {results_count}/{len(sampled_data)} saved.")

    print(f"\nSuccessfully generated {results_count} safety samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()