import os 
from openai import OpenAI
from .llm import openai_client

@openai_client(os.getenv("DEEPSEEK_ENDPOINT"), os.getenv("DEEPSEEK_API_KEY"))
def request_prompt(endpoint, api_key, prompt: str = "", model: str = "deepseek-r1-distill-qwen-32b") -> str:
    """
    Request a prompt from Aliyun DeepSeek API.

    Args:
        prompt (str): The input prompt to send to the model.
        model (str): The model to use for the request.

    Returns:
        str: The response from the model.
    """
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model=model,  # 此处以 deepseek-r1-distill-qwen-7b 为例，可按需更换模型名称。
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
            ]
    )

    return completion.choices[0].message.content

def run():
    """
    Run the request prompt function with a sample prompt.
    """
    prompt = "请问9.11和9.8哪个数字更大？"
    model = "deepseek-reasoner"
    response = request_prompt(prompt=prompt, model=model)
    print(f"Response: {response}")