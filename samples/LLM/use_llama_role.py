from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="JaaackXD/Llama-3-8B-GGUF",
    local_dir="data/pretrained/llama/",
    filename="ggml-model-f16.gguf",
)

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response["choices"])
