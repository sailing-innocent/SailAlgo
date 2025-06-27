from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "data/pretrained/qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"  # 3B int8 速度倒是不错，内容也还可以
# model_name = "data/pretrained/qwen/Qwen2.5-14B-Instruct" # 全精度几乎完全跑不动，删了
# model_name = "data/pretrained/qwen/Qwen2.5-14B-Instruct-GPTQ-Int4" # 3B int4
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "给我介绍一下大语言模型"
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
