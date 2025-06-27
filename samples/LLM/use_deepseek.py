from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "data/pretrained/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
# check the model dtype
print(model.dtype)

tokenizer = AutoTokenizer.from_pretrained(model_name)
with torch.no_grad():
    # prompt = "给我写一段例程调用QwenVL 7B模型来识别理解test_img.png中的内容"
    # prompt = "写一段C++代码，读取用户输入的两个数字，比较大小并输出比较结果"
    # prompt = "案件逐渐发酵，变得松软可口"
    # prompt = "我脑子不好了怎么办？"
    prompt = "被别人当枪使如何应对？"
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(model_inputs)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
