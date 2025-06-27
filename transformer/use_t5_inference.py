from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(outputs.shape) # 1, 7
print(outputs[0])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
little_mod = outputs[0].clone()
little_mod[1] = little_mod[1] + 1

print(tokenizer.decode(little_mod, skip_special_tokens=True))