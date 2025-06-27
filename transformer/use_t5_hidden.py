from transformers import AutoTokenizer, T5EncoderModel
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5EncoderModel.from_pretrained("google-t5/t5-small")
input_ids = tokenizer(
        "Blue Apple", return_tensors="pt"
    ).input_ids  # Batch size 1
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # torch.Size([1, 15, 512])