from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch 

with torch.no_grad():
    # model_path = "data/pretrained/t5_tokenizer_base.model"
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    print(input_ids.shape) 
    print(input_ids)
    labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    print(labels.shape)
    print(labels)

    loss = model(input_ids=input_ids, labels=labels).loss
    print(loss.item()) # 3.7837

    input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

    # the forward function automatically creates the correct decoder_input_ids
    loss = model(input_ids=input_ids, labels=labels).loss
    print(loss.item()) # 0.2542
