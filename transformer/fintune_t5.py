import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Define the dataset
data = {
    'high_dimensional_data': ['bird', 'dog', 'human'],
    'labels': ['001', '010', '100']  # Class labels
}

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Load pre-trained T5 model and tokenizer
model_name = "google-t5/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define data processing function
def preprocess_function(examples):
    inputs = examples['high_dimensional_data']
    targets = examples['labels']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=16, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.01,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the model and tokenizer weight
save_dir = "data/pretrained/t5_finetuned/"
model.save_pretrained(save_dir)


