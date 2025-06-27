from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the finetuned model and tokenizer
model_name = "google-t5/t5-small"
save_dir = "data/pretrained/t5_finetuned/"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(save_dir)

# Function to get low-dimensional vector
def get_low_dim_vector(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    low_dim_vector = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return low_dim_vector

# Example usage
low_dim_vector_bird = get_low_dim_vector("bird")
low_dim_vector_dog = get_low_dim_vector("dog")
low_dim_vector_brown_dog = get_low_dim_vector("brown dog")
low_dim_vector_human = get_low_dim_vector("human")
low_dim_vector_women = get_low_dim_vector("women")
low_dim_vector_compex_human = get_low_dim_vector("a blue eyed student")
low_dim_vector_flying_bird = get_low_dim_vector("flying bird")

print("Low-dimensional vector for 'bird':", low_dim_vector_bird)
print("Low-dimensional vector for 'dog':", low_dim_vector_dog)
print("Low-dimensional vector for 'brown dog':", low_dim_vector_brown_dog)
print("Low-dimensional vector for 'human':", low_dim_vector_human)
print("Low-dimensional vector for 'women':", low_dim_vector_women)
print("Low-dimensional vector for 'a blue eyed student':", low_dim_vector_compex_human)
print("Low-dimensional vector for 'flying bird':", low_dim_vector_flying_bird)

print("---------------------------------")
print("Original model")
print("---------------------------------")
original_model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to get high-dimensional vector
def get_low_dim_vector_original(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = original_model.generate(input_ids)
    low_dim_vector = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return low_dim_vector

# Example usage
low_dim_vector_bird = get_low_dim_vector_original("bird")

low_dim_vector_dog = get_low_dim_vector_original("dog")
low_dim_vector_brown_dog = get_low_dim_vector_original("brown dog")
low_dim_vector_human = get_low_dim_vector_original("human")
low_dim_vector_women = get_low_dim_vector_original("women")

print("Low-dimensional vector for 'bird':", low_dim_vector_bird)
print("Low-dimensional vector for 'dog':", low_dim_vector_dog)
print("Low-dimensional vector for 'brown dog':", low_dim_vector_brown_dog)
print("Low-dimensional vector for 'human':", low_dim_vector_human)
print("Low-dimensional vector for 'women':", low_dim_vector_women)