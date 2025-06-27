import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# Load the trained model and processor
original_model_id = "data/pretrained/SmolVLM-Instruct"
model_id = "data/mid/smolvlm-instruct-trl-sft-ChartQA"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Idefics3ForConditionalGeneration.from_pretrained(
    # model_id,
    original_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    # _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(original_model_id)


def generate_text_from_image(
    image_path, query, model, processor, max_new_tokens=1024, device="cuda"
):
    # Load and preprocess the image
    from PIL import Image

    image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert("RGB")
    # Transform to tensor

    # Prepare the input data
    sample = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": query,
                },
            ],
        },
    ]

    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],
        add_generation_prompt=True,  # Use the sample without the system message
    )

    image_inputs = [[image]]

    # Prepare the inputs for the model
    model_inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]  # Return the first decoded output text


# Example usage
image_path = "data/mid/doc/tikz_sample_line_graph.png"
query = "What is the value at the peak?"
output = generate_text_from_image(image_path, query, model, processor)
print(output)
