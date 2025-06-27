# import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import json 
import matplotlib.pyplot as plt


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_path = "microsoft/Florence-2-large"
model_path = "data/pretrained/vision/Florence-2-large"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

prompt = "<OD>"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# res = requests.get(url, stream=True).raw
res = "data/test/test.JPG"
image = Image.open(res)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)


od = parsed_answer["<OD>"]
for bbox, label in zip(od["bboxes"], od["labels"]):
    print(f"{label}: {bbox}")
    # visualizing bounding boxes
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor="red", lw=2))
    plt.text(bbox[0], bbox[1], label, color="red")
plt.show()
    
