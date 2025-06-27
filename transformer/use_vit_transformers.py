from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

model_path = "data/pretrained/vit/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTModel.from_pretrained(model_path)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape) # torch.Size([1, 197, 768])
