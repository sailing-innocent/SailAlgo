import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image_path = "data/datasets/lerf/datasets/bouquet/images_8/frame_00001.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0)
print(image.shape)  # prints: torch.Size([1, 3, 224, 224])
text = tokenizer(["a diagram", "a dog", "a cat", "a bouquet of flowers"])
print(text.shape)  # prints: torch.Size([4, 77])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    print(image_features.shape)  # prints: torch.Size([1, 512])
    text_features = model.encode_text(text)
    print(text_features.shape)  # prints: torch.Size([4, 512])
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
