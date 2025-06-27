from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_name = "data/pretrained/qwen/Qwen2-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(model_name)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There is a dog and"},
        ],
    },
]
