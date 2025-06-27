# -*- coding: utf-8 -*-
# @file use_m3e_embedding.py
# @brief M3E Embedding Model Demo
# @author sailing-innocent
# @date 2025-02-16
# @version 1.0
# ---------------------------------

from sentence_transformers import SentenceTransformer

model_path = "data/pretrained/embedding/m3e-base"
# model_path = "moka-ai/m3e-base"
model = SentenceTransformer(model_path)

# Our sentences we like to encode
sentences = [
    "* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem",
    "* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练",
    "* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one",
]

# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 768)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
