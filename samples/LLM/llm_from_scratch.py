# import urllib
import torch
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd

MASTER_CONFIG = {}


def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # split = 'train', 'val', 'test' = 0.8, 0.1, 0.1
    train = data[: int(0.8 * len(data))]
    val = data[int(0.8 * len(data)) : int(0.9 * len(data))]
    test = data[int(0.9 * len(data)) :]

    batch_data = train
    if split == "val":
        batch_data = val
    if split == "test":
        batch_data = test

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i : i + context_window] for i in ix]).long()
    y = torch.stack([batch_data[i + 1 : i + context_window + 1] for i in ix]).long()
    return x, y


# 构造一个评估函数
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    # 评估结果存储变量
    out = {}

    # 将模型置为评估模式
    model.eval()

    # 分别会在训练集和验证集里通过get_batchs()函数取评估数据
    for split in ["train", "val"]:
        losses = []

        # 评估10个batch
        for _ in range(10):
            # 拿到特征值（输入数据），以及目标值（输出数据）
            xb, yb = get_batches(
                dataset, split, config["batch_size"], config["context_window"]
            )

            # 把拿到的数据丢进模型，得到loss值
            _, loss = model(xb, yb)

            # 更新loss存储
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    model.train()

    return out


class SimpleBrokenModel(nn.Module):
    # init里的跟上面一样，没变化
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.linear = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.ReLU(),
            nn.Linear(config["d_model"], config["vocab_size"]),
        )

        # 添加前向传播函数

    def forward(self, idx, targets=None):
        # 实例化embedding层，输入映射为id的数据，输出嵌入后的数据
        x = self.embedding(idx)

        # 线性层承接embedding层输出的数据
        a = self.linear(x)

        # 对线性层输出的数据在最后一个维度，做softmax，得到概率分布
        logits = F.softmax(a, dim=-1)

        # 如果有目标值（也就是我们前面的y），则计算通过交叉熵损失计算loss结果。给输出的概率矩阵变个形状，再给目标值变个形状。  统一一下输入输出，然后计算loss。其中最后一维代表着一条数据。
        # 此处需要了解tensor.view()函数，带上几何空间想象力去想一下矩阵的形状。
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss

        # 如果没有目标值，则只返回概率分布的结果
        else:
            return logits

        # 查看参数量
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))


# 构建训练函数
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # loss存储
    losses = []

    # 训练时间记录开始时间
    start_time = time.time()

    # 循环训练指定epoch的轮数
    for epoch in range(config["epochs"]):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        optimizer.zero_grad()

        # 获取训练数据
        xs, ys = get_batches(
            dataset, "train", config["batch_size"], config["context_window"]
        )

        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改，比如学习率周期性变化，或者梯度减小，增加，具体策略需要综合考虑进行设置，详情自行查询，关键字：lr_scheduler
        if scheduler:
            scheduler.step()

        # 打印log
        if epoch % config["log_interval"] == 0:
            # 训练时间
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            x = evaluate_loss(model)

            # Store the validation loss
            losses += [x]

            # 打印进度日志
            if print_logs:
                print(
                    f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch) / config['log_interval']:.3f}"
                )

            # 重置开始时间，用于计算下一轮的训练时间
            start_time = time.time()

            # 打印下一轮的学习率，如果使用了lr_scheduler
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # 上面所有epoch训练结束，打印最终的结果
    print("Validation loss: ", losses[-1]["val"])

    # 返还每一步loss值的列表，因为我们要画图，返还的是loss迭代的图像
    return pd.DataFrame(losses).plot()


if __name__ == "__main__":
    # url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
    # file_name = "xiyouji.txt"
    # urllib.request.urlretrieve(url, file_name)
    logging.basicConfig(level=logging.INFO)
    lines = open("data/assets/text/xiyouji.txt", "r", encoding="utf-8").read()
    logging.info(f"文本长度: {len(lines)}")
    vocab = sorted(list(set(lines)))  # get all unique characters
    head_num = 50
    logging.info(f"词表前{head_num}个: {vocab[:head_num]}")
    logging.info(f"词表长度: {len(vocab)}")

    itos = {i: ch for i, ch in enumerate(vocab)}  # index to string
    stoi = {ch: i for i, ch in enumerate(vocab)}  # string to index

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    logging.info(encode("悟空"))  # [1318, 2691]
    logging.info(decode(encode("悟空")))

    dataset = torch.tensor(encode(lines), dtype=torch.int16)
    logging.info(dataset.shape)
    logging.info(dataset)

    MASTER_CONFIG.update(
        {
            "batch_size": 8,  # 不解释
            "context_window": 16,  # 滑动窗口采样，设置采样大小
            "vocab_size": 4325,  # 咱们的西游记数据集，一共包含4325个不重复的汉字，标点符号
        }
    )
    xs, ys = get_batches(
        dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
    )
    logging.info(xs.shape)
    logging.info(xs)
    logging.info(ys.shape)
    logging.info(ys)

    decoded_samples = [
        (decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))
    ]
    logging.info(decoded_samples)
    # 这里我们设置这个模型为128维的embedding
    MASTER_CONFIG.update(
        {
            "d_model": 128,
            "log_interval": 10,  # 每10个batch打印一次log
            "batch_size": 32,
        }
    )

    # 实例化模型，传参
    model = SimpleBrokenModel(MASTER_CONFIG)

    # 再看看参数量
    print("咱们的模型这么多参数量:", sum([m.numel() for m in model.parameters()]))
    # 于是乎，我们创建了一个1128307个参数的模型，上面参数想怎么改，自己改！电脑不会爆炸！
    logits, loss = model(xs, ys)
    print(loss)
    optimizer = torch.optim.Adam(
        model.parameters(),  # 优化器执行优化全部的模型参数
    )
    # 启动训练
    train(model, optimizer)
