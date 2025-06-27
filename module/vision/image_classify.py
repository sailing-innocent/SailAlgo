# A Simple MLP classifyer for MNIST dataset

import torch
from cent.dataset.mnist import MNISTConfig, MNIST
from cent.dataset.fmnist import FMNISTConfig, FMNIST
from cent.dataset.cifar10 import CIFAR10Config, CIFAR10
from torch.utils.tensorboard import SummaryWriter
from cent.model.lenet5 import LeNet5
from cent.model.resnet import ResNet, BasicBlock 
from cent.model.mlp import MLP
from vit_pytorch import ViT
import argparse
import os 
import json 

def _transform_x_identity(x):
    return x

def _transform_x_flatten(x):
    return x.view(x.shape[0], -1)

# use pytorch resnet18
def create_model_lenet(sample_shape, num_classes=10):
    img_channels = sample_shape[0]
    H, W = sample_shape[1:]
    return LeNet5(img_channels, num_classes, W, H).to(device)

def create_model_resnet(sample_shape, num_classes=10):
    img_channels = sample_shape[0]
    return ResNet(img_channels=img_channels, num_layers=18, block=BasicBlock, num_classes=num_classes).to(device)

def create_model_mlp(sample_shape, num_classes = 10):
    hidden_channels = [128, 64]
    input_channels = 1
    for i in sample_shape:
        input_channels = input_channels * i
    output_channels = num_classes
    
    return MLP(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        active_fn="relu"
    )

def create_model_vit(sample_shape, num_classes=10):
    img_channels = sample_shape[0]
    H, W = sample_shape[1:]
    return ViT(
        image_size = (H, W),
        patch_size = 4,
        num_classes = num_classes,
        dim = 16,
        depth = 3,
        heads = 4,
        mlp_dim = 128,
        dropout = 0.1,
        emb_dropout = 0.1
    )

if __name__ == "__main__":
    # arg --test for test only
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="lenet")
    parser.add_argument("--dataset_root", type=str, default="data/datasets/")
    parser.add_argument("--result_path", type=str, default="data/result/exp/")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    name = "_".join([args.model ,args.dataset, str(args.epochs)])

    transform_x_dict = {
        "lenet": _transform_x_identity,
        "resnet": _transform_x_identity,
        "mlp": _transform_x_flatten,
        "vit": _transform_x_identity
    }

    _transform_x = transform_x_dict[args.model]
    create_model_dict = {
        "lenet": create_model_lenet,
        "resnet": create_model_resnet,
        "mlp": create_model_mlp,
        "vit": create_model_vit
    }
    create_model = create_model_dict[args.model]

    dataset_config_dict = {
        "mnist": MNISTConfig,
        "cifar10": CIFAR10Config,
        "fmnist": FMNISTConfig
    }
    dataset_dict = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "fmnist": FMNIST
    }

    assert args.dataset in dataset_dict, f"Dataset {args.dataset} not found"
    dataset_func = dataset_dict[args.dataset]

    config = dataset_config_dict[args.dataset](args.dataset_root)
    if not args.test:
        train_dataset = dataset_func(config)

    config.usage = "test"
    test_dataset = dataset_func(config)

    sample = test_dataset[0]

    os.makedirs(args.result_path, exist_ok=True)
    result_path = f"{args.result_path}/{name}.pth"
    # set summary writer
    log_path = f"{args.result_path}/log"
    result_json = f"{args.result_path}/result.json"
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(f"{log_path}/{name}")

    num_classes = len(test_dataset.classes())
    model = create_model(sample[0].shape, num_classes)
    model.to(device)

    if not args.test:
        # train
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(args.epochs):
            for idx, batch in enumerate(train_dataset):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                # convert labels to one-hot
                labels = torch.nn.functional.one_hot(labels, num_classes).float()
                optimizer.zero_grad()
                output = model(_transform_x(images)) # [batch_size, N_classes]
                loss = criterion(output, labels)
                loss.backward()
                with torch.no_grad():
                    optimizer.step()
                    if idx % 100 == 0:
                        print(f"Epoch: {epoch}, Loss: {loss.item()}")     
            writer.add_scalar("Loss/train", loss, epoch)

        print("Training finished")
        writer.flush()
        # save path
        torch.save(model.state_dict(), result_path)
    
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(result_path, map_location=device, weights_only=True))
    # test
    model.eval()
    correct = 0
    total = 0
    for idx, batch in enumerate(test_dataset):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        output = model(_transform_x(images))
        pred = torch.argmax(output, dim=1)
        correct += torch.sum(pred == labels).item()
        total += labels.shape[0]
    
    print(f"Accuracy: {correct / total}")
    print(f"Correct: {correct}, Total: {total}")
    print("Test finished")

    if not os.path.exists(result_json):
        with open(result_json, "w") as f:
            json.dump({}, f)

    with open(result_json, "r") as f:
        result_dict = json.load(f)
        result_dict[name] = correct / total
    with open(result_json, "w") as f:
        json.dump(result_dict, f)
