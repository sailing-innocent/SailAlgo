import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float = 0.5,
        dropout: float = 0.1,
    ):
        # weight from original linear layer
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(p=dropout)

        self.linear.weight.requires_grad = False  # freeze the original linear layer
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original linear layer
        fronzen_out = self.linear(x)

        # LoRA
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        return fronzen_out + (self.alpha / self.rank) * lora_out
