import torch.nn as nn
import torch

torch.manual_seed(1)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=1.0)


class Policy(nn.Module):
    def __init__(self, in_size: int, hidden_size: float, out_size: int):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, out_size, bias=False)
        )
        # self.net.apply(init_weights)

    def forward(self, inp):
        return self.net(inp)
