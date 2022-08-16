import torch
import torch.nn as nn
import baseblock as B


class nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = B.conv(1, 2, 3, 1, 0, mode='CR')
        self.d = [B.conv(2, 2, 3, 1, 0, mode='CR')]
        self.add_module('m', self.d[0])

m = nn()

print(m.state_dict())
print(m.get_submodule('m'))

