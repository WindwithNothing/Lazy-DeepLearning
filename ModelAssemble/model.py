import torch
import torch.nn as nn
#import sys
#sys.path.append("..")
#import Playground.Model.baseblock as B
import Model.baseblock as B

# model 模块的进一步集成包

class FallBlock(nn.Module):
    def __init__(self, features=64, layers: int = 4, batch_norm=False, core_layer_func=None):
        super().__init__()
        activate_method = 'R'
        if batch_norm:
            activate_method = 'B' + activate_method

        # self.encoder = B.conv(3, 64, 1, 1, 0, mode='CR')
        # self.up = B.conv(3, 64, 2, 2, 0, mode='TR')
        def interlayer(fec):
            return nn.Sequential(
                B.conv(fec, fec, 3, 1, 1, mode='C'+activate_method),
                B.conv(fec, fec, 1, 1, 0, mode='C'+activate_method))

        if core_layer_func:
            interlayer = core_layer_func
        for i in range(layers):
            self.add_module(f'p{i}', interlayer(features))
        self.process = [self.get_submodule(f'p{i}') for i in range(layers)]
        self.decoder = B.conv(features * (layers + 1), features, 1, 1, 0, mode='CR')

    def forward(self, x):
        # x = self.encoder(x)
        value = [x]
        site = x
        for p in self.process:
            site = p(site)
            value.append(site)
        x = self.decoder(torch.cat(value, dim=1))
        return x


class UnetBlock(nn.Module):
    def __init__(self, features=64, layers: int = 4, batch_norm=False):
        super().__init__()
        activate_method = 'R'
        if batch_norm:
            activate_method = 'B'+activate_method

        def down():
            return B.conv(features, features, 3, 2, 1, mode='C'+activate_method)

        def up():
            return B.conv(features*2, features, 4, 2, 1, mode='T'+activate_method)

        self.downlayers = [down() for _ in range(layers)]
        self.centerlayer = B.conv(features, features, 1, 1, 0, mode='C'+activate_method)
        self.uplayers = [up() for _ in range(layers)]
        for i, l in enumerate(self.downlayers):
            self.add_module(f'down{i}', l)
        for i, l in enumerate(self.uplayers):
            self.add_module(f'up{i}', l)

    def forward(self, x):
        value = []
        for down in self.downlayers:
            x = down(x)
            value.append(x)
        x = self.centerlayer(x)
        value.reverse()
        for up,v in zip(self.uplayers, value):
            x = torch.cat([x, v], dim=1)
            x = up(x)
        return x


class FallSRNN(nn.Module):
    def __init__(self, input_features=3, features=64, layers: int = 4, batch_norm=False):
        super().__init__()
        self.name = f'FallSRNN_io{input_features}_ft{features}_l{layers}'
        if batch_norm: self.name += '_bn'

        self.up = B.conv(input_features, features, 2, 2, 0, mode='TR')
        self.fall = FallBlock(features, layers, batch_norm=batch_norm)
        self.decoder = B.conv(features, input_features, 1, 1, 0, mode='CR')

    def forward(self, x):
        x = self.up(x)
        x = self.fall(x)
        return self.decoder(x)


class UnetSRNN(nn.Module):
    def __init__(self, input_features=3, features=64, layers: int = 4, batch_norm=False):
        super().__init__()
        self.name = f'UnetSRNN_io{input_features}_ft{features}_l{layers}'
        if batch_norm: self.name += '_bn'

        self.up = B.conv(input_features, features, 2, 2, 0, mode='TR')
        self.unet = UnetBlock(features=features, layers=layers, batch_norm=batch_norm)
        self.decoder = B.conv(features, input_features, 1, 1, 0, mode='CR')

    def forward(self, x):
        x = self.up(x)
        x = self.unet(x)
        return self.decoder(x)



if __name__ == '__main__':
    model = FallSRNN(layers=2, batch_norm=True)
    x = torch.randn(10, 3, 64, 64)
    print(model(x).shape)
    print(model.name, sum(p.nelement() for p in model.parameters()))
