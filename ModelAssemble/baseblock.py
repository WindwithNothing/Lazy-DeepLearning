import numpy as np
import torch
import torch.nn as nn


# from itertools import pairwise


# Base block of ANN


def pairwise(iterable):
    return zip(iterable[:-1], iterable[1:])


def clear_cuda():
    torch.cuda.empty_cache()


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, inplace=True,
         mode='CR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=bias))
        elif t == 'L':
            L.append(nn.Linear(in_features=in_channels, out_features=out_channels, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.LeakyReLU(0.01, inplace=inplace))
        elif t == 'h':
            L.append(nn.Tanh())
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    if len(L) == 1:
        return L[0]
    else:
        return nn.Sequential(*L)



class ConvGenerator:
    """
    Factory method for creating.
    examples:
        ConvGenerator('D2RFR') to building a factory
        call it to generate a model sequence
        when use different channels factory will increase channels in first layer
    mode command:
        'D2'    -> down convolution by half size
        'U2'    -> up transposed convolution by two times kernel_size
        'C3'    -> convolution by (3,1,1) kernel
        'F''C1' -> full connect convolution use 1X1 kernel
        'R'     -> LeakyReLU
    """

    def __init__(self, mode=''):
        def find(m, _com):
            st = m[0]
            if st.find(_com) == 0:
                m[0] = st[len(_com):]
                return True
            else:
                return False

        mode = [mode]
        self.command_list = []
        while len(mode[0]) > 0:
            if find(mode, ' '):
                continue
            elif find(mode, 'D2'):
                # down 2
                dic = dict(kernel_size=3, stride=2, padding=1, bias=False, mode='C')
            elif find(mode, 'U2'):
                dic = dict(kernel_size=4, stride=2, padding=1, bias=False, mode='T')
            elif find(mode, 'C3'):
                dic = dict(kernel_size=3, stride=1, padding=1, bias=False, mode='C')
            elif find(mode, 'C1'):
                dic = dict(kernel_size=1, stride=1, padding=0, bias=False, mode='C')
            elif find(mode, 'F'):
                dic = dict(kernel_size=1, stride=1, padding=0, bias=False, mode='C')
            elif find(mode, 'R'):
                dic = dict(mode='R')
            else:
                raise ValueError('Unknown mode ' + f'{mode[0]}')
            self.command_list.append(dic)

    def __call__(self, in_channels, out_channels):
        L = []
        for i, com in enumerate(self.command_list):
            if i == 0:
                in_ch = in_channels
                out_ch = out_channels
            else:
                in_ch = out_channels
                out_ch = out_channels
            L.append(conv(in_channels=in_ch, out_channels=out_ch, **com))
        return nn.Sequential(*L)


class ResBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        res = self.model(x)
        return x + res


class UnetBlock(nn.Module):
    def __init__(self, in_channels=64, layers=3, bias=False):
        super().__init__()
        conv_set = dict(kernel_size=3, stride=2, padding=1)
        convt_set = dict(kernel_size=4, stride=2, padding=1)
        self.layers = layers
        c = 2 ** np.arange(self.layers + 1)
        self.channels = (np.full((self.layers + 1), in_channels) * c).tolist()
        self.down = []
        self.up = []
        for i, (ch1, ch2) in enumerate(pairwise(self.channels)):
            self.down.append(conv(ch1, ch2, **conv_set, bias=bias, mode='CR'))
            if ch2 == self.channels[-1]:
                self.up.append(conv(ch2, ch1, **convt_set, bias=bias, mode='TR'))
            else:
                self.up.append(conv(ch2 * 2, ch1, **convt_set, bias=bias, mode='TR'))
            self.add_module(f'down_lv{i}', self.down[-1])
            self.add_module(f'up_lv{i}', self.up[-1])
        self.up.reverse()
        self.decorder = conv(self.channels[0] * 2, self.channels[0], 1, 1, 0, bias, mode='CR')

    def forward(self, x):
        value = [x]
        for d in self.down:
            x = d(x)
            value.append(x)
        value = value[:-1]
        value.reverse()
        for u, v in zip(self.up, value):
            x = u(x)
            x = torch.cat((x, v), dim=1)
        return self.decorder(x)


class DownSequence(nn.Module):
    def __init__(self, in_channels: int, layer_channels: list, bias=False, through=True, method=None):
        """
        :param in_channels: input channels
        :param layer_channels: list include each layer of output channels
        :param bias:
        :param through: if false return shape like layer_channels, if ture return additionaly include input
        """
        super().__init__()
        if layer_channels is None:
            layer_channels = [64, 64, 64]
        if method is None:
            def method(ci, co):
                return conv(ci, co, 3, 2, 1, bias=bias, mode='CR')
        channels = [in_channels] + layer_channels
        self.through = through
        self.conv_layers = []
        for i in range(len(channels) - 1):
            lay = method(channels[i], channels[i + 1])
            # self.add_module(f'down_lv{i}', lay)
            self.conv_layers.append(lay)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        if self.through:
            out = [x]
        else:
            out = []
        for layer in self.conv_layers:
            x = layer(x)
            out.append(x)
        return out


class UpSequence(nn.Module):
    def __init__(self, in_channels: list, layer_channels: list, out_channels: int, bias=False,
                 reverse_input=True, method=None):
        """
        if use to construct a Unet, in_channels shoult be a reverse of DownSequence output
        :param in_channels: list of input channels
        :param layer_channels: inter channels, length must be equal to in_channels length - 1
        :param out_channels: int
        :param bias: bool
        :param reverse_input: default is true, convolution will begin in least tensor in input.
        """
        super(UpSequence, self).__init__()
        if layer_channels is None:
            layer_channels = [64] * (len(in_channels) - 1)
        if method is None:
            def method(ci, co):
                return conv(ci, co, 4, 2, 1, bias=bias, mode='TR')
        if len(layer_channels) != len(in_channels) - 1:
            raise ValueError('layer_channels length must be equal to in_channels length - 1')
        self.reverse_input = reverse_input
        layer_channels = [0] + layer_channels + [out_channels]
        self.conv_layers = []
        for i in range(len(in_channels)):
            layer = method(in_channels[i] + layer_channels[i], layer_channels[i + 1])
            # self.add_module(f'down_lv{len(in_channels) - i - 1}', layer)
            self.conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x_list):
        x_list = x_list.copy()
        if self.reverse_input:
            x_list.reverse()
        temp_y = None
        for x, layer in zip(x_list, self.conv_layers):
            if temp_y is not None:
                x = torch.cat([x, temp_y], dim=1)
            temp_y = layer(x)
        return temp_y


if __name__ == '__main__':
    x = torch.randn((1, 3, 32, 32))  # 28 14 7 <--2 4 8 16 32
    # convblock = conv(64,64, 3, 2, 3, mode='CR')
    # testblock = UnetBlock(64, layers=4)
    testblock = nn.Sequential(conv(3, 64, 3, 1, 1, mode='CR'),
                              DownSequence(64, layer_channels=[64, 128, 256], through=False,
                                           method=ConvGenerator('D2R FR')),
                              UpSequence([256, 128, 64], [128, 64], 64, method=ConvGenerator('U2R FR')),
                              conv(64, 3, 1, 1, 0, mode='CR')
                              )
    # print(testblock.channels, testblock.down, testblock.up, sep='\n')
    # print([y.shape for y in testblock(x)])
    print(testblock)
    print(testblock(x).shape)
