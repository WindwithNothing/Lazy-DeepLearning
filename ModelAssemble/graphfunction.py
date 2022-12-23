import networkx as nx
import torch
import torch.nn as nn


def split_command(st):
    """
    Analysis layer command
    The command only describes one layer, such as a convolution layer or a action layer

    Example:
        >>> split_command('32c3s1p1')
        {'ch': 32, 'layer': 'c', 'c': 3, 's': 1, 'p': 1}
        >>> split_command('leakyrelu')
        {'layer': 'leakyrelu'}

    _: underline will be looked upon as symbol which will combine both sides number as a tuple
        >>> split_command('resize32_32')
        {'layer': 'resize', 'set': (32, 32), 'resize': (32, 32)}
    """
    if st == '':
        return {}
    out = []
    out_dict = {}
    # seperate digit and letter
    unit = ''
    for s in st + '#':
        if s is not '#' and s.isdigit() == unit.isdigit() or len(unit) == 0:
            unit += s
            continue
        else:
            if unit.isdigit():
                if '.' in unit:
                    unit = float(unit)
                else:
                    unit = int(unit)
            out.append(unit)
            unit = s
    # process underline
    for i, s in enumerate(out):
        if s == '_':
            left, right = out[i - 1], out[i + 1]
            if type(left) is tuple:
                new_tuple = left + (right,)
            else:
                new_tuple = (left, right)
            out[i + 1] = new_tuple
            out[i - 1] = '_'
    while '_' in out:
        out.remove('_')
    # analyze
    if len(out) == 1 and (type(out[0]) is int or type(out[0]) is tuple):
        # only one element and it is value
        out_dict['layer'] = 'ch'
        out_dict['ch'] = out[0]
    elif 'fc' in out:
        out_dict['layer'] = 'fc'
        if out.index('fc') == 0:
            out_dict['ch'] = out[1]
        elif out.index('fc') == 1:
            out_dict['ch'] = out[2]
            out_dict['in_ch'] = out[0]
        else:
            raise ValueError('can\'t analyze:{}'.format(out))
    elif type(out[0]) is int:
        out_dict['ch'] = out[0]
        if len(out) > 1:
            out_dict['layer'] = out[1]
            sub_out = out[1:]
            for i in range(len(sub_out) // 2):
                out_dict[sub_out[i * 2]] = sub_out[i * 2 + 1]
    else:
        out_dict['layer'] = out[0]
        if len(out) > 1:
            out_dict['set'] = out[1]
            sub_out = out
            for i in range(len(sub_out) // 2):
                out_dict[sub_out[i * 2]] = sub_out[i * 2 + 1]
    if 'reshape' in out:
        if isinstance(out_dict['set'], int):
            out_dict['ch'] = out_dict['set']
        elif isinstance(out_dict['set'], tuple):
            out_dict['ch'] = out_dict['set'][0]
    return out_dict


def expression_parsing(_str):
    """
    Analysis layer expression.
    Layer expression is a formula that describe a sequence of layers.
    This function transform the expression into a sequence of layer command separate by blank.

    Examples:
        >>> expression_parsing('(32c3s1p1+leakyrelu) * 3 + 4c3s1p1+leakyrelu')
        '32c3s1p1 leakyrelu 32c3s1p1 leakyrelu 32c3s1p1 leakyrelu 4c3s1p1 leakyrelu'
    """
    stack = []
    stack_out = []
    while _str.find(' ' * 2) >= 0:
        _str = _str.replace(' ' * 2, ' ')
    _str = _str.strip()
    priority = {'(': 1, ')': 2, '*': 3, '+': 4}
    start = 0
    for i, s in enumerate(_str):
        if s in priority:
            end = i
            unit_str = _str[start: end]
            start = i + 1
            stack_out.append(unit_str)
            if s == '(':
                stack.append(s)
            elif s == ')':
                while stack[-1] != '(':
                    stack_out.append(stack.pop())
                stack_out.append(stack.pop())
            elif s in '+*':
                if stack == [] or priority[s] < priority[stack[-1]]:
                    stack.append(s)
                else:
                    if stack[-1] not in '()':
                        stack_out.append(stack.pop())
                    stack.append(s)
    stack_out.append(_str[start:])
    while stack:
        stack_out.append(stack.pop())
    while stack_out.count(''):
        stack_out.remove('')
    while stack_out.count(' '):
        stack_out.remove(' ')

    stack = []
    for s in stack_out:
        s = s.replace(' ', '')
        if s == '(':
            continue
        if s == '':
            continue
        if s == '+':
            a1 = stack.pop()
            a2 = stack.pop()
            stack.append(a2 + ' ' + a1)
        elif s == '*':
            a1 = stack.pop()
            a2 = stack.pop()
            if a1.isdigit():
                stack.append((a2 + ' ') * int(a1))
            else:
                stack.append((a1 + ' ') * int(a2))
        else:
            stack.append(s)
    return ' '.join(stack[0].split())


def generate_layer_sequence(_list, in_ch=None):
    if _list[0]['layer'] == 'ch':
        in_ch = _list[0]['ch']
        _list = _list[1:]
    block_list = []
    #print(in_ch, _list)
    for com in _list:
        if com['layer'] == 'c':
            block_list.append(nn.Conv2d(in_ch, com['ch'], kernel_size=com['c'],
                                        stride=com['s'], padding=com['p']))
        elif com['layer'] == 't':
            block_list.append(nn.ConvTranspose2d(in_ch, com['ch'], kernel_size=com['t'],
                                                 stride=com['s'], padding=com['p']))
        elif com['layer'] == 'fc':
            # block_list.append(nn.LazyLinear(out_features=com['ch']))
            block_list.append(nn.Linear(in_ch, com['ch']))
        elif com['layer'] == 'relu':
            block_list.append(nn.ReLU(inplace=True))
        elif com['layer'] == 'tanh':
            block_list.append(nn.Tanh())
        elif com['layer'] == 'sigmoid':
            block_list.append(nn.Sigmoid())
        elif com['layer'] in ['leakyrelu', 'r']:
            if len(com) > 1:
                k = float(com[1])
            else:
                k = 0.01
            block_list.append(nn.LeakyReLU(k, inplace=True))
        elif com['layer'] == 'maxpool':
            block_list.append(nn.MaxPool2d(kernel_size=com['set'], stride=com['s'], padding=com['p']))
        elif com['layer'] == 'a_avgpool':
            block_list.append(nn.AdaptiveAvgPool2d(output_size=com['set']))
        elif com['layer'] == 'a_maxpool':
            block_list.append(nn.AdaptiveMaxPool2d(output_size=com['set']))
        elif com['layer'] == 'reshape':
            block_list.append(ReshapeLayer(com['set']))
        else:
            raise ValueError("Unknown layer {}".format(com))
        if 'ch' in com:
            in_ch = com['ch']
    return nn.Sequential(*block_list)


class Command:
    """
    A class that hold describe string
    """

    def __init__(self, _str):
        self._str = _str
        self.input_type = None  # res cat None
        # input shape
        self.input_shape = None
        # long string describe
        self.layers_command = None
        # list of describe dict
        self.layers_command_list = None
        # nn.Module Sequential
        self.layers = None
        # out channels shape
        self.out_ch = None
        # out layer type: conv or fc
        self.out_layer_type = None
        self.analysis_describe(self._str)

    def analysis_describe(self, _str):
        # first analyze the command, seperate by ','
        com = _str.split(',')
        for c in com[:-1]:
            c = split_command(c)
            if c['layer'] == 'ch':
                self.input_shape = c['ch']
            elif c['layer'] == 'res':
                self.input_type = 'res'
        self.layers_command = expression_parsing(com[-1])
        self.layers_command_list = [split_command(com) for com in self.layers_command.split()]
        for com in self.layers_command_list[::-1]:
            if 'ch' in com:
                self.out_ch = com['ch']
                if com['layer'] == 'fc':
                    self.out_layer_type = 'fc'
                else:
                    self.out_layer_type = 'conv'
                break
        #if self.input_shape is None and 'ch' in self.layers_command_list[0]:
        #    self.input_shape = self.layers_command_list[0]['ch']

    def set_input_shape(self, shape: list):
        if self.input_shape is None:
            if len(shape) == 0:
                pass
            elif len(shape) > 1 and self.input_type is None:
                self.input_type = 'cat'
                self.input_shape = sum(shape)
            elif len(shape) > 1 and self.input_type == 'res':
                self.input_shape = shape[0]
            elif self.input_type is None:
                self.input_shape = shape[0]
            else:
                raise ValueError('Unknown input, self.input_type is {}, but get input shape is {}'.format(
                    self.input_type, shape
                ))
        else:
            raise ValueError("node already has input shape:{}, given shape is:{}".format(
                self.input_shape, shape))

    def generate_layers(self):
        return generate_layer_sequence(self.layers_command_list, self.input_shape)

    def __str__(self):
        return 'input_shape:{}, layers_command:{}, output_shape{}'.format(
            self.input_shape, self.layers_command, self.out_ch)


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        if type(self.shape) is int:
            self.shape = [self.shape]

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return 'shape={}'.format(
            self.shape
        )

'''
#
# ___  _    ____ ____ _  _    ____ ____ _  _ _  _ ____ _  _ ___
# |__] |    |  | |    |_/     |    |  | |\/| |\/| |__| |\ | |  \
# |__] |___ |__| |___ | \_    |___ |__| |  | |  | |  | | \| |__/
#
#


def block_command(command_str, out_ch=False, return_info=False):
    def split_command(st):
        if st == '':
            return {}
        out = []
        out_dict = {}
        # seperate digit and letter
        unit = ''
        for s in st:
            if s.isdigit() == unit.isdigit() or len(unit) == 0:
                unit += s
                continue
            else:
                out.append(unit)
                unit = s
        out.append(unit)
        # analyze
        if 'fc' in out:
            out_dict['layer'] = 'fc'
            if out.index('fc') == 0:
                out_dict['ch'] = int(out[1])
            elif out.index('fc') == 1:
                out_dict['ch'] = int(out[2])
                out_dict['in_ch'] = int(out[0])
            else:
                raise ValueError('can\'t analyze:{}'.format(out))
        elif out[0].isdigit():
            out_dict['ch'] = int(out[0])
            if len(out) > 1:
                out_dict['layer'] = out[1]
                sub_out = out[1:]
                for i in range(len(sub_out) // 2):
                    out_dict[sub_out[i * 2]] = int(sub_out[i * 2 + 1])
        else:
            out_dict['layer'] = out[0]
            if len(out) > 1:
                out_dict['set'] = float(out[1])
                sub_out = out
                for i in range(len(sub_out) // 2):
                    out_dict[sub_out[i * 2]] = int(sub_out[i * 2 + 1])
        return out_dict

    # 字符串拆分为无运算符格式?"
    _str = command_str
    stack = []
    stack_out = []
    priority = {'(': 1, ')': 2, '*': 3, '+': 4, '': 5}
    start = 0
    for i, s in enumerate(_str):
        if s in '()*+ ':
            end = i
            unit_str = _str[start: end]
            stack_out.append(unit_str)
            if s == '(':
                start = i + 1
                stack.append(s)
            elif s == ')':
                start = i + 1
                while stack[-1] != '(':
                    stack_out.append(stack.pop())
                stack_out.append(stack.pop())
            elif s in '+*':
                if stack == [] or priority[s] < priority[stack[-1]]:
                    stack.append(s)
                else:
                    if stack[-1] not in '()':
                        stack_out.append(stack.pop())
                    stack.append(s)
                start = i + 1
    stack_out.append(_str[start:])
    while stack:
        stack_out.append(stack.pop())

    stack = []
    for s in stack_out:
        s = s.replace(' ', '')
        if s == '(':
            continue
        if s == '':
            continue
        if s == '+':
            a1 = stack.pop()
            a2 = stack.pop()
            stack.append(a2 + ' ' + a1)
        elif s == '*':
            a1 = stack.pop()
            a2 = stack.pop()
            if a1.isdigit():
                stack.append((a2 + ' ') * int(a1))
            else:
                stack.append((a1 + ' ') * int(a2))
        else:
            stack.append(s)
    _str = stack[0]

    # 生成神经网络Sequential
    # 例子'32 64c3s1p1 relu 64c3s1p1 tanh'
    while _str.find('  ') >= 0:
        _str = _str.replace('  ', ' ')
    _str = _str.strip()
    command_list = [split_command(st) for st in _str.lower().split(' ')]

    # only return-channel check point
    info = {}
    for com in command_list[::-1]:
        if 'ch' in com:
            info['out_ch'] = com['ch']
            if com['layer'] == 'fc':
                info['out_type'] = 'fc'
            else:
                info['out_type'] = 'conv'
    for com in command_list:
        if 'ch' in com:
            info['in_ch'] = com['ch']
    if out_ch:
        return info['out_ch']
        # raise ValueError('can\' find channel info in ', command_str)
    if return_info:
        return info

    in_ch = command_list[0]['ch']
    block_list = []
    for com in command_list[1:]:
        if com['layer'] == 'c':
            block_list.append(nn.Conv2d(in_ch, com['ch'], kernel_size=com['c'],
                                        stride=com['s'], padding=com['p']))
        elif com['layer'] == 't':
            block_list.append(nn.ConvTranspose2d(in_ch, com['ch'], kernel_size=com['t'],
                                                 stride=com['s'], padding=com['p']))
        elif com['layer'] == 'fc':
            # block_list.append(nn.LazyLinear(out_features=com['ch']))
            block_list.append(nn.Linear(in_ch, com['ch']))
        elif com['layer'] == 'relu':
            block_list.append(nn.ReLU(inplace=True))
        elif com['layer'] == 'tanh':
            block_list.append(nn.Tanh())
        elif com['layer'] in ['leakyrelu', 'r']:
            if len(com) > 1:
                k = float(com[1])
            else:
                k = 0.01
            block_list.append(nn.LeakyReLU(k, inplace=True))
        elif com['layer'] == 'maxpool':
            block_list.append(nn.MaxPool2d(kernel_size=com['set'], stride=com['s'], padding=com['p']))
        if 'ch' in com:
            in_ch = com['ch']
    return nn.Sequential(*block_list)
'''

test_str_experssion = """(32c3s1p1+leakyrelu) * 3 + 4c3s1p1+leakyrelu"""

if __name__ == '__main__':
    print(split_command('reshape1024'))
