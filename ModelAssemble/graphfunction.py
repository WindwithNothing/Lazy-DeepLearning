import networkx as nx
import torch
import torch.nn as nn


def split_command(st):
    """
    Analysis layer command
    The command only describes one layer, such as a convolution layer or a action layer

    Example:
        32c3s1p1 -> {'ch': 32, 'layer': 'c', 'c': 3, 's': 1, 'p': 1}
        leakyrelu -> {'layer': 'leakyrelu'}

    """
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
    if len(out) == 1 and out[0].isdigit():
        out_dict['layer'] = 'ch'
        out_dict['ch'] = int(out[0])
    elif 'fc' in out:
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
    print(_str)
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
        #print(s, stack)
    return ' '.join(stack[0].split())


class Command:
    def __init__(self, _str):
        self._str = _str


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


test_str_experssion = """ (32c3s1p1+leakyrelu) * 3 + 4c3s1p1+leakyrelu"""

if __name__ == '__main__':
    print(split_command('leakyrelu'))
