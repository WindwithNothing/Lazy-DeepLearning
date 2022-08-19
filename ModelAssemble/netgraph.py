import networkx as nx
import torch.nn as nn
import torch


def str_analysis(_str):
    """
    processing a passage describe a neural network
    Args:
        _str: a

    Returns: net graph with node have block describe
    """
    lines = _str.splitlines()
    net_defines_line = []
    net_relations_line = []
    graph = nx.DiGraph()
    # pre-process
    # separate lines into define lines and relation lines
    for line in lines:
        line = line.split('#')[0]
        if ':' in line:
            net_defines_line.append(line)
        else:
            net_relations_line.append(line)
    # ============
    # net_defines
    # block定义段
    # ============
    net_defines_dict = {}
    # 提取定义命令
    for line in net_defines_line:
        ls = line.split(':')
        if len(ls) == 2:
            pass
        else:
            raise ValueError("无法处理定义 \"" + line + '\"')
        sub_l = [' '.join(ll.split()) for ll in ls]
        net_defines_dict[sub_l[0]] = sub_l[1]
    # 参考定义赋值
    for key, value in list(net_defines_dict.items()):
        index = value.replace('(', '#')
        index = index.replace(')', '#')
        index = index.replace('+', '#')
        index = index.replace('*', '#')
        index = index.split('#')
        for dex in index:
            if dex in net_defines_dict:
                value = value.replace(dex, '(' + net_defines_dict[dex] + ')')
        if value.replace(' ', '').find('(') == 0 and \
                value.replace(' ', '').rfind(')') == len(value.replace(' ', '')) - 1:
            value = value[value.find('(') + 1:value.rfind(')')]
        net_defines_dict[key] = value
        # if value in net_defines_dict:
        #    net_defines_dict[key] = net_defines_dict[value]
    # 创建
    graph.add_nodes_from([(name, {'describe': desc}) for name, desc in net_defines_dict.items()])
    # ============
    # net_relations
    # 网络连接段
    # ============
    for line in net_relations_line:
        line = line.replace(' ', '')
        line = line.split('>')
        for b1, b2 in zip(line[:-1], line[1:]):
            graph.add_edge(b1, b2)
    for node in list(nx.isolates(graph)):
        #  删除独立节点
        graph.remove_node(node)
    # type defines
    for node in graph.nodes:
        if node.find('input') == 0:
            graph.nodes[node]['type'] = 'input'
        elif node.find('output') == 0:
            graph.nodes[node]['type'] = 'output'
        else:
            if graph.nodes[node]['describe'].split()[0] == 'res':
                graph.nodes[node]['type'] = 'res_layer'
                graph.nodes[node]['describe'] = \
                    graph.nodes[node]['describe'][
                    graph.nodes[node]['describe'].find('res') + 3:]
            else:
                graph.nodes[node]['type'] = 'layer'
    return graph


def graph_order(graph, begin, end):
    """
    generate compute order in graph from begin node to end node

    Args:
        graph:
        begin:
        end:

    Returns:

    """
    flag = {begin: True}
    queue = [begin]
    out = []
    while queue:
        node = queue[0]
        if all([(n in flag) for n in graph.pred[node]]):
            flag[node] = True
            queue = queue + [n for n in graph.succ[node].keys()]
            out.append(node)
            if node == end:
                break
        queue = queue[1:]
    return out


#       _   _      _    _____                 _
#      | \ | |    | |  / ____|               | |
#      |  \| | ___| |_| |  __ _ __ __ _ _ __ | |__
#      | . ` |/ _ \ __| | |_ | '__/ _` | '_ \| '_ \
#      | |\  |  __/ |_| |__| | | | (_| | |_) | | | |
#      |_| \_|\___|\__|\_____|_|  \__,_| .__/|_| |_|
#                                      | |
#                                      |_|


class NetGraph(nn.Module):
    def __init__(self, string, info=None):
        super().__init__()
        self.string = string
        self.graph = str_analysis(string)
        self.add_ch()
        self.order = self.block_order()
        self.model_dict = {str(node): NodeModel(data) for node, data in self.graph.nodes.items()}
        self.model_dict = nn.ModuleDict(self.model_dict)
        self.info = info

    def forward(self, x):
        data = {'input': x}
        for block_name in self.order[1:]:
            xx = [data[bn] for bn in self.graph.predecessors(block_name)]
            data[block_name] = self.model_dict[block_name](xx)
        return data['output']

    def block_order(self):
        # return nx.dfs_predecessors(self.graph, 'input')
        return graph_order(self.graph, 'input', 'output')

    def add_ch(self):
        # add input channels for all blocks
        # get all blocks' output channels
        for block_name in self.graph.nodes():
            self.graph.nodes[block_name]['out_ch'] = \
                block_command(self.graph.nodes[block_name]['describe'], out_ch=True)
        # apply input channels
        for block_name in self.graph.nodes():
            in_ch = []
            for pred_block_name in self.graph.pred[block_name]:
                in_ch.append(self.graph.nodes[pred_block_name]['out_ch'])
            self.graph.nodes[block_name]['in_ch'] = in_ch
            if self.graph.nodes[block_name]['type'] == 'layer':
                self.graph.nodes[block_name]['describe'] = \
                    str(sum(in_ch)) + '+' + self.graph.nodes[block_name]['describe']
                if len(in_ch) > 1:
                    self.graph.nodes[block_name]['input_type'] = 'cat'
                else:
                    self.graph.nodes[block_name]['input_type'] = None
            elif self.graph.nodes[block_name]['type'] == 'res_layer':
                if max(in_ch) != min(in_ch):
                    raise ('res layer of ' + str(block_name) +
                           ' suppose to have same input rather than ' + str(in_ch))
                self.graph.nodes[block_name]['describe'] = \
                    str(in_ch[0]) + '+' + self.graph.nodes[block_name]['describe']
                self.graph.nodes[block_name]['input_type'] = 'res'
        # check the output channels equal to upper layer output
        nodelist = self.graph.pred['output']
        for node in nodelist:
            if self.graph.nodes[node]['out_ch'] != int(self.graph.nodes['output']['describe']):
                raise ValueError('output channels is {} but node \'{}\' output is {}'.format(
                    int(self.graph.nodes['output']['describe']), node, self.graph.nodes[node]['out_ch']
                ))
        if len(nodelist) > 1:
            raise ValueError('too many layer connect to output: {}'.format(nodelist))


    def test_order(self):
        # 测试计算顺序
        order = self.block_order()
        compute_dict = {str(node): False for node in nx.nodes(self.graph)}
        compute_dict['input'] = True
        for block_name in order:
            for pred_block_name in self.graph.pred[block_name]:
                if pred_block_name not in compute_dict:
                    raise ValueError(pred_block_name, ' not in self nodes:',
                                     list(compute_dict.keys()))
                if compute_dict[pred_block_name] is True:
                    continue
                else:
                    raise ValueError(block_name, ' compute need ',
                                     pred_block_name, ' but it is uncomputed in ',
                                     compute_dict)
            compute_dict[block_name] = True
        print('compute result is:')
        for k, v in compute_dict.items():
            print(k.ljust(15, '.'), v)
        print('compute order all pass.')

    def test_describe(self):
        print('model node describe:')
        for block_name in self.order:
            print(block_name, ':', self.graph.nodes[block_name]['describe'])

    def plot(self):
        import matplotlib.pyplot as plt
        nx.draw_kamada_kawai(self.graph, with_labels=True, font_weight='bold')
        plt.show()


def layer_cat_process(_x):
    return torch.cat(_x, dim=1)


def layer_res_process(_x):
    output_x = _x[0]
    for xx in _x[1:]:
        output_x += xx
    return output_x


def layer_none_process(_x):
    return _x[0]


class NodeModel(nn.Module):
    """
    nn.Model for graph node
    """

    def __init__(self, node):
        super().__init__()
        self.layer = block_command(node['describe'])
        if node.get('input_type') == 'cat':
            self.preprocess = layer_cat_process
        elif node.get('input_type') == 'res':
            self.preprocess = layer_res_process
        else:
            self.preprocess = layer_none_process

    def forward(self, _x):
        _x = self.preprocess(_x)
        return self.layer(_x)


#
# ___  _    ____ ____ _  _    ____ ____ _  _ _  _ ____ _  _ ___
# |__] |    |  | |    |_/     |    |  | |\/| |\/| |__| |\ | |  \
# |__] |___ |__| |___ | \_    |___ |__| |  | |  | |  | | \| |__/
#
#


def block_command(command_str, out_ch=False):
    def split_command(st):
        if st == '':
            return []
        out = []
        out_dict = {}
        unit = ''
        for s in st:
            if s.isdigit() == unit.isdigit() or len(unit) == 0:
                unit += s
                continue
            else:
                out.append(unit)
                unit = s
        out.append(unit)
        if out[0].isdigit():
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
    priority = {'(': 1, ')': 2, '*': 3, '+': 4}
    start = 0
    for i, s in enumerate(_str):
        if s in '()*+':
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
    if out_ch:
        for com in command_list[::-1]:
            if 'ch' in com:
                return com['ch']
        raise ValueError('can\' find channel info in ', command_str)
    in_ch = command_list[0]['ch']
    block_list = []
    for com in command_list[1:]:
        if com['layer'] == 'c':
            block_list.append(nn.Conv2d(in_ch, com['ch'], kernel_size=com['c'],
                                        stride=com['s'], padding=com['p']))
        elif com['layer'] == 't':
            block_list.append(nn.ConvTranspose2d(in_ch, com['ch'], kernel_size=com['t'],
                                                 stride=com['s'], padding=com['p']))
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


test_str_unet = """# UNet
input: 3
output: 3
encoder: 32c3s1p1 * 3
decoder: 32c3s1p1 * 3
down: 32c3s2p1
down1: down
down2: down
down3: down
up: 32t4s2p1
up1: up
up2: up
up3: up
center: 32c1s1p0
input > encoder
encoder > down1 > down2 > down3 > center
center > up3 > up2 > up1 > decoder > output
down3 > up3
down2 > up2
down1 > up1
"""

test_str_dense_block = """# DenseNet
input: 3
output: 2
encoder: 32c3s1p1
decoder: 32c3s1p1 + leakyrelu + 2c1s1p0
denseBlock: 32c3s1p1 + relu
dense1: denseBlock
dense2: denseBlock
dense3: denseBlock
dense4: denseBlock
input > encoder
encoder > dense1
encoder > dense2
encoder > dense3
encoder > dense4
encoder > decoder
dense1 > dense2
dense1 > dense3
dense1 > dense4
dense1 > decoder
dense2 > dense3
dense2 > dense4
dense2 > decoder
dense3 > dense4
dense3 > decoder
dense4 > decoder
decoder > output
"""

test_str_unet_plus = """#unet++
input: 4
output: 3
encoder: 32c3s1p1 + leakyrelu
decoder: 32c3s1p1 + leakyrelu + 3c1s1p0 + leakyrelu
down: 32c3s2p1 + leakyrelu + 32c3s2p1 + leakyrelu + 32c1s1p0 + leakyrelu
up: 32t4s2p1 + leakyrelu + 32c1s1p0 + leakyrelu + 32t4s2p1 + leakyrelu
center: (32c3s1p1 + leakyrelu)*3
down1:down
down2:down
down3:down
up1_1:up
up1_2:up
up1_3:up
up2_2:up
up2_3:up
up3_3:up


input>encoder
decoder>output
encoder>down1>down2>down3
down3>up3_3>up2_3>up1_3>decoder
down2>up2_2>up1_2
#down1>up1_1
#encoder>decoder
#up1_1>decoder
up1_2>decoder
up1_3>decoder
"""

test_str_multi_blcok_1 = """#encoder-decoder
# multi level network
input: 4
output: 4

encoder: 32c3s1p1 + leakyrelu
decoder: 32c3s1p1 + leakyrelu + 3c1s1p0 + leakyrelu

down: 32c3s2p1 + leakyrelu + 32c3s2p1 + leakyrelu
d1: down*2
d2: down*2
d3: down*2
up: 32t4s2p1 + leakyrelu + 32c1s1p0 + leakyrelu
u1: up*2
u2: up*2
u3: up*2

code: encoder+d1+d2+d3
decode:u1+u2+u3+decoder

input>code>decode>output
"""

test_str_resnet = """#resnet
input: 4
output: 4
encoder: (32c3s1p1+leakyrelu) * 3
decoder: (32c3s1p1+leakyrelu) * 3 + 4c3s1p1+leakyrelu
down: (32c3s2p1+leakyrelu)*2
down1: down
down2: down
down3: down
up: res (32t4s2p1+leakyrelu)*2
up1: up
up2: up
up3: up
center: 32c1s1p0
input > encoder
encoder > down1 > down2 > down3 > center
center > up3 > up2 > up1 > decoder > output
down3 > up3
down2 > up2
down1 > up1
"""

if __name__ == "__main__":
    # 32+(64c3s1p1 +relu)+ (64c1s1p0 + leakyrelu) * 3 + tanh
    # print('result', block_command('(32c3s1p1 + leakyrelu)*3', out_ch=False))
    net = NetGraph(test_str_resnet)
    print(graph_order(net.graph, 'input', 'output'))
    net.test_order()
    net.test_describe()
    # net.test_order()
    # net.plot()
    x = torch.randn(1, 4, 128, 128)
    y = net(x)
    print(y.shape)
    # net.plot()
    pass
