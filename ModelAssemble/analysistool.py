import torch
import torch.nn as nn


def state_distance(state1, state2, mode=''):
    # 比较两个状态字典的差距
    distance_dict = {}
    for (key1, tensor1), (key2, tensor2) in zip(state1.items(), state2.items()):
        if key1 != key2:
            raise ValueError("unequal layer between model1 and model2:", key1, key2)
        distance_dict[key1] = tensor1 - tensor2
    if mode == 'sum':
        return sum([t.abs().sum() for t in distance_dict.values()])
    return distance_dict


def model_distance(model1: nn.Module, model2: nn.Module, mode=''):
    # 比较两个模型的差距
    model1_state = model1.state_dict()
    model2_state = model2.state_dict()
    state_distance(model1_state, model2_state, mode)
