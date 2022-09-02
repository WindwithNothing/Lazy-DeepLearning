import os
import time
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import write_png
from tqdm.auto import tqdm as Tqdm
from .netgraph import NetGraph
import seaborn as sns
import matplotlib.pyplot as plt


class TrainFrame:
    __doc__ = """A Framework for train a CNN model, 
    just give Model, Dataset, LossFunction, Optimizer 
    method and batch"""

    def __init__(self, model, traindata, testdata, loss_func, optim, batchsize,
                 output_mask=None,
                 fakebatch=None,
                 checkdata=None,
                 path=None,
                 device=None,
                 info=None,
                 manager=None,
                 **kwargs
                 ):
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a nn.Module")
        if not isinstance(traindata, Dataset):
            raise ValueError("traindate must be a Dataset")
        if not isinstance(testdata, Dataset):
            raise ValueError("testdate must be a Dataset")
        if batchsize % fakebatch != 0:
            raise ValueError("batchsize must to be divisible by fakebatch")

        self.manager = manager
        self.model = model
        if device:
            self.device = device
            self.model.to(device)
        else:
            self.device = 'cpu'
        self.loss_func = loss_func
        self.optim = optim(model.parameters())
        self.info = info
        if path is None:
            path = '.\train'
            if not os.path.exists(path):
                os.makedirs(path)
        self.set_dict = {'batchsize': batchsize, 'fakebatch': fakebatch, 'path': path}
        if fakebatch:
            self.set_dict['batch'] = fakebatch
        else:
            self.set_dict['batch'] = batchsize
        for key, value in kwargs.items():
            if key in self.set_dict:
                self.set_dict[key] = value
            else:
                raise ValueError("Unknow key " + key)
        self.traindata = DataLoader(traindata, batch_size=self.set_dict['batch'], shuffle=True)
        self.testdata = DataLoader(testdata, batch_size=self.set_dict['batch'], shuffle=False)
        self.checkdata = None
        self.output_mask = output_mask
        self.set_checkdata(checkdata)
        self.history = pd.DataFrame(columns='epoch train_loss test_loss date'.split(' '))

    def get_params(self):
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict(),
            history=self.history,
        )

    def load_params(self, params_dict):
        self.model.load_state_dict(params_dict['model'])
        self.optim.load_state_dict(params_dict['optim'])
        self.history = params_dict['history']
        optimizer_to(self.optim, self.device)

    def set_checkdata(self, data):
        if data:
            self.checkdata = DataLoader(data, batch_size=self.set_dict['batch'], shuffle=False)
        else:
            self.checkdata = None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.set_dict:
                self.set_dict[key] = value
            else:
                raise ValueError(key + ' not in self.set_dict')

    def record(self, epoch, train_loss, test_loss):
        dic = dict(
            epoch=[epoch],
            train_loss=[train_loss],
            test_loss=[test_loss],
            date=[time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())],
        )
        df = pd.DataFrame(dic)
        self.history = pd.concat([self.history, df], ignore_index=True)

    def plot_history(self):
        sns.set_theme(style="whitegrid")
        s = sns.lineplot(data=self.history[['train_loss', 'test_loss']])
        s.axes.set_yscale('log')
        s.axes.set_title('minimum: train_loss:{:.8f}, test_loss{:.8f}'.format(self.history['train_loss'].min(),
                                                                              self.history['test_loss'].min()))
        min_index = self.history['test_loss'].idxmin()
        min_value = self.history['test_loss'][min_index]
        s.axes.scatter(min_index, min_value, color='r')
        s.axes.set_xlim(0)

    def run(self, times, checkpoint: int = 50):
        # main method for train model
        for i in Tqdm(range(times), desc=self.info['filename']):
            train_loss, test_loss = self.epoch()
            self.record(i, train_loss, test_loss)
            if test_loss <= self.history['test_loss'].min() and self.manager:
                self.manager.save_frame(self, info='best')
            if checkpoint is not None and (i % checkpoint == 0 or i == times - 1):
                # checkpoint
                self.manager.save_frame(self)
                if self.checkdata:
                    loss, img = self.run_dataloader(self.checkdata, tqdm=False, output=True)
                    # self.save_img(img, name=f'epoch{i}_{loss}.png')

    def epoch(self, **kwargs):
        train_loss = self.epoch_train(**kwargs)
        test_loss = self.epoch_test(**kwargs)
        return train_loss, test_loss

    def epoch_train(self, **kwargs):
        return self.run_dataloader(self.traindata, optim=True, **kwargs)

    def epoch_test(self, **kwargs):
        return self.run_dataloader(self.testdata, optim=False, **kwargs)

    def run_dataloader(self, dataloader: DataLoader,
                       optim=False,
                       tqdm=True,
                       output=False,
                       loss=None):
        """
        base mathod for run a dataloader
        :param dataloader: DataLoader object in torch
        :param optim: set True to optimize model, defult False
        :param tqdm: set True to use tqdm, or get a tqdm object, defult True
        :param output: set True to output all result in a big batch with detached, defult False
        :param loss: give a loss function to use, defult None is model_set loss
        :return: loss value, result image if ouput set to True
        """
        model = self.model
        if loss is None:
            loss_func = self.loss_func
        else:
            loss_func = loss
        if optim is not False:
            optim = self.optim
            optim.zero_grad()
            model.train()
        else:
            model.eval()
        batch_size = self.set_dict['batchsize']
        loss_sum = 0.0
        batch_sum = 0
        return_y = []
        if tqdm is True:
            iterative = Tqdm(enumerate(dataloader), desc='batch', leave=False, total=len(dataloader))
        elif callable(tqdm):
            iterative = tqdm(enumerate(dataloader), desc='batch', leave=False, total=len(dataloader))
        else:
            iterative = enumerate(dataloader)

        for i, (epoch_x, epoch_y) in iterative:
            epoch_x = epoch_x.to(self.device)
            epoch_y = epoch_y.to(self.device)
            if optim is not False:
                pred_y = model(epoch_x)
            else:
                with torch.no_grad():
                    pred_y = model(epoch_x)
            if self.output_mask is not None:
                _loss = loss_func(pred_y * self.output_mask, epoch_y * self.output_mask)
            else:
                _loss = loss_func(pred_y, epoch_y)
            loss_sum += float(_loss) * pred_y.shape[0]
            if optim is not False:
                _loss.backward()
                batch_sum += epoch_x.shape[0]
                if batch_sum >= batch_size or i == len(dataloader) - 1:
                    optim.step()
                    optim.zero_grad()
                    batch_sum = 0
            if output:
                return_y.append(pred_y.detach())
        loss_sum /= len(dataloader.dataset)
        if output:
            return loss_sum, torch.cat(return_y, dim=0)
        return loss_sum

    def to(self, device):
        self.model.to(device)
        if self.output_mask is not None:
            self.output_mask = self.output_mask.to(device)
        optimizer_to(self.optim, device)
        self.device = device

    def save_img(self, img, path=None, name=None):
        if path is None:
            if self.set_dict['path'] is not None:
                if name is not None:
                    path = os.path.join(self.set_dict['path'], name)
                else:
                    raise ValueError('should add name to save_img')
            else:
                raise ValueError("TrainFrame havn\'t set path to save")
        write_png(img, path)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class FolderManager:
    def __init__(self):
        self.database = './database'
        self.base = './modelbase'
        self.loss_library_name = os.path.join(self.base, 'loss_library.pt')
        self.optim_library_name = os.path.join(self.base, 'optim_library.pt')
        # Frame save file names:
        self.save_names = dict(
            params='Parameters.pt',
            best_params='Parameters_best.pt',
            model_set='model_set.txt',
            history='history.csv',
            history_plot='history_plot.svg'
        )
        if not os.path.exists(self.base):
            os.makedirs(self.base)
        if not os.path.exists(self.database):
            os.makedirs(self.database)
        self._init_file()
        pass

    def models(self):
        file_list = os.listdir(self.base)
        file_list = filter(lambda n: n.split('_')[0].isdigit(), file_list)
        file_list = list(file_list)
        file_list.sort(key=lambda n: int(n.split('_')[0]))
        return file_list

    def default_new_model_name(self):
        file_list = self.models()
        if len(file_list) == 0:
            return '0'
        number = int(file_list[-1].split('_')[0]) + 1
        return str(number)

    def create_frame(self, model_set: dict):
        # use a dict to create a ANN frame, dict value should all be strings
        # the dict must have:
        # model_str, train_data(, test_data), loss, optim, batch_size
        # and optiones:
        # bias, output_mask, fakebatch, filename, data_device, normalize
        model_set = model_set.copy()
        model = NetGraph(model_set['model_str'])
        loss = self.loss_library(model_set['loss'])
        optim = self.optim_library(model_set['optim'])
        train_data, test_data = self.load_dataset(model_set)
        if 'data_device' in model_set:
            train_data.to(model_set['data_device'])
            test_data.to(model_set['data_device'])
        if 'output_mask' in model_set:
            output_mask = self.data_library(model_set['output_mask'])
        else:
            output_mask = None
        batch_size = model_set['batch_size']
        fakebatch = model_set.get('fakebatch')
        if not model_set.get('filename'):
            model_set['filename'] = self.default_new_model_name()
        else:
            if model_set['filename'] not in self.models():
                model_set['filename'] = self.default_new_model_name() + '_' + model_set['filename']
        path = os.path.join(self.base, model_set['filename'])
        return TrainFrame(model,
                          traindata=train_data, testdata=test_data,
                          loss_func=loss, optim=optim, output_mask=output_mask,
                          batchsize=batch_size, fakebatch=fakebatch,
                          path=path,
                          info=model_set,
                          manager=self,
                          )

    def save_frame(self, frame: TrainFrame, info=None):
        if info is None:
            pre = ''
        else:
            pre = info + '_'
        file = os.path.join(self.base, frame.info['filename'])
        if not os.path.exists(file):
            os.mkdir(file)
        # params file
        data = frame.get_params()
        torch.save(data, os.path.join(file, pre + self.save_names['params']))
        # set file
        model_set_str = json.dumps(frame.info)
        with open(os.path.join(file, pre + self.save_names['model_set']), 'w') as f:
            f.write(model_set_str)
        # history file
        data['history'].to_csv(os.path.join(file, pre + self.save_names['history']))
        # save history plot
        if info is None:
            plt.figure(figsize=(12, 6))
            frame.plot_history()
            plt.savefig(os.path.join(file, self.save_names['history_plot']))
            plt.close()

    def load_frame_set(self, filename, info=None, new_set: dict = None):
        if info is None:
            pre = ''
        else:
            pre = info + '_'
        file = os.path.join(self.base, filename)
        # set file
        with open(os.path.join(file, pre + self.save_names['model_set']), 'r') as f:
            model_set_str = f.read()
        model_set = json.loads(model_set_str)
        if new_set is not None:
            model_set.update(new_set)
        return model_set

    def load_frame(self, filename, info=None, new_set: dict = None):
        if info is None:
            pre = ''
        else:
            pre = info + '_'
        file = os.path.join(self.base, filename)
        model_set = self.load_frame_set(filename=filename, info=info, new_set=new_set)
        frame = self.create_frame(model_set)
        # params
        data = torch.load(os.path.join(file, pre + self.save_names['params']))
        frame.load_params(data)
        return frame

    def load_dataset(self, model_set: dict):
        train_data = self.data_library(model_set['train_data'])
        if len(train_data) == 4:
            train_x, train_y, test_x, test_y = train_data
            if 'test_data' in model_set:
                test_x, test_y = self.data_library(model_set['test_data'])
        else:
            train_x, train_y = train_data
            if 'test_data' in model_set:
                test_x, test_y = self.data_library(model_set['test_data'])
            else:
                raise ValueError('train_data ' + model_set['train_data'] +
                                 ' not includ test_data, \'test_data\' is needed')
        _bias = None
        if 'bias' in model_set:
            _bias = self.data_library(model_set['bias'])
        if 'normalize' in model_set:
            normal = self.load_normal(model_set['normalize'])
        else:
            normal = True
        train_dataset = TensorDataset(train_x, train_y, bias=_bias, normal=normal)
        test_dataset = TensorDataset(test_x, test_y, bias=_bias, normal=normal)
        if 'data_device' in model_set:
            train_dataset.to(model_set['data_device'])
            test_dataset.to(model_set['data_device'])
        return train_dataset, test_dataset

    def load_normal(self, name):
        return NormalObject(params_dict=self.data_library(name))

    def load_history(self, model: str, info: str = None):
        file = os.path.join(self.base, model)
        if not os.path.exists(file):
            raise FileNotFoundError('model:', file, 'not found')
        if info is None:
            pre = ''
        else:
            pre = info + '_'
        file = os.path.join(file, pre + self.save_names['history'])
        if not os.path.exists(file):
            raise FileNotFoundError('file:', file, 'not found')
        return pd.read_csv(file)

    def generate_normalize(self, name, data=None, input_data=None, target_data=None):
        if data is not None:
            no = NormalObject(input_data=data[0], target_data=data[1])
        else:
            no = NormalObject(input_data=input_data, target_data=target_data)
        torch.save(no.get_params(), os.path.join(self.database, name + '.pt'))

    def _init_file(self):
        if not os.path.exists(self.loss_library_name):
            loss_dict = dict(
                L1='nn.L1Loss()',
                L2='nn.MSELoss()',
                SmoothL1='nn.SmoothL1Loss()',
                Huber='nn.HuberLoss()'
            )
            torch.save(loss_dict, self.loss_library_name)
        if not os.path.exists(self.optim_library_name):
            optim_dict = dict(
                Adam='torch.optim.Adam',
                SGD='torch.optim.SGD',
            )
            torch.save(optim_dict, self.optim_library_name)

    def loss_library(self, key=None):
        if key:
            return eval(torch.load(self.loss_library_name)[key], {'nn': nn})
        else:
            return torch.load(self.loss_library_name)

    def optim_library(self, key=None):
        if key:
            return eval(torch.load(self.optim_library_name)[key], {'torch': torch})
        else:
            return torch.load(self.optim_library_name)

    def data_library(self, key=None):
        file_list = os.listdir(self.database)
        if key:
            for file in file_list:
                if key == os.path.splitext(file)[0]:
                    data = torch.load(os.path.join(self.database, file))
                    return data
            raise ValueError("Could not find file named as:", key)
        else:
            return file_list


class NormalObject:
    """
    Normalize data for Frame, this will auto scale data from data.
    """
    def __init__(self, input_data=None, target_data=None,
                 input_boundary=None, target_boundary=None, params_dict=None):
        if input_boundary is not None and target_boundary is not None:
            self.input_boundary = input_boundary
            self.target_boundary = target_boundary
        elif params_dict:
            self.input_boundary = params_dict['input_boundary']
            self.target_boundary = params_dict['target_boundary']
        elif input_data is not None and target_data is not None and \
                target_boundary is None and input_boundary is None:
            self.input_boundary = self.get_boundary(input_data)
            self.target_boundary = self.get_boundary(target_data)
        else:
            raise ValueError("only input_data target_data combination or "
                             "input_boundary target_boundary combination "
                             "can as input for NormalObject")

    def get_params(self):
        return dict(
            input_boundary=self.input_boundary,
            target_boundary=self.target_boundary,
        )

    def norm_input(self, data):
        new_data = torch.ones_like(data)
        for i, (low, high) in enumerate(self.input_boundary):
            new_data[:, i] = self.normalize(data[:, i], low, high)
        return new_data

    def norm_target(self, data):
        new_data = torch.ones_like(data)
        for i, (low, high) in enumerate(self.target_boundary):
            new_data[:, i] = self.normalize(data[:, i], low, high)
        return new_data

    def denorm_input(self, data):
        new_data = torch.ones_like(data)
        if data.shape[1] != len(self.input_boundary):
            print('warning: data shape not equal to input_boundary')
        for i, (low, high) in enumerate(self.input_boundary):
            new_data[:, i] = self.denormalize(data[:, i], low, high)
        return new_data

    def denorm_target(self, data):
        new_data = torch.ones_like(data)
        if data.shape[1] != len(self.target_boundary):
            print('warning: data shape not equal to target_boundary')
        for i, (low, high) in enumerate(self.target_boundary):
            new_data[:, i] = self.denormalize(data[:, i], low, high)
        return new_data

    def get_boundary(self, data):
        out = []
        for i in range(data.shape[1]):
            high = data[:, i, :, :].max()
            low = data[:, i, :, :].min()
            out.append([low, high])
        return out

    def normalize(self, data, low, high):
        return (data - low) / (high - low)

    def denormalize(self, data, low, high):
        return data * (high - low) + low


class TensorDataset(Dataset):
    def __init__(self, input, target, bias=None, normal: NormalObject = False):
        self.input = input
        self.target = target
        self.bias = bias
        self.normal = normal
        if type(self.normal) == NormalObject:
            self.input = self.normal.norm_input(self.input)
            self.target = self.normal.norm_target(self.target)
        elif self.normal is True:
            print('Warning: ', self, 'normal is True, TensorDataset will scale data automatic, '
                                     'but this may lead to different scale between train data and test data.'
                                     'a uniform NormalObject shuld set to normal')
            self.input_boundary = self.get_boundary(self.input)
            self.target_boundary = self.get_boundary(self.target)
            self.input = self.norm_input(self.input)
            self.target = self.norm_target(self.target)

    def __getitem__(self, i):
        if self.bias is not None:
            return torch.cat([self.input[i], self.bias]), self.target[i]
        return self.input[i], self.target[i]

    def __len__(self):
        return self.input.shape[0]

    def to(self, device):
        self.input = self.input.to(device)
        self.target = self.target.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)

    def get_boundary(self, data):
        out = []
        for i in range(data.shape[1]):
            high = data[:, i, :, :].max()
            low = data[:, i, :, :].min()
            out.append([low, high])
        return out

    def norm_input(self, data):
        for i, (low, high) in enumerate(self.input_boundary):
            data[:, i] = self.normalize(data[:, i], low, high)
        return data

    def norm_target(self, data):
        for i, (low, high) in enumerate(self.target_boundary):
            data[:, i] = self.normalize(data[:, i], low, high)
        return data

    def normalize(self, data, low, high):
        return (data - low) / (high - low)

    def denormalize(self, data, low, high):
        return data * (high - low) + low
