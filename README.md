# Lazy DeepLearning

A Troch enhance model for simplify coding process in deep-learning

## Introduction

This project offer two class for machine learning, 
an artificial neural network(ANN) model 
and a framework for manage and train models.

## NetGraph

NetGraph is a model generate class. 
The main purpose of this section is to explain how to write 
an ANN describe string for this project.

In this project an ANN can be described by a text like this:
```python
unet_str =\ 
"""#unet
input: 4
output: 3
encoder: 32c3s1p1 + leakyrelu
decoder: 32c3s1p1 + leakyrelu + 3c1s1p0 + leakyrelu
down: (32c3s2p1 + leakyrelu + 32c3s1p1 + leakyrelu)*2
up: (32t4s2p1 + leakyrelu + 32c3s1p1 + leakyrelu)*2 + 32c1s1p0 + leakyrelu
center: 32c2s1p0 + leakyrelu + 32c1s1p0 + leakyrelu + 32t2s1p0 + leakyrelu
down1:down
down2:down
down3:down
down4:down
up1:up
up2:up
up3:up
up4:up

input>encoder>down1>down2>down3>down4>center
center>up4>up3>up2>up1>decoder>output
down4>up4
down3>up3
down2>up2
down1>up1
"""
```

`\n` is used to split the describe. Every line is a definition of ANN. 
But after `#` is recognized as notes.

There are two types of definition:
 
Lines include `:` are blocks definition, which define various layers to be used.

Lines include `>` are connect definition, which define the connect relationship between blocks. Data flow from left to right.

Most time ANN construct is a chain, which mean each layer only have one layer connect to input one layer connect to output.
But in some ANN, a part of layers need to have multi input or output like u-net, so I design this describe method. 
Chain connect layers describe as blocks. Complex connect describe in connect definition additionally.

`input` `output` is key words in this project to identify models input and output.
(At present, only support one input and output, so avoid use these key words in other node)



Use NetGraph generate model by above text.
```python
from ModelAssemble.netgraph import NetGraph
net = NetGraph(unet_str)
```

You can use `.plot()` to show the model's graph
```python
net.plot()
```
![Plot image](/assets/images/graphplot.png)

The class is inherit from `torch.nn.Model`, use model as original.
```python 
>>> net(torch.randn(1,4,512,512)).shape
torch.Size([1, 3, 512, 512])
```

At present design, import NetGraph is to verify 
the text describe can be recognized correctly. 
Use manager above will create NetGraph automatically. 

## FolderManager

FolderManager is a class for manage models, 
which support a easy way to manage your models. 
This class have above features: 
- Storage models construct and models' parameters.
- Save optimzer state for continue train.
- Integration train method. 
    - Record train history
    - Save checkpoint
- Manage multiple datasets.


Above example create a model, 
set several training parameters, 
also specify training dataset. 
Then train the model and save it. 
And easy load frame to continue training:
```python
from ModelAssemble.frame import FolderManager
# Create FolderManager
manage = FolderManager()

# Describe model construction
unet_str =\
"""#unet
input: 4
output: 3
encoder: 32c3s1p1 + leakyrelu
decoder: 32c3s1p1 + leakyrelu + 3c1s1p0 + leakyrelu
down: (32c3s2p1 + leakyrelu + 32c3s1p1 + leakyrelu)*2
up: (32t4s2p1 + leakyrelu + 32c3s1p1 + leakyrelu)*2 + 32c1s1p0 + leakyrelu
center: 32c2s1p0 + leakyrelu + 32c1s1p0 + leakyrelu + 32t2s1p0 + leakyrelu
down1:down
down2:down
down3:down
down4:down
up1:up
up2:up
up3:up
up4:up

input>encoder>down1>down2>down3>down4>center
center>up4>up3>up2>up1>decoder>output
down4>up4
down3>up3
down2>up2
down1>up1
"""

# Set frame parameters
model_set = dict(
    model_str=unet_str,
    train_data = 'data200',
    test_data='data_001_test',
    loss='SmoothL1',
    optim='Adam',
    batch_size=32,
    fakebatch=2,
    bias='bias',
    output_mask='mask',
    data_device='cuda',
    device='cuda',
    filename='unet',
    normalize='normalization'
)

# Create frame
frame = manage.create_frame(model_set)
# Train model
frame.run(1000)
# Save model
manage.save_frame(frame)
```
Load frame in the other program:
```python
from ModelAssemble.frame import FolderManager
# Create FolderManager
manage = FolderManager()

# Load frame
frame = manage.load_frame('unet')
# Continue Train
frame.run(2000)
```

## How to use
#### Create FolderManager 

To use this framework, you need to import and create object.
```python
from ModelAssemble.frame import FolderManager
manage = FolderManager()
```
**Notice: when you create the object, 
it will create several folder under work path in the first time.** 

Folder construct like this:
```
├─ database
├─ modelbase
│   ├─ loss_library.pt
│   └─ optim_library.pt
```

These folders save 4 types of data( and method to list names in folder): 
- Dataset `.data_library()`
- Model `.models()`
- LossFunction `.loss_library()`
- OptimFunction `.optim_library()`
```python
>>> manage.loss_library()
{'L1': 'nn.L1Loss()',
 'L2': 'nn.MSELoss()',
 'SmoothL1': 'nn.SmoothL1Loss()',
 'Huber': 'nn.HuberLoss()'}
```

#### Create Frame
Use `manage.create_frame(model_set)` to create a Frame.

`model_set` is a dict which keys and values should be string or numerical specify the frame setting.
So the setting is easy to save and load.

A full `model_set` looks like this as before show:
```python
model_set = dict(
    model_str=unet_str,
    train_data = 'data200',
    test_data='data_001_test',
    loss='SmoothL1',
    optim='Adam',
    batch_size=32,
    fakebatch=2,
    bias='bias',
    output_mask='mask',
    device='cuda',
    data_device='cuda',
    filename='unet',
    normalize='normalization'
)
```

`model_str`:describe string in NetGraph, usually it is a long string, 
put it in other place can make code clear.

`train_data`: Dataset name, frame will use this dataset to train. 

`test_data`: Dataset name, Optional. 
If train_data not include test data, use this to specify test dataset.
If you want use test dataset included in `train_data`, delete this key.
More detail in above `Dataset` section.

`loss`: loss name, loss function be used.

`optim`: optimzer name, optimzer to be used.

`batch_size`: batch size.

`fakebatch`: little batch, when memory can hold a batch, use this. 
The frame will compute in `fakebatch` size, and backward in `batch_size`.

`bias`: specify unchanged input layer, only use in some question. 
Available value in database

`output_mask`: specify a mask filter output, 
only filtered output effective in backward and evaluation.

`device`: model device. model will train in this device.

`data_device`: dataset device. If in different device with model, 
frame will only transform training data into model's device.

`filename`: name of the model's file. If not specify, 
model will have a serial number depend on models which the manager already have.
If specify, final name still have a serial number before specified name.

`normalize`: if dataset without normalized, specify a normalization, 
it will auto normal data in running the model. 


    
### Dataset
You need to put you dataset into `database` folder. 
you need to use `torch.save()` to prepare the dataset, 
and the name's extension is '.pt'.

The format of dataset should be a list of `torch.Tensor`.
Each Tensor should be a 'Big Batch', 
which mean the first value of shape of Tensor is the number of samples. 
Other values is samples shapes correspond to model's first layer. 
For example, the shape of an image dataset is `(samples number, image layers, H, W)` 
 
List length should be 2 or 4. If 2, the first Tensor is input value,  second Tensor is target value.
Like `[input, target]`. If length is 4, list should be `[input1, target1, input2, target2]`. 
When frame load a 4 length dataset, 
the first group will be take to be training data, 
the secod group will be take to be testing data.

So, when frame use a 4 length dataset just need specify the `train_data`.
If use a 2 length dataset, you need specify `test_data` additionaly.



## catalog

### baseblock

include some method to generate muti-layer or a part of model

### frame

a framework to manage deep-learning model

### netgraph

use string to generate deep-learning network by using graph

### analysistool

analysis model, such as compare models parameters different