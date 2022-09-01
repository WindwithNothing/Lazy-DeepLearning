# Lazy DeepLearning

A Troch enhance model for simplify coding process in deep-learning

## Introduction

This project offer two class for machine learning, 
a neural network model 
and a framework for manage and train models.

### NetGraph

NetGraph is a model generate class.

In this project a neural network can be described by a text like this:
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

Use NetGraph generate model by this text.
```python
from ModelAssemble.netgraph import NetGraph
net = NetGraph(unet_str)
```

You can use `.plot()` to show the graph construt
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

### FolderManager

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
manage = FolderManager()

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

frame = manage.create_frame(model_set)
frame.run(1000)
manage.save_frame(frame)
```
```python
# Load frame in the other program.
from ModelAssemble.frame import FolderManager
manage = FolderManager()

frame = manage.load_frame('unet')
frame.run(2000)
```


To use this framework, you need to import and create object.
```python
from ModelAssemble.frame import FolderManager
manage = FolderManager()
```
Notice: when you create the object, 
it will create several folder under work path in the first time.

Folder construct like this:
```
├─ database
├─ modelbase
│   ├─ loss_library.pt
│   └─ optim_library.pt
```
    
#### Dataset
You need to put you dataset into `database` folder. 
you need to use `torch.save()` to prepare the dataset.

The format of dataset should be a list of `torch.Tensor`.
 
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