# Lazy DeepLearning

A Troch enhance model for simplify coding process in deep-learning

In this project a neural network can be described by a text like this:
```
unet_str = """#unet
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
You can use `.plot()` to show the graph construt.plot()
```python
net.plot()
```
![Plot image](/assets/images/graphplot.png)




## catalog

### baseblock

include some method to generate muti-layer or a part of model

### frame

a framework to manage deep-learning model

### netgraph

use string to generate deep-learning network by using graph

### analysistool

analysis model, such as compare models parameters different