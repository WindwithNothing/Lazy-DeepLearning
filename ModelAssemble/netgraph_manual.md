# NetGraph 神经网络图生成器

完全使用字符串描述神经网络结构生成神经网络

一个期望实现的U-Net描述：
```
input: 3
output: 3
encoder: (32c3s1p1 + relu) * 3
decoder: (32c3s1p1 + relu) * 3
down: 32c3s2p1 + relu
down1: down
down2: down
down3: down
up: 32t4s2p1 + relu
up1: up
up2: up
up3: up
center: 32c1s1p0 + relu
input > encoder
encoder > down1 > down2 > down3 > center
center > up3 > up2 > up1 > decoder > output
down3 > up3
down2 > up2
down1 > up1
```

## 目标功能
+ block设置计算符，设计计算顺序和计算方法，构建block 
+ 针对多输入的block能做cat或res

## 描述定义：
### 卷积层定义
一个卷积层以字母分割并标识数字，例如`32c3s1p1` ，其各字符含义如下

meaning|out channel|convolution type and kernel size|stride|padding 
---|---|---|---|---
alpha|-|C:convolution T:transpose|s|p|
value|32|3|1|1|

## 处理顺序

#### 1.建立图结构

一行中含有`:`判断为层定义行，否则为关系定义行

层定义行：定义神经网络结构，可以使用`()+*`运算符，进行复杂的定义。
`+`串联多个网络块；网络块`*`数字与`()`组合复制网络块，通过复制字符串实现。

关系定义行：定义神经网络间的连接关系，使用`>`定义链接方向。
对于多输入的层，自动进行`cat`操作，并行连接张量。

根据定义将相应的字符串赋予给图节点。

#### 2.建立神经网络

根据每个节点的字符串定义建立相应的神经网络。
