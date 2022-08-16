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
