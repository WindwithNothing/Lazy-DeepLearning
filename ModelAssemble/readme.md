# 预制模型

## 超分辨率
### 瀑布结构 fallSRNN
```mermaid  
graph TD;
Input-->Decoder-->Output;
Input-->ConvL1-->Decoder;
ConvL1-->ConvL2-->Decoder;
ConvL2-->ConvL...-->Decoder;
```
