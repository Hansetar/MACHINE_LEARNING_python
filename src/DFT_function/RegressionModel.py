import torch.nn as nn
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activations, dropout_rate=0.0):
        super(RegressionModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # 处理激活函数参数
        if isinstance(activations, str):
            # 如果是字符串，所有层使用相同激活函数
            activations = [activations] * len(hidden_sizes)
        elif len(activations) != len(hidden_sizes):
            # 如果是列表但长度不匹配，抛出错误
            raise ValueError("激活函数列表长度必须与隐藏层数量相同")
        
        # 构建隐藏层
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # 根据指定的激活函数添加相应层
            if activations[i] == 'relu':
                self.layers.append(nn.ReLU())
            elif activations[i] == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activations[i] == 'tanh':
                self.layers.append(nn.Tanh())
            elif activations[i] == 'leaky_relu':
                self.layers.append(nn.LeakyReLU(0.01))
            elif activations[i] == 'elu':
                self.layers.append(nn.ELU())
            elif activations[i] == 'none':
                pass  # 不添加激活函数
            else:
                raise ValueError(f"不支持的激活函数: {activations[i]}")
            
            # 添加Dropout（除了最后一层）
            if dropout_rate > 0 and i < len(hidden_sizes) - 1:
                self.layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # 输出层（没有激活函数）
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x