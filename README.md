# 一个简单的神经网络模型（V1.0）

本代码仅适合纯数值的回归模型，仅可借鉴，后果自负，支持超参数搜索



# 安装环境

```
pip install numpy  scikit-learn matplotlib
pytorch(根据自己环境安装)
```

# 训练流程

## 1. 配置文件初始化

运行
```
src/config_ini.py
```
注意，路径一定要改正确，不要出错，有修改权限。
config_dir_save文件很重要，不能随便改动，配置文件改动之后一定要执行一次本脚本，另外如果单独修改训练参数，不进行超参数搜索，需要创建`configpath`对应的文件，格式如下：
```json
{
    "input_size": 18,
    "hidden_sizes": [
        64,
        32
    ],
    "output_size": 1,
    "activation": [
        "relu",
        "tanh"
    ],
    "dropout_rate": 0.3,
    "training_params": {
        "batch_size": 64,
        "learning_rate": 0.0005,
        "num_epochs": 100,
        "weight_decay": 1e-06
    }
}
```

## 2. 超参数最优搜索

使用
```
src/hyperparametersearch.py
```


## 3. 训练

使用
```
src/train.py
```


## 4. 评估

使用
```
src/evaluate.py
```

## 5. 预测

使用
```
src/evaluate.py
```

## 6. 再训练

使用
```
src/train.py
```

确保模型参数一致
