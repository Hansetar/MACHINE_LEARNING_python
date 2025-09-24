

def load_xy_from_file(file_path, x_cols, y_cols):
    """
    从文件中加载x和y变量。
    
    :param file_path: 文件路径
    :param x_cols: x变量所在的列数（从前开始计数）
    :param y_cols: y变量所在的列数（从x变量后开始计数）
    :return: x, y数据的元组
    """
    import pandas as pd
    # 读取文件
    data = pd.read_csv(file_path)
    
    # 确保x_cols和y_cols不超出列的范围
    if x_cols + y_cols > data.shape[1]:
        raise ValueError("x_cols和y_cols的总和超过了文件的总列数")
    
    # 分割x和y变量
    x = data.iloc[:, :x_cols]
    y = data.iloc[:, x_cols:x_cols+y_cols]
    
    return x, y

def load_model(model_path, config_path):
    """加载模型"""
    import json
    from DFT_function.RegressionModel import RegressionModel
    import torch
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建模型实例
    model = RegressionModel(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        output_size=config['output_size'],
        activations=config['activation'],
        dropout_rate=config['dropout_rate']
    )
    
    # 加载模型状态
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    
    print(f"模型已从 {model_path} 加载")
    print(f"配置已从 {config_path} 加载")
    
    return model, config


def load_config(config_path):
    """加载模型和配置"""
    import json
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"配置已从 {config_path} 加载")
    
    return config


def load_model_retrian(model_path, config_path):
    """加载模型"""
    import json
    from DFT_function.RegressionModel import RegressionModel
    import torch
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建模型实例
    model = RegressionModel(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        output_size=config['output_size'],
        activations=config['activation'],
        dropout_rate=config['dropout_rate']
    )
    
    # 加载模型状态
    model.load_state_dict(torch.load(model_path))
    model.train()  # 设置为训练模式
    
    print(f"模型已从 {model_path} 加载")
    print(f"配置已从 {config_path} 加载")
    
    return model, config