

def save_model_config(config, config_path):
    """保存配置"""
    import json
    
    # 保存配置信息
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到 {config_path}")

def save_model_model(model, model_path):
    """保存模型"""
    import json
    import torch
    # 保存模型状态字典
    torch.save(model.state_dict(), model_path)
    
    print(f"模型已保存到 {model_path}")

def save_model(model, config, model_path, config_path):
    """保存模型和配置"""
    import torch
    import json
    # 保存模型状态字典
    torch.save(model.state_dict(), model_path)
    
    # 保存配置信息
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"模型已保存到 {model_path}")
    print(f"配置已保存到 {config_path}")