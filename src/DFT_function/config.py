
def load_params_from_json(filepath):
    """
    从JSON文件加载参数。
    
    :param filepath: JSON文件的路径
    :return: 加载的参数字典
    """
    import json
    with open(filepath, 'r') as f:
        params_loaded = json.load(f)
    return params_loaded


def save_params_to_json(params, filepath):
    """
    将参数字典保存到JSON文件。
    
    :param params: 要保存的参数字典
    :param filepath: JSON文件的路径
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    
    print("初始化配置文件保存到",filepath,"完毕")