
# 分析函数

def save_data(X, y, file_path):
    """保存数据到文件"""
    import numpy as np
    np.savez_compressed(file_path, X=X, y=y)
    print(f"数据已保存到 {file_path}")

def load_data(file_path):
    """从文件加载数据"""
    import numpy as np
    data = np.load(file_path)
    return data['X'], data['y']