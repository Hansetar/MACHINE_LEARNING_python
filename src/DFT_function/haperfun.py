
# 超参数搜索专用函数
# 训练函数
def train_model(model, train_loader, val_loader, params, num_epochs=50, patience=5):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import json
    import random
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=params['learning_rate'], 
                          weight_decay=params['weight_decay'])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"早停于第 {epoch+1} 轮")
                break
        
        # 打印进度
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, best_val_loss


# 超参数搜索函数
def hyperparameter_search(param_space,X_train,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,results_path,model_save_dir, num_trials=20, num_epochs=50, patience=5):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import json
    import random
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    import csv
    import os   
    import shutil
    from itertools import product
    from DFT_function.RegressionModel import RegressionModel


    
    best_params = None
    best_val_loss = float('inf')
    best_model = None
    results = []
    
    # 生成所有可能的参数组合
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    all_combinations = list(dict(zip(param_names, combo)) 
                           for combo in product(*param_values))
    

    # 动态调整，避免报错和太小
    total_combinations = len(all_combinations)
    print("总组合数为：",total_combinations)
    min_trials = max(1, int(total_combinations * 0.1))  # 至少为1，但不少于总数的10%

    # 动态调整num_trials
    original_num_trials = num_trials
    if num_trials < min_trials:
        #num_trials = min_trials
        print(f"警告: 原始num_trials({original_num_trials})低于总组合数的10%，如果需要，请自行做出调整，总组合数*0.1={min_trials}")
    elif num_trials > total_combinations:
        num_trials = total_combinations
        print(f"警告: 原始num_trials({original_num_trials})超过总组合数，已自动调整为{num_trials}")
    



    # 随机选择组合
    random.shuffle(all_combinations)
    selected_combinations = all_combinations[:num_trials]
    
    print(f"开始超参数搜索，共 {num_trials} 组参数...")

    
    is_first_call_csv = True
    
    for i, params in enumerate(tqdm(selected_combinations, desc="超参数搜索")):
        print(f"\n试验 {i+1}/{num_trials}: {params}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # 创建模型
        model = RegressionModel(
            input_size=X_train.shape[1],
            hidden_sizes=params['hidden_sizes'],
            output_size=1,
            activations=params['activations'],
            dropout_rate=params['dropout_rate']
        )
        
        # 训练模型
        model, train_losses, val_losses, val_loss = train_model(
            model, train_loader, val_loader, params, num_epochs, patience
        )
        
        # 记录结果
        result = {
            'params': params,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        results.append(result)

        # 保存搜索结果（实时）
        results_path_csv=results_path+"h_s_result.csv"
            
        # 保存为csv文件
        # 如果是第一次调用，插入标题
        fieldnames = ['params', 'val_loss', 'train_losses', 'val_losses']
        if is_first_call_csv:
            if os.path.exists(results_path_csv):
                # 如果文件存在，进行备份
                shutil.copy(results_path_csv, results_path_csv + '.bak')
            dir_path = os.path.dirname(results_path_csv)
            # 如果目录不存在，则创建它
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # 写入标题和数据
            with open(results_path_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(result)
            is_first_call_csv = False
        else:
            # 如果不是第一次调用，追加数据
            with open(results_path_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)




        # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = model
            
            # 保存最佳模型
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            best_model_pth=model_save_dir+'best_model.pth'
            torch.save(best_model.state_dict(),best_model_pth)
            print(f"发现新最佳模型，验证损失: {best_val_loss:.4f}")
    
    results_path_json=results_path+"h_s_result.json"
    dir_path = os.path.dirname(results_path_json)
    # 如果目录不存在，则创建它
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 保存搜索结果
    with open(results_path_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n超参数搜索完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳参数: {best_params}")
    
    return best_model, best_params, results


# 超参数生成函数
def generate_matching_param_space(min_layers=2, max_layers=5, 
                                 activations=['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'],
                                 units_options=[32, 64, 128, 256, 512],
                                 max_combinations=1000,
                                 other_params=None):
    """
    生成匹配的隐藏层和激活函数组合，并覆盖到param_space中
    
    参数:
        min_layers: 最小隐藏层数
        max_layers: 最大隐藏层数
        activations: 可用的激活函数列表
        units_options: 每层可能选择的节点数列表
        max_combinations: 最大组合数量限制
        other_params: 其他参数的搜索空间 (dropout_rate, learning_rate等)
    
    返回:
        更新后的param_space字典
    """
    import itertools
    import random
    import numpy as np
    # 初始化param_space
    param_space = {
        'hidden_sizes': [],
        'activations': []
    }
    
    # 添加其他参数
    if other_params:
        param_space.update(other_params)
    
    # 生成所有可能的层数
    layer_counts = range(min_layers, max_layers + 1)
    
    # 为每个层数生成组合
    all_hidden_sizes = []
    all_activations = []
    
    for num_layers in layer_counts:
        # 生成所有可能的隐藏层节点组合
        hidden_combinations = list(itertools.product(units_options, repeat=num_layers))
        
        # 生成所有可能的激活函数组合
        activation_combinations = list(itertools.product(activations, repeat=num_layers))
        
        # 取两个组合的笛卡尔积
        combined = list(itertools.product(hidden_combinations, activation_combinations))
        
        # 添加到总列表中
        for hidden, acts in combined:
            all_hidden_sizes.append(list(hidden))
            all_activations.append(list(acts))
    
    # 如果组合数量超过限制，随机采样
    if len(all_hidden_sizes) > max_combinations:
        indices = random.sample(range(len(all_hidden_sizes)), max_combinations)
        all_hidden_sizes = [all_hidden_sizes[i] for i in indices]
        all_activations = [all_activations[i] for i in indices]
    
    # 更新param_space
    param_space['hidden_sizes'] = all_hidden_sizes
    param_space['activations'] = all_activations
    
    return param_space



