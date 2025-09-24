
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
        print(f"\rEpoch {epoch+1}/{num_epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}", end="")
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
    
    # 计算总组合数（不生成所有组合）
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print("总组合数为：", total_combinations)
    
    # 动态调整num_trials
    original_num_trials = num_trials
    min_trials = max(1, int(total_combinations * 0.1))  # 至少为1，但不少于总数的10%
    
    if num_trials < min_trials:
        print(f"警告: 原始num_trials({original_num_trials})低于总组合数的10%，如果需要，请自行做出调整，总组合数*0.1={min_trials}")
    elif num_trials > total_combinations:
        num_trials = total_combinations
        print(f"警告: 原始num_trials({original_num_trials})超过总组合数，已自动调整为{num_trials}")
    
    # 检查组合数是否超过num_trials
    if total_combinations > num_trials:
        print(f"提示: 总组合数({total_combinations})超过当前试验次数({num_trials})，将随机选择{num_trials}个组合，还有{total_combinations - num_trials}个组合没有被遍历到。")
    else:
        print(f"提示: 将遍历所有{total_combinations}个组合。")
    
    # 生成组合（随机选择）
    if total_combinations <= num_trials:
        # 生成所有组合并随机打乱
        all_combinations = list(dict(zip(param_names, combo)) 
                               for combo in product(*param_values))
        random.shuffle(all_combinations)
        selected_combinations = all_combinations
    else:
        # 随机选择不重复的组合（不生成所有组合）
        param_lengths = [len(values) for values in param_values]
        selected_combinations = []
        seen = set()
        
        print("正在随机生成组合...")
        with tqdm(total=num_trials, desc="生成组合") as pbar:
            while len(selected_combinations) < num_trials:
                # 随机生成一个组合的索引元组
                indices = tuple(random.randint(0, length-1) for length in param_lengths)
                if indices not in seen:
                    seen.add(indices)
                    # 根据索引生成组合
                    combo = [param_values[i][idx] for i, idx in enumerate(indices)]
                    selected_combinations.append(dict(zip(param_names, combo)))
                    pbar.update(1)
                    # 实时显示生成进度
                    pbar.set_postfix({"已生成": len(selected_combinations), "剩余": num_trials - len(selected_combinations)})
    
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
    import random
    
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
    
    # 计算理论上的最大组合数（不生成实际组合）
    total_possible = 0
    for num_layers in layer_counts:
        hidden_combinations = len(units_options) ** num_layers
        activation_combinations = len(activations) ** num_layers
        total_possible += hidden_combinations * activation_combinations
    
    print(f"理论最大组合数: {total_possible}")
    
    # 确定实际要生成的组合数
    actual_combinations = min(total_possible, max_combinations)
    print(f"将生成 {actual_combinations} 个组合")
    
    # 用于跟踪已生成的组合，避免重复
    generated_combinations = set()
    
    # 随机生成组合
    while len(generated_combinations) < actual_combinations:
        # 随机选择层数
        num_layers = random.choice(layer_counts)
        
        # 随机生成隐藏层节点组合
        hidden = [random.choice(units_options) for _ in range(num_layers)]
        
        # 随机生成激活函数组合
        acts = [random.choice(activations) for _ in range(num_layers)]
        
        # 创建组合的唯一标识（用于去重）
        combination_key = (tuple(hidden), tuple(acts))
        
        # 如果组合未生成过，则添加到结果中
        if combination_key not in generated_combinations:
            generated_combinations.add(combination_key)
            param_space['hidden_sizes'].append(hidden)
            param_space['activations'].append(acts)
            
            # 实时显示进度
            if len(generated_combinations) % 100 == 0 or len(generated_combinations) == actual_combinations:
                print(f"已生成 {len(generated_combinations)}/{actual_combinations} 个组合")
    
    return param_space




# 超参数搜索函数
def hyperparameter_search_all(param_space,X_train,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,results_path,model_save_dir, num_trials=20, num_epochs=50, patience=5):
    '''
    先生成全部组合在计算，适合10万组合以下的方案，超过10万组合以上的不好用
    '''
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
def generate_matching_param_space_all(min_layers=2, max_layers=5, 
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
