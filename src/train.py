# 训练
from DFT_function.env import *
from DFT_function.RegressionModel import RegressionModel # 导入核心类
import joblib
from DFT_function.fun import  read_path_from_file



# 读取参数
from DFT_function.config import load_params_from_json # 读取参数函数

modelconfig_dir=read_path_from_file("config_dir_save")
os.path.exists(modelconfig_dir) or print(f"文件 {modelconfig_dir} 不存在") or exit(1)
readpara=load_params_from_json(modelconfig_dir)


# 训练参数
num_samples=readpara["dataparams"]["numsamples"]
noise_level=readpara["dataparams"]["noiselevel"]
test_size=readpara["dataparams"]["testsize"]
t_patience=readpara["trainingparams"]["patience"]
t_min_delta=readpara["trainingparams"]["min_delta"]

# 文件路径
inputdatapath=readpara["filepaths"]["inputdatapath"]
data_path=readpara["filepaths"]["datapath"]
model_path=readpara["filepaths"]["modelpath"]
config_path=readpara["filepaths"]["configpath"]
train_result_dir=readpara["filepaths"]["train_result_dir"]
scal_x_dir=readpara["filepaths"]["scal_x_dir"]
scal_y_dir=readpara["filepaths"]["scal_y_dir"]

# 检测调优版本是否存在，存在优先使用调优部分

if os.path.exists(config_path):
    # 文件存在时的操作
    print(f"{config_path}文件存在,读取配置文件")
    from DFT_function.readfile import load_config
    loaded_config = load_config(config_path)
    # 模型参数
    input_size=loaded_config["input_size"]
    hidden_sizes=loaded_config["hidden_sizes"]
    output_size=loaded_config["output_size"]
    activations=loaded_config["activation"]
    dropout_rate=loaded_config["dropout_rate"]

    # 数据参数
    batch_size=loaded_config["training_params"]["batch_size"]
    learning_rate=loaded_config["training_params"]["learning_rate"]
    num_epochs=loaded_config["training_params"]["num_epochs"]
    weight_decay=loaded_config["training_params"]["weight_decay"]
    
else:
    # 文件不存在时的操作
    print(f"{config_path}文件不存在，使用初始化文件配置")
    # 模型参数
    input_size=readpara["modelparams"]["inputsize"]
    hidden_sizes=readpara["modelparams"]["hiddensizes"]
    output_size=readpara["modelparams"]["outputsize"]
    activations=readpara["modelparams"]["activations"]
    dropout_rate=readpara["modelparams"]["dropoutrate"]

    # 数据参数
    batch_size=readpara["trainingparams"]["batchsize"]
    learning_rate=readpara["trainingparams"]["learningrate"]
    num_epochs=readpara["trainingparams"]["numepochs"]
    weight_decay=readpara["trainingparams"]["weightdecay"]
    # 保存模型配置
    from DFT_function.writefile import save_model_config

    model_config = {
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size,
        'activation': activations,
        'dropout_rate': dropout_rate,
        'training_params': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'weight_decay': weight_decay
        }
    }

    save_model_config(model_config, config_path)

    print(f"配置文件保存到{config_path}")
    from DFT_function.readfile import load_config
    loaded_config = load_config(config_path)





from DFT_function.readfile import load_xy_from_file  # 读取数据原始文件函数
X, y = load_xy_from_file(inputdatapath, input_size, output_size)

print(f"生成的数据形状: X={X.shape}, y={y.shape}")
print(f"X示例: \n{X[:3]}")
print(f"y示例: \n{y[:3]}")


# 数据处理和分析
from DFT_function.analyize import save_data,load_data  # 读取数据原始文件函数



# 保存生成的数据
save_data(X, y, data_path)

# 从文件加载数据
X_loaded, y_loaded = load_data(data_path)

# 数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_loaded, y_loaded, test_size=test_size, random_state=42
)

# 数据标准化

if os.path.exists(scal_x_dir):
    # 文件存在时的操作
    print(f"{scal_x_dir}文件存在,使用已有标准化方案")
    scaler_X = joblib.load(scal_x_dir)
    
else:
    # 文件不存在时的操作
    print(f"{scal_x_dir}文件不存在，初始化标准并保存")
    scaler_X = StandardScaler()
    joblib.dump(scaler_X, scal_x_dir)


#X_train = scaler_X.fit_transform(X_train)
#X_test = scaler_X.transform(X_test)


if os.path.exists(scal_y_dir):
    # 文件存在时的操作
    print(f"{scal_y_dir}文件存在,使用已有标准化方案")
    scaler_y = joblib.load(scal_y_dir)
    
else:
    # 文件不存在时的操作
    print(f"{scal_y_dir}文件不存在，初始化标准并保存")
    scaler_y = StandardScaler()
    joblib.dump(scaler_y, scal_y_dir)



scaler_y = StandardScaler()
#y_train = scaler_y.fit_transform(y_train)
#y_test = scaler_y.transform(y_test)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)} 样本")
print(f"测试集大小: {len(test_dataset)} 样本")
print(f"批次数量: {len(train_loader)}")



if os.path.exists(model_path):
    # 文件存在时的操作
    print(f"{model_path} 模型存在，继续训练，如果报错，说明 {model_path} 损坏或者和预定于参数不一致，请删掉原模型，重新训练")
    from DFT_function.readfile import load_model_retrian # 读取参数函数
    model, loaded_config=load_model_retrian(model_path, config_path)

    # 模型参数
    input_size=loaded_config["input_size"]
    hidden_sizes=loaded_config["hidden_sizes"]
    output_size=loaded_config["output_size"]
    activations=loaded_config["activation"]
    dropout_rate=loaded_config["dropout_rate"]

    # 数据参数
    batch_size=loaded_config["training_params"]["batch_size"]
    learning_rate=loaded_config["training_params"]["learning_rate"]
    num_epochs=loaded_config["training_params"]["num_epochs"]
    weight_decay=loaded_config["training_params"]["weight_decay"]
else:
    # 文件不存在时的操作
    print(f"{model_path}模型不存在，全新训练")
    # 实例化模型
    model = RegressionModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activations=activations,
        dropout_rate=dropout_rate
    )

# 打印模型结构
print(model)

# 统计模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay
)

# 早停参数
patience = t_patience  # 允许损失不下降的epoch数
min_delta = t_min_delta  # 认为改善的最小变化量
best_loss = float('inf')  # 初始化最佳损失为无穷大
counter = 0  # 计数器，记录连续未改善的epoch数
early_stop = False  # 早停标志

# 训练循环
train_losses = []
test_losses = []

for epoch in range(num_epochs):

    if early_stop:
        print(f"触发早停机制\nEarly stopping triggered at epoch {epoch+1}")
        break
    # 训练模式
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    # 计算平均训练损失
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)
    
    # 评估模式
    model.eval()
    running_test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)
    
    # 计算平均测试损失
    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)

    # 早停逻辑
    if epoch_test_loss < best_loss - min_delta:
        best_loss = epoch_test_loss
        counter = 0
        # 保存最佳模型
        #torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            early_stop = True
    
    # 打印进度
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], ' 
              f'Train Loss: {epoch_train_loss:.4f}, ' 
              f'Test Loss: {epoch_test_loss:.4f}')

loss_fig=train_result_dir+"loss.png"
loss_csv=train_result_dir+"loss.csv"

# 保存为CSV文件
import pandas as pd
loss_data = pd.DataFrame({
    "Epoch": list(range(1, len(train_losses) + 1)),
    "Training Loss": train_losses,
    "Test Loss": test_losses
})
loss_data.to_csv(loss_csv, index=False)
print("损失数据已保存到:", loss_csv)

# 绘制训练和测试损失曲线(保存为文件)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig(loss_fig) 
print("损失函数保存到:",loss_fig)

# 保存模型配置
from DFT_function.writefile import save_model

# 准备配置信息
model_config = {
    'input_size': input_size,
    'hidden_sizes': hidden_sizes,
    'output_size': output_size,
    'activation': activations,
    'dropout_rate': dropout_rate,
    'training_params': {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'weight_decay': weight_decay
    }
}

# 保存模型和配置
save_model(model, model_config, model_path, config_path)

