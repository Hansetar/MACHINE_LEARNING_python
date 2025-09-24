# 评估
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
eva_dir=readpara["filepaths"]["eva_dir"]
pre_dir=readpara["filepaths"]["pre_dir"]
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
X_train, y_train = load_xy_from_file(eva_dir, input_size, output_size)

print(f"生成的数据形状: X={y_train.shape}, y={y_train.shape}")
print(f"X示例: \n{X_train[:3]}")
print(f"y示例: \n{y_train[:3]}")




# 数据标准化
if os.path.exists(scal_x_dir):
    # 文件存在时的操作
    print(f"{scal_x_dir}文件存在,使用已有标准化方案")
    scaler_X = joblib.load(scal_x_dir)
    
else:
    # 文件不存在时的操作
    print(f"{scal_x_dir}文件不存在，初始化标准并保存，警告，评估过程重新生成标准化，对预测结果不一定是好事")
    scaler_X = StandardScaler()
    joblib.dump(scaler_X, scal_x_dir)
#X_train = scaler_X.fit_transform(X_train)


if os.path.exists(scal_y_dir):
    # 文件存在时的操作
    print(f"{scal_y_dir}文件存在,使用已有标准化方案")
    scaler_y = joblib.load(scal_y_dir)
    
else:
    # 文件不存在时的操作
    print(f"{scal_y_dir}文件不存在，初始化标准并保存，警告，评估过程重新生成标准化，对预测结果不一定是好事")
    scaler_y = StandardScaler()
    joblib.dump(scaler_y, scal_y_dir)


#y_train = scaler_y.fit_transform(y_train)


# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values)


# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



print(f"测试集大小: {len(train_dataset)} 样本")

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

# 打印模型结构
print(model)

# 统计模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")



from DFT_function.fun import evaluate_model 

# 评估加载的模型
predictions, actuals = evaluate_model(model, train_loader, scaler_y,train_result_dir)

# 展示一些预测结果
print("\n预测结果示例:")
for i in range(10):
    print(f"样本 {i+1}: 实际值={actuals[i][0]:.4f}, 预测值={predictions[i][0]:.4f}, 差值={abs(actuals[i][0]-predictions[i][0]):.4f}")