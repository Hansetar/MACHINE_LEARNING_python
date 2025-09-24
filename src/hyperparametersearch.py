# 超参数搜索
from DFT_function.env import *
from DFT_function.RegressionModel import RegressionModel # 导入核心类
import joblib
from DFT_function.fun import  read_path_from_file





# 读取参数
from DFT_function.config import load_params_from_json # 读取参数函数

modelconfig_dir=read_path_from_file("config_dir_save")
os.path.exists(modelconfig_dir) or print(f"文件 {modelconfig_dir} 不存在") or exit(1)
readpara=load_params_from_json(modelconfig_dir)

# 模型参数
input_size=readpara["modelparams"]["inputsize"]
hidden_sizes=readpara["modelparams"]["hiddensizes"]
output_size=readpara["modelparams"]["outputsize"]
activations=readpara["modelparams"]["activations"]
dropout_rate=readpara["modelparams"]["dropoutrate"]
# 训练参数
num_samples=readpara["dataparams"]["numsamples"]
noise_level=readpara["dataparams"]["noiselevel"]
test_size=readpara["dataparams"]["testsize"]
# 数据参数
batch_size=readpara["trainingparams"]["batchsize"]
learning_rate=readpara["trainingparams"]["learningrate"]
num_epochs=readpara["trainingparams"]["numepochs"]
weight_decay=readpara["trainingparams"]["weightdecay"]
# 文件路径
inputdatapath=readpara["filepaths"]["inputdatapath"]
data_path=readpara["filepaths"]["datapath"]
model_path=readpara["filepaths"]["modelpath"]
config_path=readpara["filepaths"]["configpath"]
scal_x_dir=readpara["filepaths"]["scal_x_dir"]
scal_y_dir=readpara["filepaths"]["scal_y_dir"]
# 超参数搜索涉及的参数
model_save_dir=readpara["hyperparametersearch"]["modelsavedir"]
results_path=readpara["hyperparametersearch"]["resultspath"]

# 超参数参数读入

h_min_layers=readpara["hyperparameterpara"]["min_layers"]
h_max_layers=readpara["hyperparameterpara"]["max_layers"]
h_activations=readpara["hyperparameterpara"]['activations']
h_units_options=readpara["hyperparameterpara"]["units_options"]
h_max_combinations=readpara["hyperparameterpara"]["max_combinations"]
h_dropout_rate=readpara["hyperparameterpara"]["dropout_rate"]
h_learning_rate=readpara["hyperparameterpara"]["learning_rate"]
h_batch_size=readpara["hyperparameterpara"]["batch_size"]
h_weight_decay=readpara["hyperparameterpara"]["weight_decay"]

h_other_params = {
        'dropout_rate': h_dropout_rate,
        'learning_rate': h_learning_rate,
        'batch_size': h_batch_size,
        'weight_decay': h_weight_decay
    }

# 超参数执行参数读入

h_num_trials=readpara["hyperparameterexepara"]["num_trials"]
h_num_epochs=readpara["hyperparameterexepara"]["num_epochs"]
h_patience=readpara["hyperparameterexepara"]["patience"]




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
scaler_X = StandardScaler()
joblib.dump(scaler_X, scal_x_dir)
#X_train = scaler_X.fit_transform(X_train)
#X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
joblib.dump(scaler_y, scal_y_dir)
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


# 超参数搜索
from DFT_function.haperfun import train_model,hyperparameter_search,generate_matching_param_space  # 读取数据原始文件函数
# 创建目录
import os

os.makedirs(model_save_dir, exist_ok=True)

# 生成参数空间
param_space = generate_matching_param_space(
        min_layers=h_min_layers,          # 最小5层
        max_layers=h_max_layers,          # 最大5层（即只生成5层的组合）
        activations=h_activations,  # 可用激活函数
        units_options=h_units_options,  # 每层可能的神经元数量
        max_combinations=h_max_combinations,  # 最大组合数量
        other_params=h_other_params  # 其他参数
    )

# 打印生成的参数空间信息
print(f"生成了 {len(param_space['hidden_sizes'])} 种隐藏层和激活函数组合")
print("前5个组合示例:")
for i in range(5):
    print(f"组合 {i+1}:")
    print(f"  隐藏层大小: {param_space['hidden_sizes'][i]}")
    print(f"  激活函数: {param_space['activations'][i]}")

# 打印其他参数
print("\n其他参数搜索空间:")
for key, value in param_space.items():
    if key not in ['hidden_sizes', 'activations']:
        print(f"  {key}: {value}")

# 执行超参数搜索
h_best_model, h_best_params, search_results = hyperparameter_search(
    param_space, 
    X_train,
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    model_path,
    model_save_dir,
    num_trials=h_num_trials,  # 实际使用时可增加
    num_epochs=h_num_epochs,
    patience=h_patience
)

# 训练参数替换为测试最优参数
print(h_best_model)
hidden_sizes=h_best_params["hidden_sizes"]
dropout_rate=h_best_params["dropout_rate"]
activations=h_best_params["activations"]
batch_size=h_best_params["batch_size"]
learning_rate=h_best_params["learning_rate"]
weight_decay=h_best_params["weight_decay"]


print("已替换内容")
print("隐藏层：",hidden_sizes)
print("Dropout率：",dropout_rate)
print("激活函数：",activations)
print("batch_size：",batch_size)
print("learning_rate：",learning_rate)
print("weight_decay：",weight_decay)


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

print("超参数搜索结束，请检查文件并及时更换训练的路径配置")