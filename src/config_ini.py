
from DFT_function.config import save_params_to_json
from DFT_function.fun import  save_path_to_file


params = {
    "modelparams": {
        "inputsize": 18,
        "hiddensizes": [512, 256, 256, 128, 128, 64, 32, 16],
        "outputsize": 1,
        "activations": ['relu', 'sigmoid', 'tanh', 'relu', 'sigmoid', 'relu', 'tanh', 'tanh'],
        "dropoutrate": 0.2
    },
    "trainingparams": {
        "batchsize": 128,
        "learningrate": 0.001,
        "numepochs": 100,
        "weightdecay": 1e-5,
        "patience": 10,  # 允许损失不下降的epoch数
        "min_delta": 0.001  # 认为改善的最小变化量 
    },
    "dataparams": {
        "numsamples": 3000,
        "noiselevel": 0.1,
        "testsize": 0.2
    },
    "filepaths": {
        "inputdatapath":"<your-dir>/data/train_data.csv",
        "datapath": "<your-dir>/regressiondata.npz",
        "modelpath": "<your-dir>/modle/regression_model.pth",
        "configpath": "<your-dir>/config/modelconfig-now.json",
        "train_result_dir": "<your-dir>/trainresult/",
        "eva_dir": "<your-dir>/掺杂点区域3_I_I面.csv",
        "pre_dir": "<your-dir>/pre.csv",
        "scal_x_dir": "<your-dir>/config/scaler_X.pkl",
        "scal_y_dir": "<your-dir>/config/scaler_Y.pkl",
        "predic_dir": "<your-dir>/predic/"
    },
    "hyperparametersearch": {
        "modelsavedir": "<your-dir>/modle/hyperparametersearch/",   # 搜索保存的最佳模型路径
        "resultspath": "<your-dir>/modle/hyperparametersearch-result/" # 搜索结果保存路径
    },
    "hyperparameterpara": {                   # 超参数搜索部分
        "dropout_rate": [0.1, 0.2, 0.3, 0.4],
        "learning_rate": [0.001, 0.0005, 0.0001],
        "batch_size": [32, 64, 128],
        "weight_decay": [1e-5, 1e-4, 1e-6],
        "min_layers":2,          
        "max_layers":2,          
        "activations":['relu', 'sigmoid', 'tanh', 'leaky_relu'],  
        "units_options":[32, 64],  
        "max_combinations":10,  
    }
    ,
    "hyperparameterexepara": {
        "num_trials": 4,  ## 搜索可能数最大数量
        "num_epochs": 50, 
        "patience": 5
    }
}


# JSON文件路径
configpath = "<your-dir>/config/modelconfig.json"

# 调用函数保存参数
save_params_to_json(params, configpath)

save_path_to_file(configpath,"config_dir_save")