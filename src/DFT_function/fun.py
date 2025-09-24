
# 杂项函数
def evaluate_model(model, test_loader, scaler_y,png_dir):
    """评估模型性能"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    #print("data:\n",predictions)
    
    # 反标准化数据
    #predictions = scaler_y.inverse_transform(predictions)
    #actuals = scaler_y.inverse_transform(actuals)
    
    # 计算评估指标
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    


    # 确保保存目录存在
    os.makedirs(png_dir, exist_ok=True)

    csv_file=png_dir+"eva.csv"
    
    data = pd.DataFrame({
        'Actual Values': actuals.flatten(),
        'Predicted Values': predictions.flatten()
    })
    data.to_csv(csv_file, index=False)

    

    print("CSV文件保存到了", csv_file)

    


    png_dir=png_dir+"eva.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    #plt.show()
    plt.savefig(png_dir) 
    print("文件保存到了",png_dir)
    
    return predictions, actuals

def predict_single_sample(model, sample, scaler_X, scaler_y):
    """使用模型预测单个样本不好用"""
    import torch
    # 标准化输入
    sample_scaled = scaler_X.fit_transform(sample)
    
    # 转换为张量
    sample_tensor = torch.FloatTensor(sample_scaled)
    
    # 预测
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(sample_tensor).numpy()
    
    # 反标准化预测结果
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction



def predict_modle(model, test_loader):
    """评估模型性能"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    
    # 转换为numpy数组
    
    predictions = np.array(predictions)
    #actuals = np.array(actuals)
    
    #print("data:\n",predictions)
    
    # 反标准化数据
    #predictions = scaler_y.inverse_transform(predictions)
    #actuals = scaler_y.inverse_transform(actuals)
    #print(predictions)
    #print(np.array(test_loader))
    
    return predictions





def insert_column_and_save(df, prediction_data, column_name, file_path,index_out=False):
    """
    在DataFrame的最后一列插入一个新列，并保存到新的变量和指定路径的文件中。

    :param df: 原始的Pandas DataFrame。
    :param prediction_data: 要插入的NumPy数组。
    :param column_name: 新列的名称。
    :param file_path: 保存文件的路径。
    :return: 新的DataFrame with the added column。
    """
    import pandas as pd
    import numpy as np
    # 确保prediction_data的长度与df的行数相同
    if len(prediction_data) != len(df):
        raise ValueError("The length of the prediction data does not match the number of rows in the DataFrame.")

    # 添加新列
    df_with_prediction = df.assign(**{column_name: prediction_data})

    # 保存到指定路径
    df_with_prediction.to_csv(file_path, index=index_out)

    print(f"预测结果保存到 {file_path}")

    # 返回新的DataFrame
    return df_with_prediction

# 示例使用：
# 假设有一个DataFrame df 和一个NumPy数组 prediction_data
# df = pd.DataFrame(...)
# prediction_data = np.array(...)

# 调用函数
# new_df = insert_column_and_save(df, prediction_data, 'Prediction Result', 'path/to/your/file.csv')





def get_datetime_string():
    '''
    获取当前时分秒
    '''
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")



def save_path_to_file(path, file_name):
    """将路径保存到指定文件"""
    with open(file_name, 'w') as file:
        file.write(path)

def read_path_from_file(file_name):
    """从指定文件读取路径"""
    with open(file_name, 'r') as file:
        path = file.read()
    return path