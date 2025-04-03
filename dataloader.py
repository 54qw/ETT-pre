import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

class TransformerOilTempDataset(Dataset):
    def __init__(self, data_path, window, is_test=False):
        # 读取 CSV 数据
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])  # 转换日期格式
        df = df.drop(columns=['date'])  # 移除日期列，仅保留数值数据

        # 归一化数据
        scaler = preprocessing.MinMaxScaler()
        df_scaled = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

        features = df_scaled
        seq_len = window
        amount_of_features = len(features.columns)  #有几列
        features = features.values  #pd.DataFrame(stock) 表格转化为矩阵
        sequence_length = seq_len + 1   #序列长度

        data = []
        # 生成时间序列样本
        for index in range(len(features) - sequence_length):
            data.append(features[index: index + sequence_length])
        data = np.array(data)  # 转换为 NumPy 数组

        # 训练集与测试集划分
        train_size = int(0.9 * data.shape[0])  # 90% 用于训练，10% 用于测试
        train_data, test_data = data[:train_size], data[train_size:]
        x_train = train_data[:, :-1, :-1]
        y_train = train_data[:, -1, -1]
        x_test = test_data[:, :-1, :-1]
        y_test = test_data[:, -1, -1]
        if not is_test:
            self.data = x_train  # 取前 window 天作为输入
            self.label = y_train  # 取最后一天的 OT 作为目标值
        else:
            self.data = x_test
            self.label = y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])

if __name__ == '__main__':
    dataset = TransformerOilTempDataset('Dataset/ETT-small/ETTm1.csv', window=10, is_test=False)
    print(f"Dataset size: {len(dataset)}")  # 确保数据量正常
    print(f"Sample input shape: {dataset[0][0].shape}")  # 确保输入 shape 正确
    print(f"Sample output shape: {dataset[0][1].shape}")  # 确保标签 shape 正确


    # df = pd.read_csv("./Dataset/ETT-small/ETTm1.csv", parse_dates=["date"])
    # # 设置时间索引
    # df.set_index("date", inplace=True)
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df["OT"], label="Oil Temperature (OT)")
    # plt.xlabel("Time")
    # plt.ylabel("Temperature")
    # plt.title("Transformer Oil Temperature Over Time")
    # plt.legend()
    # plt.show()
