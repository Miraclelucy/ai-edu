import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# 读取数据集
test_data = pd.read_csv('./data_format1/test_format1.csv')
train_data = pd.read_csv('./data_format1/train_format1.csv')

user_info = pd.read_csv('./data_format1/user_info_format1.csv')
user_log = pd.read_csv('./data_format1/user_log_format1.csv')

# 数据集样例查看
print(train_data.head(5)) # 用户购买训练集
print(test_data.head(5)) # 用户购买测试集
print(user_info.head(5)) # 用户画像信息
print(user_log.head(5)) # 用户行为日志

# 单变量数据分析
print(user_info.info()) # 用户画像信息
print(user_log.info()) # 用户行为日志
print(train_data.info()) # 用户购买训练集

