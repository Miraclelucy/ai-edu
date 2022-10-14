import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# seaborn知识补充：sns.distplot()用法 画直方图
# 和条形图的区别，条形图有空隙，直方图没有，条形图一般用于类别特征，直方图一般用于数字特征（连续型）
# seaborn.displot()用法参考这里https://seaborn.pydata.org/generated/seaborn.displot.html
# stats.probplot()用法参考这里https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html

train_data_file = "./zhengqi_train.txt"
test_data_file =  "./zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

# 给出样本数据的相关信息概览 ：行数，列数|列索引，列非空值个数，列类型|内存占用
print(train_data.info())
print(test_data.info())

# 直接给出样本数据的一些基本的统计量，包括均值，标准差，最大值，最小值，分位数等。
print(train_data.describe())
print(test_data.describe())

# head()给出了前5条数据的基本信息
print(train_data.head())
print(test_data.head())

# 画箱形图探索数据
fig = plt.figure(figsize=(4, 6))  # 指定绘图对象宽度和高度
sns.boxplot(train_data['V0'], orient="v", width=0.5)
plt.show(block=False) # 不阻止进程继续执行

# 画箱式图
column = train_data.columns.tolist()[:39]  # 列表头
fig = plt.figure(figsize=(20, 40))  # 指定绘图对象宽度和高度
for i in range(38):
    plt.subplot(13, 3, i + 1)  # 13行3列子图
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)  # 箱式图
    plt.ylabel(column[i], fontsize=8)
plt.show(block=False) # 不阻止进程继续执行

fig = plt.figure(figsize=(10,5))
# 查看特征变量‘V0’的数据分布直方图
ax=plt.subplot(1,2,1)
#sns.distplot(train_data['V0'],fit=stats.norm,) #拟合标准正态分布
res = sns.distplot(train_data['V0'],fit=stats.norm,color="y") #加其他参数可以自定义
res.set_titles("VO data histograms ")

# 查看特征变量‘V0’的数据分布是否近似于正态分布,概率分布图
ax=plt.subplot(1,2,2)
stats.probplot(train_data['V0'], plot=plt)
plt.show()

