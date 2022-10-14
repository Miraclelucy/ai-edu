import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# seaborn知识补充：sns.distplot()用法 画直方图和条形图的区别，条形图有空隙，直方图没有，条形图一般用于类别特征，直方图一般用于数字特征（连续型）
# 3种主要的分布图：直方分布图，核密度估计，经验累积分布图 {“hist”, “kde”, “ecdf”}
# histograms (hist) and kernel density estimates (KDEs), you can also draw empirical cumulative distribution functions (ECDFs)
# seaborn.displot() 画不同的图：直方分布图，核密度估计，经验累积分布图，通过指定kind{“hist”, “kde”, “ecdf”} 用法参考这里https://seaborn.pydata.org/generated/seaborn.displot.html
# stats.probplot() 画概率分布图 用法参考这里https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
# seaborn.kdeplot() 画核密度估计图


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
plt.show(block=False) # 不阻止进程继续执行 后面的图将连续展示

# 画箱式图
column = train_data.columns.tolist()[:39]  # 列表头
fig = plt.figure(figsize=(20, 40))  # 指定绘图对象宽度和高度
for i in range(38):
    plt.subplot(13, 3, i + 1)  # 13行3列子图,宽屏电脑展示效果不好看可以自己调整行列值，反正把39个小小图展示出来就可以了
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)  # 箱式图
    plt.ylabel(column[i], fontsize=8)
plt.show(block=False) # 不阻止进程继续执行 后面的图将连续展示

fig = plt.figure(figsize=(10,5))  # 指定绘图对象宽度和高度
# 查看特征变量‘V0’的数据分布直方图
ax=plt.subplot(1,2,1)
#sns.distplot(train_data['V0'],fit=stats.norm,) #拟合标准正态分布
sns.distplot(train_data['V0'],fit=stats.norm,color="y") #加其他参数可以自定义

# 查看特征变量‘V0’的数据分布是否近似于正态分布,概率分布图
ax=plt.subplot(1,2,2)
stats.probplot(train_data['V0'], plot=plt)
plt.show(block=False) # 不阻止进程继续执行 后面的图将连续展示

# 展示所有列[V0,V1,...,V38]共39列的数据分布直方图和概率分布图，一共需要78个小图
train_cols = 6
train_rows = int(len(train_data.columns) / 3) # 13行
plt.figure(figsize=(4 * train_cols, 2 * train_rows))  # 指定绘图对象宽度和高度
i = 0
for col in train_data.columns:
    i += 1
    ax = plt.subplot(train_rows, train_cols, i) # 13行6列的子图 一共78个小图图
    sns.distplot(train_data[col], fit=stats.norm)

    i += 1
    ax = plt.subplot(train_rows, train_cols, i) # 13行6列的子图 一共78个小图图
    res = stats.probplot(train_data[col], plot=plt)
plt.show(block=False)

# 对比同一特征变量‘V0’下，训练集数据和测试集数据的分布情况
ax = sns.kdeplot(train_data['V0'], color="Red", shade=True)
ax = sns.kdeplot(test_data['V0'], color="Blue", shade=True)
ax.set_xlabel('V0')
ax.set_ylabel("Frequency")
ax = ax.legend(["train","test"])
plt.show(block=False)

# 查看所有特征变量下，训练集数据和测试集数据的分布情况，一共38个核密度估计子图
# 可以看到特征'V5','V9','V11','V17','V22','V28' 训练集数据与测试集数据分布不一致，会导致模型泛化能力差，采用删除此类特征方法
dist_cols = 5
dist_rows = 8
plt.figure(figsize=(4 * dist_cols, 2 * dist_rows))

i = 1
for col in test_data.columns:
    ax = plt.subplot(dist_rows, dist_cols, i)
    ax = sns.kdeplot(train_data[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])

    i += 1
plt.show(block=False)

# 查看特征变量‘V0’与'target'变量的线性回归关系
fcols = 2
frows = 1

plt.figure(figsize=(8,4))

ax=plt.subplot(1,2,1)
sns.regplot(x='V0', y='target', data=train_data, ax=ax,
            scatter_kws={'marker':'.','s':3,'alpha':0.3},
            line_kws={'color':'k'});
plt.xlabel('V0')
plt.ylabel('target')

ax=plt.subplot(1,2,2)
sns.distplot(train_data['V0'].dropna())
plt.xlabel('V0')

plt.show(block=False)

# 删除会导致模型泛化能力差的列之后，这些列的训练集数据与测试集数据分布不一致，查看特征变量的相关性
data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'],axis=1)
train_corr = data_train1.corr()
print(train_corr)

# 画出相关性热力图
ax = plt.subplots(figsize=(20, 16))#调整画布大小
ax = sns.heatmap(train_corr, vmax=.8, square=True, annot=True)#画热力图   annot=True 显示系数
plt.show()

# 找出相关程度
data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'],axis=1)

plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
colnm = data_train1.columns.tolist()  # 列表头
mcorr = data_train1[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
plt.show()


