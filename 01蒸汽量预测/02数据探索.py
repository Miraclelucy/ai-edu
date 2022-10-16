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
#plt.show() # 不阻止进程继续执行 后面的图将连续展示

# 画箱式图
column = train_data.columns.tolist()[:39]  # 列表头
fig = plt.figure(figsize=(20, 40))  # 指定绘图对象宽度和高度
for i in range(38):
    plt.subplot(13, 3, i + 1)  # 13行3列子图,宽屏电脑展示效果不好看可以自己调整行列值，反正把39个小小图展示出来就可以了
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)  # 箱式图
    plt.ylabel(column[i], fontsize=8)
#plt.show() # 不阻止进程继续执行 后面的图将连续展示

fig = plt.figure(figsize=(10,5))  # 指定绘图对象宽度和高度
# 查看特征变量‘V0’的数据分布直方图
ax=plt.subplot(1,2,1)
#sns.distplot(train_data['V0'],fit=stats.norm,) #拟合标准正态分布
sns.distplot(train_data['V0'],fit=stats.norm,color="y") #加其他参数可以自定义

# 查看特征变量‘V0’的数据分布是否近似于正态分布,概率分布图
ax=plt.subplot(1,2,2)
stats.probplot(train_data['V0'], plot=plt)
#plt.show() # 不阻止进程继续执行 后面的图将连续展示

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
#plt.show()

# 对比同一特征变量‘V0’下，训练集数据和测试集数据的分布情况
ax = sns.kdeplot(train_data['V0'], color="Red", shade=True)
ax = sns.kdeplot(test_data['V0'], color="Blue", shade=True)
ax.set_xlabel('V0')
ax.set_ylabel("Frequency")
ax = ax.legend(["train","test"])
#plt.show()

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
#plt.show()

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

#plt.show()

# 删除会导致模型泛化能力差的列之后，这些列的训练集数据与测试集数据分布不一致，查看特征变量的相关性
data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'],axis=1)
train_corr = data_train1.corr()
print(train_corr)

# 画出相关性热力图
ax = plt.subplots(figsize=(20, 16))#调整画布大小
ax = sns.heatmap(train_corr, vmax=.8, square=True, annot=True)#画热力图   annot=True 显示系数
#plt.show()

# 找出相关程度
data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'],axis=1)

plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
colnm = data_train1.columns.tolist()  # 列表头
mcorr = data_train1[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象-调色板对象
g1 = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
#plt.show()

# 寻找K个最相关的特征信息
k = 10 # number of variables for heatmap
cols = train_corr.nlargest(k, 'target')['target'].index

cm = np.corrcoef(train_data[cols].values.T) #返回相关系数矩阵
hm = plt.subplots(figsize=(10, 10))#调整画布大小
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True)
#g = sns.heatmap(train_data[cols].corr(),annot=True,square=True,cmap="RdYlGn")
hm = sns.heatmap(train_data[cols].corr(),annot=True,square=True)

#plt.show()

# 查找出特征变量和target变量相关系数大于0.5的特征变量
threshold = 0.5
corrmat = train_data.corr()
top_corr_features = corrmat.index[abs(corrmat["target"])>threshold]
plt.figure(figsize=(220,10))
g2 = sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()

# 查找出特征变量和target变量相关系数小于0.5的特征变量
corr_matrix = data_train1.corr().abs()
drop_col=corr_matrix[corr_matrix["target"]<threshold].index
plt.figure(figsize=(20,10))
g3 = sns.heatmap(train_data[drop_col].corr(),annot=True,cmap="RdYlGn")
#plt.show()
print(drop_col)

# 删除之前测试集和训练集分布不一致的列，会导致模型泛化能力差的列
drop_columns = ['V5','V9','V11','V17','V22','V28']
#merge train_set and test_set
train_x =  train_data.drop(['target'], axis=1)

#data_all=pd.concat([train_data,test_data],axis=0,ignore_index=True)
data_all = pd.concat([train_x,test_data])
data_all.drop(drop_columns,axis=1,inplace=True)
#View data
print(data_all.head())  # [5 rows x 32 columns]

# 归一化处理
# normalise numeric columns
cols_numeric=list(data_all.columns)

def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax,axis=0)
data_all[cols_numeric].describe()

# 训练集和测试集的数据都归一化处理
#col_data_process = cols_numeric.append('target')
train_data_process = train_data[cols_numeric]
train_data_process = train_data_process[cols_numeric].apply(scale_minmax,axis=0)

test_data_process = test_data[cols_numeric]
test_data_process = test_data_process[cols_numeric].apply(scale_minmax,axis=0)

# 把所有的列分成2组：第一组13列和第二组是剩下的列
cols_numeric_left = cols_numeric[0:13]
cols_numeric_right = cols_numeric[13:]

train_data_process = pd.concat([train_data_process, train_data['target']], axis=1)

fcols = 6
frows = len(cols_numeric_left)
plt.figure(figsize=(4 * fcols, 4 * frows))
i = 0

# 第一组13列每列的6种数据
# 分别是[数据直方图，概率分布图，相关系数图，空值和归一化处理后的直方图，概率分布图，相关系数图，]
for var in cols_numeric_left:
    dat = train_data_process[[var, 'target']].dropna()

    i += 1
    plt.subplot(frows, fcols, i)
    sns.distplot(dat[var], fit=stats.norm);
    plt.title(var + ' Original')
    plt.xlabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    _ = stats.probplot(dat[var], plot=plt)
    plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[var])))
    plt.xlabel('')
    plt.ylabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(dat[var], dat['target'], '.', alpha=0.5)
    plt.title('corr=' + '{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))

    i += 1
    plt.subplot(frows, fcols, i)
    trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
    trans_var = scale_minmax(trans_var)
    sns.distplot(trans_var, fit=stats.norm);
    plt.title(var + ' Tramsformed')
    plt.xlabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    _ = stats.probplot(trans_var, plot=plt)
    plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
    plt.xlabel('')
    plt.ylabel('')

    i += 1
    plt.subplot(frows, fcols, i)
    plt.plot(trans_var, dat['target'], '.', alpha=0.5)
    plt.title('corr=' + '{:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
