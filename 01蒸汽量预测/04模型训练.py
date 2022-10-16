import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

train_data_file = "./zhengqi_train.txt"
test_data_file =  "./zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

print(test_data.describe())

# 箱型图做异常值检测
plt.figure(figsize=(18, 10))
plt.boxplot(x=train_data.values,labels=train_data.columns)
plt.hlines([-7.5, 7.5], 0, 40, colors='r')
plt.show()

# 删除异常值
train_data = train_data[train_data['V9']>-7.5]
train_data.describe()

test_data.describe()

# 最大最小值归一化
from sklearn import preprocessing

features_columns = [col for col in train_data.columns if col not in ['target']]

min_max_scaler = preprocessing.MinMaxScaler()

min_max_scaler = min_max_scaler.fit(train_data[features_columns])

train_data_scaler = min_max_scaler.transform(train_data[features_columns])
test_data_scaler = min_max_scaler.transform(test_data[features_columns])

train_data_scaler = pd.DataFrame(train_data_scaler)
train_data_scaler.columns = features_columns

test_data_scaler = pd.DataFrame(test_data_scaler)
test_data_scaler.columns = features_columns

train_data_scaler['target'] = train_data['target']

# 查看所有特征变量下，训练集数据和测试集数据的分布情况，一共38个核密度估计子图
# 可以看到特征'V5','V9','V11','V17','V22','V28' 训练集数据与测试集数据分布不一致，会导致模型泛化能力差，采用删除此类特征方法
dist_cols = 5
dist_rows = 8

plt.figure(figsize=(4 * dist_cols, 2 * dist_rows))

for i, col in enumerate(test_data_scaler.columns):
    ax = plt.subplot(dist_rows, dist_cols, i + 1)
    ax = sns.kdeplot(train_data_scaler[col], color="Red", shade=True)
    ax = sns.kdeplot(test_data_scaler[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])

plt.show()

# 查看特征'V5', 'V17', 'V28', 'V22', 'V11', 'V9'数据的数据分布
drop_col = 6
drop_row = 1

plt.figure(figsize=(5*drop_col,5*drop_row))

for i, col in enumerate(["V5","V9","V11","V17","V22","V28"]):
    ax =plt.subplot(drop_row,drop_col,i+1)
    ax = sns.kdeplot(train_data_scaler[col], color="Red", shade=True)
    ax= sns.kdeplot(test_data_scaler[col], color="Blue", shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train","test"])
plt.show()

# 特征相关性分布图
plt.figure(figsize=(20, 16))
column = train_data_scaler.columns.tolist()
mcorr = train_data_scaler[column].corr(method="spearman")
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
plt.show()

# 相关性分析
# 跟target相关性大于0.1的列倒序排序
mcorr=mcorr.abs()
numerical_corr=mcorr[mcorr['target']>0.1]['target']
print(numerical_corr.sort_values(ascending=False))

index0 = numerical_corr.sort_values(ascending=False).index
print(train_data_scaler[index0].corr('spearman'))

# 相关性初筛(单个特征之间的相关性)
features_corr = numerical_corr.sort_values(ascending=False).reset_index()
features_corr.columns = ['features_and_target', 'corr']
features_corr_select = features_corr[features_corr['corr']>0.3] # 筛选出大于相关性大于0.3的特征
print(features_corr_select)
# 选出和target相关性大于0.3的特征
select_features = [col for col in features_corr_select['features_and_target'] if col not in ['target']]
new_train_data_corr_select = train_data_scaler[select_features+['target']] # 选中的列select_features加上target列组成的list
new_test_data_corr_select = test_data_scaler[select_features]

# 多重共线性分析
from statsmodels.stats.outliers_influence import variance_inflation_factor #多重共线性方差膨胀因子

#多重共线性(多个特征之间的相关性)
# 神仔的视频讲VIF算法 https://www.bilibili.com/video/BV1R44y1z7Zc 第P4节
# 张翔老师的视频讲VIF算法 https://www.youtube.com/watch?v=q_PjZOkPR90
# https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
new_numerical=['V0', 'V2', 'V3', 'V4', 'V5', 'V6', 'V10','V11',
                         'V13', 'V15', 'V16', 'V18', 'V19', 'V20', 'V22','V24','V30', 'V31', 'V37']
X=np.matrix(train_data_scaler[new_numerical])
VIF_list=[variance_inflation_factor(X, i) for i in range(X.shape[1])] # i用来索引是X中的第几个变量的VIF值
print('多重共线性的值')
print(VIF_list)

# PCA去除多重共线性 降维
from sklearn.decomposition import PCA   #主成分分析法

#PCA方法降维
#保持90%的信息
pca = PCA(n_components=0.9)
new_train_pca_90 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_90 = pca.transform(test_data_scaler)
new_train_pca_90 = pd.DataFrame(new_train_pca_90)
new_test_pca_90 = pd.DataFrame(new_test_pca_90)
new_train_pca_90['target'] = train_data_scaler['target']
print(new_train_pca_90.describe())
print('PCA方法降维 保持90%的信息')
print(train_data_scaler.describe())

# 再算一次多重共线性
X=new_train_pca_90.iloc[:,0:-1].values # 等同于np.matrix(new_train_pca_90.iloc[:,0:-1])
VIF_list=[variance_inflation_factor(X, i) for i in range(X.shape[1])] # i用来索引是X中的第几个变量的VIF值
print('PCA方法降维后 多重共线性的值')
print(VIF_list)


#PCA方法降维
#保留16个主成分
pca = PCA(n_components=0.95)
new_train_pca_16 = pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_16 = pca.transform(test_data_scaler)
new_train_pca_16 = pd.DataFrame(new_train_pca_16)
new_test_pca_16 = pd.DataFrame(new_test_pca_16)
new_train_pca_16['target'] = train_data_scaler['target']
new_train_pca_16.describe()
print('PCA方法降维 保持95%的信息')
print(train_data_scaler.describe())

# 再算一次多重共线性
X=new_train_pca_16.iloc[:,0:-1].values # 等同于np.matrix(new_train_pca_16.iloc[:,0:-1])
VIF_list=[variance_inflation_factor(X, i) for i in range(X.shape[1])] # i用来索引是X中的第几个变量的VIF值
print('PCA方法降维后 多重共线性的值')
print(VIF_list)


# 线性回归模型
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.neighbors import KNeighborsRegressor  #K近邻回归
from sklearn.tree import DecisionTreeRegressor     #决策树回归
from sklearn.ensemble import RandomForestRegressor #随机森林回归
from sklearn.svm import SVR  #支持向量回归
import lightgbm as lgb #lightGbm模型
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split # 切分数据
from sklearn.metrics import mean_squared_error #评价指标

# 切分训练数据和线下验证数据
#采用 pca 保留16维特征的数据
new_train_pca_16 = new_train_pca_16.fillna(0)
train = new_train_pca_16[new_test_pca_16.columns]
target = new_train_pca_16['target']

# 切分数据 训练数据80% 验证数据20%
train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.2,random_state=0)


# 给予不同的数据量，查看模型的学习效果
train_score = []
test_score = []
for i in range(10, len(train_data) + 1, 10): # start=10,stop=len(train_data) + 1,step=10
    lin_reg = LinearRegression()
    lin_reg.fit(train_data[:i], train_target[:i])
    # LinearRegression().fit(X_train[:i], y_train[:i])

    # 查看模型的预测情况：两种，模型基于训练数据集预测的情况(可以理解为模型拟合训练数据集的情况)，模型基于测试数据集预测的情况
    # 此处使用 lin_reg.predict(X_train[:i])，为训练模型的全部数据集
    y_train_predict = lin_reg.predict(train_data[:i])
    train_score.append(mean_squared_error(train_target[:i], y_train_predict))

    y_test_predict = lin_reg.predict(test_data)
    test_score.append(mean_squared_error(test_target, y_test_predict))

# np.sqrt(train_score)：将列表 train_score 中的数开平方
plt.plot([i for i in range(1, len(train_score) + 1)], train_score, label='train')
plt.plot([i for i in range(1, len(test_score) + 1)], test_score, label='test')

# plt.legend()：显示图例（如图形的 label）；
plt.legend()
plt.show()

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# 使用sklearn中自带的库，定义绘制模型学习曲线函数
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(train_scores_mean)
    print(test_scores_mean)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



# 多元线性回归模型
clf = LinearRegression()
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("LinearRegression:   ", score)

# 绘制线性回归模型学习曲线
X = train_data.values
y = train_target.values
title = r"LinearRegression"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = LinearRegression()  # 建模
plot_learning_curve(estimator, title, X, y, ylim=(0.5, 0.8), cv=cv, n_jobs=1)

# K近邻回归
clf = KNeighborsRegressor(n_neighbors=8) # 最近8个
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("KNeighborsRegressor:   ", score)

# 绘制K近邻回归学习曲线
X = train_data.values
y = train_target.values
title = r"KNeighborsRegressor"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = KNeighborsRegressor(n_neighbors=8)  # 建模
plot_learning_curve(estimator, title, X, y, ylim=(0.3, 0.9), cv=cv, n_jobs=1)

# 决策树回归
clf = DecisionTreeRegressor()
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("DecisionTreeRegressor:   ", score)

# 绘制决策树学习曲线
X = train_data.values
y = train_target.values
title = r"DecisionTreeRegressor"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = DecisionTreeRegressor()  # 建模
plot_learning_curve(estimator, title, X, y, ylim=(0.1, 1.3), cv=cv, n_jobs=1)

# 随机森林回归
clf = RandomForestRegressor(n_estimators=200) # 200棵树模型
clf.fit(train_data, train_target)
score = mean_squared_error(test_target, clf.predict(test_data))
print("RandomForestRegressor:   ", score)

# 绘制随机森林回归学习曲线
X = train_data.values
y = train_target.values
# 随机森林
title = r"RandomForestRegressor"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = RandomForestRegressor(n_estimators=200)  # 建模
plot_learning_curve(estimator, title, X, y, ylim=(0.4, 1.0), cv=cv, n_jobs=1)


# lgb模型回归
# lgb回归模型
clf = lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type='gbdt',
        random_state=2019,
        objective='regression',
    )

# 训练模型
clf.fit(
        X=train_data, y=train_target,
        eval_metric='MSE',
        verbose=50
    )

score = mean_squared_error(test_target, clf.predict(test_data))
print("lightGbm:   ", score)

# 绘制lgb回归学习曲线
X = train_data.values
y = train_target.values

# lgb回归
title = r"LGBMRegressor"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = lgb.LGBMRegressor(
    learning_rate=0.01,
    max_depth=-1,
    n_estimators=100,
    boosting_type='gbdt',
    random_state=2019,
    objective='regression'
)  # 建模

plot_learning_curve(estimator, title, X, y, ylim=(0.4, 1.0), cv=cv, n_jobs=1)
