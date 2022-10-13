import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams["figure.figsize"] = (20,5)

# 包含第1天(d_1)到1913天(d_1913)的销量信息
sales = pd.read_csv("./data/b16/sales_train_validation.csv")
print('--------历史天销售数据-------')
print(sales[0:4])

calendar = pd.read_csv("./data/b16/calendar.csv")
print('--------日历数据-------')
print(calendar[0:10])

sell_prices = pd.read_csv("./data/b16/sell_prices.csv")
print('--------历史周价格数据-------')
print(sell_prices[0:10])

#选取了左右历史数据，其他列没有要
d_cols = [c for c in sales.columns if 'd_' in c] # sales data columns
x = sales[d_cols].copy()
print('--------历史数据-------')
print(x[0:10])

# 先来构造预测第1914天销量 需要的局部特征
target_day = 1914
#使用历史数据中最后的7天构造特征
local_range = 7
# 由于使用前1913天的数据预测第1914天，历史数据与预测目标的距离只有1天，因此predict_distance=1
# 如果使用前1913天的数据预测第1915天，则历史数据与预测目标的距离有2天，因此predict_distance=2，以此类推
predict_distance = 1

# def get_local_features(target_day, predict_distance):
#     local_features = pd.DataFrame()
#     for i in range(local_range):
#         local_features['l_' + str(i + 1)] = x['d_' + str(target_day - i - predict_distance)].astype(float)
#     l_cols = ['l_' + str(i + 1) for i in range(local_range)]
#     return local_features[l_cols]
# get_local_features(target_day, predict_distance)

# 滚动累计每一天的历史值
# d_cols[::-1] 是将所有的列翻转过来，将列倒着排一遍
# 为什么要用累计历史值呢？对于单天的历史值，或多或少都有些随机因素，具有较大的不确定性，例如某天天气不好，销量突然下降。
# 实际上，我们可以用连续几天的加和（或均值），用于减缓不确定性带来的影响。 这里用累计历史值
tx = x[d_cols[::-1]].cumsum(axis=1)
print('--------逆序历史数据-------')
print(tx[0:10])

# 这样我们就从历史序列里的最近的56天，构造出了上面的12个特征。
used_history_distances = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 42, 56]

def get_accumulated_features(target_day, predict_distance):
    long_term_features = pd.DataFrame()
    for distance in used_history_distances:
        long_term_features['la_' + str(distance)] = tx['d_' + str(target_day - distance - predict_distance + 1)].astype(float)
    la_cols = ['la_' + str(distance) for distance in used_history_distances]
    return long_term_features[la_cols]

temp = get_accumulated_features(target_day, predict_distance)
print('--------accumulated_features-------')
print(temp[0:10])
# 构造周期值，在56天历史值中，也就是过去8周的数据中，我们先取得和目标预测值同周期的历史数据
def get_period_sale(target_day, predict_distance):
    period = 7
    i_start = (predict_distance + period - 1) // period  #
    period_sale = pd.DataFrame()
    for i in range(8): # 这里i代表第几周
        # 找出同周期的天在哪一天。距离当天预测天第1周，第2周，第3周的数据分别在第几天，第几天，第几天
        cur_day = target_day - (i + i_start) * period # 分别时1907,1900,1893,1886,1879,1872,1865,1858
        period_sale['p_'+str(i + 1)] = x['d_' + str(cur_day)].astype(float)
    return period_sale

temp = get_period_sale(target_day, predict_distance)
print('--------get_period_sale-------')
print(temp[0:10])

# 周期特征我们也用累计值
def get_period_features(target_day, predict_distance):
    tx_period = get_period_sale(target_day, predict_distance)
    tx_period = tx_period.cumsum(axis=1)
    return tx_period

temp = get_period_features(target_day, predict_distance)
print('--------get_period_features累计的-------')
print(temp[0:10])
# 综上，以下是我们基于历史数据构造出的所有特征。

def get_history_features(target_day, predict_distance):
    return pd.concat([get_accumulated_features(target_day, predict_distance),
                      get_period_features(target_day, predict_distance)], axis=1)

temp = get_history_features(target_day, predict_distance)
print('--------历史数据构造出的所有特征------get_history_features-------')
print(temp[0:10])

# 日历特征 我们发现，每年的12月和1月前后，该商品的销量都会比其他月份高一些。而3月和4月前后的销量偏低
# 我们挑出历史数据中所有与目标预测日期所在月份相同的日期，并计算他们的销量平均值。
def get_same_month_mean_feature(target_day, predict_distance):
    calendar_history = calendar.loc[calendar.index < target_day - predict_distance] # 取所有目标天之前的历史日历数据
    target_date_month = int(calendar.iloc[target_day - 1]["month"]) # 找出预测天所在的月份
    same_month_dates = calendar_history.loc[calendar_history["month"].astype(int) == target_date_month]["d"] # 与预测天所在月份同月份的所有天数
    same_month_mean = x[same_month_dates].mean(axis=1)
    return pd.DataFrame(same_month_mean, columns=["same_month_mean"])

temp = get_same_month_mean_feature(target_day, predict_distance)
print('--------get_same_month_mean_feature-------')
print(temp[0:10])

# 事件特征
all_event_name_1 = np.unique(list(filter(lambda val: isinstance(val, str), calendar["event_name_1"].values)))
all_event_name_2 = np.unique(list(filter(lambda val: isinstance(val, str), calendar["event_name_2"].values)))
all_event_type_1 = np.unique(list(filter(lambda val: isinstance(val, str), calendar["event_type_1"].values)))
all_event_type_2 = np.unique(list(filter(lambda val: isinstance(val, str), calendar["event_type_2"].values)))
all_event_list = np.hstack([all_event_name_1, all_event_name_2, all_event_type_1, all_event_type_2])
def get_event_features(target_day, predict_distance):
    event_mean_price_table = pd.DataFrame()

    def add_event_mean_features(event_list, event_key, target_day, predict_distance):
        calendar_history = calendar.loc[calendar.index < target_day - predict_distance]
        for event in event_list:
            if event == "nan":
                continue
            str_event_col = np.array(list(map(lambda val: str(val), calendar_history[event_key].values)))
            event_d_cols = calendar_history.loc[str_event_col == event]["d"]
            event_mean_price_table[event + "_mean_price"] = x[event_d_cols].mean(axis=1)

    add_event_mean_features(all_event_name_1, "event_name_1", target_day, predict_distance)
    add_event_mean_features(all_event_name_2, "event_name_2", target_day, predict_distance)
    add_event_mean_features(all_event_type_1, "event_type_1", target_day, predict_distance)
    add_event_mean_features(all_event_type_2, "event_type_2", target_day, predict_distance)

    target_day_event_name_1 = calendar.iloc[target_day - 1]["event_name_1"]
    target_day_event_name_2 = calendar.iloc[target_day - 1]["event_name_2"]
    target_day_event_type_1 = calendar.iloc[target_day - 1]["event_type_1"]
    target_day_event_type_2 = calendar.iloc[target_day - 1]["event_type_2"]

    event_feature = pd.DataFrame(data=np.zeros([sales.shape[0], 0]))
    series_nan = pd.Series(np.ones(sales.shape[0]) * np.nan, dtype=np.float, index=np.arange(sales.shape[0]))
    event_feature["event_name_1_mean"] = series_nan if not isinstance(target_day_event_name_1, str) else \
        event_mean_price_table[target_day_event_name_1 + "_mean_price"]
    event_feature["event_name_2_mean"] = series_nan if not isinstance(target_day_event_name_2, str) else \
        event_mean_price_table[target_day_event_name_2 + "_mean_price"]
    event_feature["event_type_1_mean"] = series_nan if not isinstance(target_day_event_type_1, str) else \
        event_mean_price_table[target_day_event_type_1 + "_mean_price"]
    event_feature["event_type_2_mean"] = series_nan if not isinstance(target_day_event_type_2, str) else \
        event_mean_price_table[target_day_event_type_2 + "_mean_price"]
    return event_feature

temp = get_event_features(target_day, predict_distance)
print('--------get_event_features-------')
print(temp[0:10])

# 消费券特征
states = ["CA", "TX", "WI"]
def get_snap_feature(target_day, predict_distance):
    snap_feature = pd.DataFrame(data=np.zeros([sales.shape[0], 1]), columns=["snap_mean"])
    calendar_history = calendar.loc[calendar.index < target_day - predict_distance]
    for state in states:
        snap_col = "snap_" + state
        state_data_indices = np.array(list(map(lambda val: val.split("_")[3] == state, sales["id"].values)))
        snap_d_cols = calendar_history.loc[calendar_history[snap_col] == 1]["d"]
        no_snap_d_cols = calendar_history.loc[calendar_history[snap_col] == 0]["d"]
        state_snap_mean = sales.loc[state_data_indices][snap_d_cols].mean(axis=1)
        state_no_snap_mean = sales.loc[state_data_indices][no_snap_d_cols].mean(axis=1)
        if calendar.iloc[target_day - 1][snap_col] == 1:
            snap_feature.loc[state_data_indices, "snap_mean"] = state_snap_mean
        else:
            snap_feature.loc[state_data_indices, "snap_mean"] = state_no_snap_mean
    return snap_feature

temp = get_snap_feature(target_day, predict_distance)
print('--------get_snap_feature-------')
print(temp[0:10])

# 汇总所有的日历数据特征
def get_calendar_features(target_day, predict_distance):
    return pd.concat([get_same_month_mean_feature(target_day, predict_distance),
                      get_event_features(target_day, predict_distance),
                      get_snap_feature(target_day, predict_distance)], axis=1)

temp = get_calendar_features(target_day, predict_distance)
print('--------日历数据构造出的所有特征------get_calendar_features包括了get_same_month_mean_feature，get_event_features，get_snap_feature-------')
print(temp[0:10])

# 价格特征

# 汇总所有的特征
def get_all_features(target_day, predict_distance):
    return pd.concat([get_history_features(target_day, predict_distance),
                      get_calendar_features(target_day, predict_distance)], axis=1)

temp = get_all_features(target_day, predict_distance)
print('--------get_all_features-------')
print(temp[0:10])
