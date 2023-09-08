from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/airline_Passengers.csv")
df.set_index('Period', inplace=True)
df.index = pd.to_datetime(df.index)
data = df["Passengers"]
data.plot(color='red', linestyle='solid')
plt.ylabel('Passengers')
plt.show()
# 将数据分解成Trend，Seasonal，和残差(Residual)
seasonal_decomp = seasonal_decompose(data, model="additive")
seasonal_decomp.plot()
plt.show()
