import pandas as pd
sleep_data = pd.read_csv('sleep_health.csv')

# 查看数据大致情况
print(sleep_data.shape)
sleep_data.head()