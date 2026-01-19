import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv('sleep_health.csv')

# 对睡眠障碍进行编码
data['Sleep Disorder'] = data['Sleep Disorder'].fillna('None')
sleep_disorder_map = {'None': 2, 'Insomnia': 1, 'Sleep Apnea': 0}
data['Sleep Disorder Encoded'] = data['Sleep Disorder'].map(sleep_disorder_map)

# 选择需要的特征列
features = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Sleep Disorder Encoded']
X = data[features]
y = data['Quality of Sleep']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)