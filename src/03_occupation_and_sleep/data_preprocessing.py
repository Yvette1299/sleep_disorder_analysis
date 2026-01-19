import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sleep_data = pd.read_csv('sleep_health.csv')

#对分类变量进行独热编码
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
sleep_data_encoded = pd.get_dummies(sleep_data, columns=categorical_cols, drop_first=False)

#移除原始血压列和Person ID
sleep_data_encoded.drop(columns=['Blood Pressure', 'Person ID'], inplace=True,axis=1)

# 合并BMI Category_Normal和BMI Category_Normal Weight
sleep_data_encoded['BMI Category_Normal'] = sleep_data_encoded['BMI Category_Normal'] | sleep_data_encoded['BMI Category_Normal Weight']

# 删除多的
sleep_data_encoded.drop(columns=[ 'BMI Category_Normal Weight'], inplace=True, axis=1)

# 计算整个数据集的相关系数矩阵
corr_matrix = sleep_data_encoded.corr()
sleep_data_encoded.corr()

# 关系矩阵热力图分析
plt.figure(figsize=(15, 12))
corr_matrix = sleep_data_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Sleep Disorder Correlation Matrix')
plt.tight_layout()
plt.show()

#数值特征
numeric_cols = [
    'Sleep Duration',
    'Quality of Sleep',
    'Physical Activity Level',
    'Stress Level',
    'Heart Rate',
    'Daily Steps'
]
#分类特征
categorical_features = sleep_data_encoded.columns.difference(numeric_cols+ ['Age']).tolist()

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,Normalizer

#对数值特征做标准化
scaler = StandardScaler()
numeric_data = sleep_data_encoded[numeric_cols]  # 获取数值数据列
numeric_processed = scaler.fit_transform(numeric_data)

#分类特征做L2正则化
normalizer = Normalizer(norm='l2')
categorical_data = sleep_data_encoded[categorical_features]  # 获取分类数据列
categorical_processed = normalizer.fit_transform(categorical_data)

#合并所有特征（AI）
final_data = np.hstack([numeric_processed, categorical_processed])
print(f"合并后数据形状: {final_data.shape}")