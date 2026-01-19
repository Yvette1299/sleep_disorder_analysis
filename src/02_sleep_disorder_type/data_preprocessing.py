import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.gridspec import GridSpec

df = pd.read_csv('/Users/xieyuwei/04_Projects/dataanalysis/sleep_disorder/data/sleep_health.csv')

# 对睡眠障碍类型进行编码
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
disorder_mapping = {'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2}
df['Sleep Disorder_encoded'] = df['Sleep Disorder'].map(disorder_mapping)

# 对职业进行独热编码
occupation_encoded = pd.get_dummies(df['Occupation'], prefix='Occupation')
df = pd.concat([df, occupation_encoded], axis=1)

# 对BMI类别进行编码
bmi_mapping = {'Normal Weight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
df['BMI_encoded'] = df['BMI Category'].map(bmi_mapping)

# 对性别进行编码
df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

# 处理血压特征
# 拆分血压为收缩压和舒张压
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)

# 转换为数值类型
df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])

# 删除原始血压列
df = df.drop('Blood Pressure', axis=1)

# 选择用于建模的特征
feature_columns = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 
    'Physical Activity Level', 'Stress Level', 'Heart Rate', 
    'Daily Steps', 'Systolic_BP', 'Diastolic_BP', 'BMI_encoded', 'Gender_encoded'
] + list(occupation_encoded.columns)

# 创建特征数据集
X = df[feature_columns]
y = df['Sleep Disorder_encoded']

#查看处理后的数据
print(X.head(5))