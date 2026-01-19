import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

# 9. 创建图形
plt.figure(figsize=(12, 5))

# 子图1: 预测值 vs 实际值散点图
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sleep Quality')
plt.ylabel('Predicted Sleep Quality')
plt.title('Predicted vs Actual\nR² = {:.4f}'.format(r2_score(y_test, y_pred)))
plt.grid(True, alpha=0.3)

# 子图2: 按特征重要性展示单个特征与目标变量的关系
plt.subplot(1, 2, 2)
most_important_feature_idx = np.argmax(np.abs(model.coef_))
feature_values = X_test[:, most_important_feature_idx]
plt.scatter(feature_values, y_test, alpha=0.6, label='Actual')
plt.scatter(feature_values, y_pred, alpha=0.6, label='Predicted', marker='x')
plt.xlabel(f'Scaled {features[most_important_feature_idx]}')
plt.ylabel('Sleep Quality')
plt.title(f'{features[most_important_feature_idx]} vs Sleep Quality')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()