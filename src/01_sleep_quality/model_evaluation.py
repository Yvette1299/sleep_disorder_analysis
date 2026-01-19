# 评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 计算MAPE（处理y_test=0的边缘情况，避免除以0）
mask = y_test != 0  # 过滤掉真实值为0的样本
if np.any(mask):
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100  # 转换为百分比
else:
    mape = np.nan  # 若所有y_test都为0，MAPE无意义，返回NaN
# 输出评估结果
print("模型评估结果:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"R² 分数: {r2:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%" if not np.isnan(mape) else "平均绝对百分比误差 (MAPE): 无意义（y_test全为0）")
print("\n特征权重 (系数):")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")