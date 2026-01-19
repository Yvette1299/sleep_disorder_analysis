# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)