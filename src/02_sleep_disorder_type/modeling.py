# 对数据做标准化处理
scaler = StandardScaler()
numerical_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                    'Physical Activity Level', 'Stress Level', 'Heart Rate', 
                    'Daily Steps', 'Systolic_BP', 'Diastolic_BP']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train：{X_train.shape}, Test：{X_test.shape}")

# 决策树模型的初始参数
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# 训练模型
tree_model.fit(X_train, y_train)

# 预测
y_pred_tree = tree_model.predict(X_test)