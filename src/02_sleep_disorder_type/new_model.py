# 根据结果再次调节参数范围
custom_ranges = {
    'max_depth': [5, 6, 7, 9, 10, None],
    'min_samples_split': [5, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7]
}

best_custom_params, best_custom_score, all_results = custom_parameter_tuning(
    X_train, y_train, custom_ranges
)

# 用调节后的参数重新跑模型
new_tree_model = DecisionTreeClassifier(
    max_depth=9,
    min_samples_split=5,
    min_samples_leaf=1,
    criterion='gini', 
    max_features='sqrt',
    random_state=42
)

# 训练模型
new_tree_model.fit(X_train, y_train)

# 预测
new_y_pred_tree = new_tree_model.predict(X_test)

# 评估模型表现
print(f"Accuracy: {accuracy_score(y_test, new_y_pred_tree):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, new_y_pred_tree, target_names=['None', 'Insomnia', 'Sleep Apnea']))

# 绘制模型的Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['None', 'Insomnia', 'Sleep Apnea'],
                yticklabels=['None', 'Insomnia', 'Sleep Apnea'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(y_test, new_y_pred_tree, 'DicisionTree')