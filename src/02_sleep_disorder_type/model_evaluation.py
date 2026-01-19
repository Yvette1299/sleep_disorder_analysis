# 评估模型表现
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_tree, target_names=['None', 'Insomnia', 'Sleep Apnea']))

# 绘制模型的混淆矩阵
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

plot_confusion_matrix(y_test, y_pred_tree, 'DicisionTree')