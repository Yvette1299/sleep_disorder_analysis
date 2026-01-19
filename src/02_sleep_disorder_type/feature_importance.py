# 决策树的特征重要性排序
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': new_tree_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Decision Tree Feature Importance Ranking：")
print(feature_importance.head(5))

# 可视化
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(5), x='importance', y='feature')
plt.title('Top 5 important features')
plt.tight_layout()
plt.show()