def custom_parameter_tuning(X_train, y_train, custom_ranges=None):
    
    best_score = 0
    best_params = {}
    results = []
    
    # 遍历所有参数组合
    from itertools import product
    param_combinations = list(product(*custom_ranges.values()))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, combination in enumerate(param_combinations):
        params = dict(zip(custom_ranges.keys(), combination))
        
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'params': params,
            'mean_score': test_accuracy,
            'train_accuracy': model.score(X_train, y_train)
        })

        if test_accuracy > best_score:
            best_score = test_accuracy
            best_params = params
        
    # 显示前5个最优参数组合
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_score', ascending=False)

    print(f"\nTop 5 Parameter Combinations:")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"{i+1}. Score: {row['mean_score']:.4f} - Params: {row['params']}")
    
    return best_params, best_score, results_df

# 自定义调参范围并运行函数
custom_ranges = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 8, 10, 15],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7]
}

best_custom_params, best_custom_score, all_results = custom_parameter_tuning(
    X_train, y_train, custom_ranges
)    

# 可视化参数调优结果（AI）
def visualize_parameter_tuning(results_df, best_params, best_score):
    
    # 设置绘图风格
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # 创建2x3的图形布局
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    axes = axes.flatten()
    
    # max_depth 对性能的影响
    depth_data = []
    for _, row in results_df.iterrows():
        depth = row['params'].get('max_depth', 20)  # 20 represents None
        depth_data.append({'max_depth': depth, 'score': row['mean_score']})
    
    depth_df = pd.DataFrame(depth_data)
    depth_grouped = depth_df.groupby('max_depth')['score'].agg(['mean', 'std']).reset_index()

    axes[0].errorbar(depth_grouped['max_depth'], depth_grouped['mean'], 
                    yerr=depth_grouped['std'], fmt='o-', capsize=5, linewidth=2, 
                    color='#A23B72', markersize=6)
    axes[0].set_xlabel('max_depth')
    axes[0].set_ylabel('Mean Accuracy')
    axes[0].set_title('Impact of max_depth on Performance')
    axes[0].grid(True, alpha=0.3)
    
    # 为max_depth图表添加数值标注
    for i, row in depth_grouped.iterrows():
        axes[0].annotate(f'{row["mean"]:.3f}', 
                        (row['max_depth'], row['mean']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=8)
    
    # min_samples_split 对性能的影响
    split_data = []
    for _, row in results_df.iterrows():
        split_val = row['params'].get('min_samples_split', 2)
        split_data.append({'min_samples_split': split_val, 'score': row['mean_score']})
    
    split_df = pd.DataFrame(split_data)
    split_grouped = split_df.groupby('min_samples_split')['score'].agg(['mean', 'std']).reset_index()
    
    axes[1].errorbar(split_grouped['min_samples_split'], split_grouped['mean'], 
                    yerr=split_grouped['std'], fmt='s-', color='green', capsize=5, 
                    linewidth=2, markersize=6)
    axes[1].set_xlabel('min_samples_split')
    axes[1].set_ylabel('Mean Accuracy')
    axes[1].set_title('Impact of min_samples_split on Performance')
    axes[1].grid(True, alpha=0.3)
    
    # 为min_samples_split图表添加数值标注
    for i, row in split_grouped.iterrows():
        axes[1].annotate(f'{row["mean"]:.3f}', 
                        (row['min_samples_split'], row['mean']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=8)
    
    # min_samples_leaf 对性能的影响
    leaf_data = []
    for _, row in results_df.iterrows():
        leaf_val = row['params'].get('min_samples_leaf', 1)
        leaf_data.append({'min_samples_leaf': leaf_val, 'score': row['mean_score']})

    leaf_df = pd.DataFrame(leaf_data)
    leaf_grouped = leaf_df.groupby('min_samples_leaf')['score'].agg(['mean', 'std']).reset_index()
    
    axes[2].errorbar(leaf_grouped['min_samples_leaf'], leaf_grouped['mean'], 
                    yerr=leaf_grouped['std'], fmt='^-', color='orange', capsize=5, 
                    linewidth=2, markersize=6)
    axes[2].set_xlabel('min_samples_leaf')
    axes[2].set_ylabel('Mean Accuracy')
    axes[2].set_title('Impact of min_samples_leaf on Performance')
    axes[2].grid(True, alpha=0.3)
    
    # 为min_samples_leaf图表添加数值标注
    for i, row in leaf_grouped.iterrows():
        axes[2].annotate(f'{row["mean"]:.3f}', 
                        (row['min_samples_leaf'], row['mean']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=8)
    
    # criterion对性能的影响
    criterion_data = []
    for _, row in results_df.iterrows():
        crit_val = row['params'].get('criterion', 'gini')
        criterion_data.append({'criterion': crit_val, 'score': row['mean_score']})
    
    criterion_df = pd.DataFrame(criterion_data)
    criterion_stats = criterion_df.groupby('criterion')['score'].describe()
    
    colors = ['lightblue', 'lightcoral']
    bars = axes[3].bar(range(len(criterion_stats)), criterion_stats['mean'], 
                      color=colors, alpha=0.7, edgecolor='black')
    
    # 添加误差线
    for i, (idx, row) in enumerate(criterion_stats.iterrows()):
        axes[3].errorbar(i, row['mean'], yerr=row['std'], fmt='k_', capsize=5, linewidth=2)

    axes[3].set_xticks(range(len(criterion_stats)))
    axes[3].set_xticklabels(criterion_stats.index)
    axes[3].set_ylabel('Mean Accuracy')
    axes[3].set_title('Impact of Splitting Criterion on Performance')
    
    # 在柱子上添加数值
    for bar, mean_val in zip(bars, criterion_stats['mean']):
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 热力图: max_depth vs min_samples_split
    heatmap_data = results_df.copy()
    heatmap_data['max_depth'] = heatmap_data['params'].apply(lambda x: x.get('max_depth', 20))
    heatmap_data['min_samples_split'] = heatmap_data['params'].apply(lambda x: x.get('min_samples_split', 2))
    
    pivot_table = heatmap_data.pivot_table(values='mean_score', 
                                         index='max_depth', 
                                         columns='min_samples_split', 
                                         aggfunc='mean')
    
    # 创建热力图
    im = axes[4].imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度和标签
    axes[4].set_xticks(range(len(pivot_table.columns)))
    axes[4].set_xticklabels(pivot_table.columns)
    axes[4].set_yticks(range(len(pivot_table.index)))
    axes[4].set_yticklabels(pivot_table.index)
    
    # 添加数值标注
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            axes[4].text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    axes[4].set_xlabel('min_samples_split')
    axes[4].set_ylabel('max_depth')
    axes[4].set_title('max_depth vs min_samples_split\nPerformance Heatmap')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[4], shrink=0.8)
    cbar.set_label('Mean Accuracy', rotation=270, labelpad=15)
    
    # 隐藏第6个子图
    axes[5].axis('off')
    
    return fig
    