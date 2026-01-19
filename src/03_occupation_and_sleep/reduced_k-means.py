# 保留的数值特征（精简版）
essential_numeric = [
    'Sleep Duration',
    'Quality of Sleep',
    'Stress Level',
    'Physical Activity Level'
]

# 保留的分类特征（关键维度）
essential_categorical = [
    'Occupation',            # 职业类型（重要分组变量）
    'BMI Category'           # BMI分类（健康状态核心指标）
]

# 创建精简数据集
reduced_data = sleep_data[essential_numeric + essential_categorical].copy()

reduced_data_encoded = pd.get_dummies(
    reduced_data,
    columns=essential_categorical,
    drop_first=False
)

# 4. 获取编码后的分类特征列名
occupation_cols = [col for col in reduced_data_encoded.columns
                   if col.startswith('Occupation_')]
bmi_cols = [col for col in reduced_data_encoded.columns
            if col.startswith('BMI Category_')]

essential_features = essential_numeric + occupation_cols + bmi_cols

print("精简后特征列表：")
print(essential_features)

essential_numeric_data = reduced_data_encoded[essential_numeric]
essential_numeric_processed = scaler.fit_transform(essential_numeric_data)

# 使用新的分类特征列名
essential_categorical_data = reduced_data_encoded[occupation_cols + bmi_cols]  # 获取分类数据列
essential_categorical_processed = normalizer.fit_transform(essential_categorical_data)

processed_data = np.hstack([essential_numeric_processed, essential_categorical_processed])
print(f"合并后数据形状: {processed_data.shape}")

#确定最佳K值（肘部法则+轮廓系数）

# 肘部法则（K=1到11）考虑到有11个职业
wcss = []
for k in range(1, 12):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(final_data)
    wcss.append(kmeans.inertia_)

# 3.2 可视化肘部法则
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 12), wcss, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.grid(True)

k_range = range(2,10)
sil_scores = []

print("Silhouette scores for different k values:")
print("k\tSilhouette Score")
print("-" * 25)

for k in k_range:
    # Train k-means model
    kmeans_temp = KMeans(n_clusters=k, random_state=0, n_init='auto')
    cluster_labels = kmeans_temp.fit_predict(processed_data)
    sil_score = silhouette_score(processed_data, cluster_labels)
    sil_scores.append(sil_score)
    print(f"{k}\t{sil_score:.4f}")

#最佳k值
best_k = k_range[np.argmax(sil_scores)]
best_sil_score = max(sil_scores)
print(f"\n根据轮廓系数，最佳聚类数 K = {best_k} (轮廓系数 = {best_sil_score:.4f})")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_range, sil_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score vs Number of Clusters', fontsize=14)
plt.grid(True, alpha=0.3)

# 标记最佳k值位置
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.text(best_k + 0.1, best_sil_score - 0.01, f'BEST k={best_k}',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
         fontsize=10)

plt.tight_layout()
plt.show()

#选择k=6尝试聚类分析
k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(processed_data)

reduced_data['Cluster'] = cluster_labels
plt.figure(figsize=(15, 10))

#睡眠时长簇分布
plt.subplot(1, 1, 1)
sns.boxplot(x='Cluster', y='Sleep Duration', data=reduced_data)
plt.title('Sleep Duration Distribution by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Sleep Duration (hours)', fontsize=12)
plt.axhline(y=7, color='r', linestyle='--', label='Healthy Sleep Threshold')
plt.legend()

#睡眠质量簇分布
plt.subplot(1, 1, 1)
sns.boxplot(x='Cluster', y='Quality of Sleep', data=reduced_data)
plt.title('Sleep Quality Distribution by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Sleep Quality (1-10 scale)', fontsize=12)
plt.axhline(y=7.5, color='r', linestyle='--', label='Good Sleep Threshold')
plt.legend()

plt.subplot(1, 1, 1)
sns.scatterplot(
    x='Stress Level',
    y='Quality of Sleep',
    data=reduced_data,
    hue='Cluster',
    palette='viridis',
    s=80,
    alpha=0.7
)
plt.title('Stress Level vs Sleep Quality', fontsize=16)
plt.xlabel('Stress Level (1-10)', fontsize=14)
plt.ylabel('Sleep Quality (1-10)', fontsize=14)

# 添加回归线
sns.regplot(
    x='Stress Level',
    y='Quality of Sleep',
    data=reduced_data,
    scatter=False,
    color='red',
    line_kws={'linestyle': '--', 'alpha': 0.7}
)

# 计算职业在簇中的分布
occupation_cluster_dist = reduced_data.groupby(['Occupation', 'Cluster']).size().unstack(fill_value=0)
occupation_cluster_percent = occupation_cluster_dist.div(occupation_cluster_dist.sum(axis=1), axis=0) * 100

# 分析每个职业的簇分布
for occupation in occupation_cluster_percent.index:
    cluster_dist = occupation_cluster_percent.loc[occupation]
    main_clusters = cluster_dist[cluster_dist > 40]  # 主要分布簇(占比40%以上)
    cluster_str = "，".join([f"簇 {cluster}({perc:.1f}%)" for cluster, perc in main_clusters.items()])
    total_count = occupation_cluster_dist.loc[occupation].sum()

    print(f"职业: {occupation} (总样本数: {total_count})")
    print(f"  主要分布: {cluster_str if cluster_str else '分布较分散，无主导簇'}")

    print("  各簇分布百分比:")
    for cluster, percentage in cluster_dist.sort_values(ascending=False).items():
        print(f"    簇 {cluster}: {percentage:.1f}%")

    # 确定代表簇
    representative_cluster = cluster_dist.idxmax()
    rep_percentage = cluster_dist.max()

    # 判断代表强度
    if rep_percentage > 30:
        print(f"  ** 该职业主要由簇 {representative_cluster} 代表 ({rep_percentage:.1f}%)")
    elif rep_percentage > 25:
        print(f"  * 簇 {representative_cluster} 是该职业的最大分布簇 ({rep_percentage:.1f}%)")
    else:
        print(f"  注意: 该职业分布较分散，没有明显的代表簇 (最高占比 {rep_percentage:.1f}%)")

    print("-" * 90)

# 绘制职业在簇中的热力图
plt.figure(figsize=(16, 12))
sns.heatmap(
    occupation_cluster_percent.T,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': ' (%)'}
)
plt.title('Occupation Distribution per Cluster', fontsize=16)
plt.xlabel('Occupation', fontsize=14)
plt.ylabel('Cluster', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.subplot(1, 1, 1)
sns.boxplot(
    x='Occupation',
    y='Sleep Duration',
    data=reduced_data,
    palette='Set3',
    order=reduced_data.groupby('Occupation')['Sleep Duration'].median().sort_values().index
)
plt.title('Sleep Duration by Occupation', fontsize=16)
plt.xlabel('Occupation', fontsize=14)
plt.ylabel('Sleep Duration (hours)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=7, color='r', linestyle='--', label='Healthy Sleep Threshold')
plt.legend()