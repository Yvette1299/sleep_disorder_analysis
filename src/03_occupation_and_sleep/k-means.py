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

#轮廓系数（在肘部候选范围内）
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


k_range = range(2, 10)
sil_scores = []

print("Silhouette scores for different k values:")
print("k\tSilhouette Score")
print("-" * 25)

for k in k_range:
    # Train k-means model
    kmeans_temp = KMeans(n_clusters=k, random_state=0, n_init='auto')
    cluster_labels = kmeans_temp.fit_predict(final_data)
    sil_score = silhouette_score(final_data, cluster_labels)
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