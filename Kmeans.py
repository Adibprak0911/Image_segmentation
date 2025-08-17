import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


img = cv2.imread('image.jpg')
H, W = img.shape[:2]
features = []
for y in range(H):
    for x in range(W):
        R, G, B = img[y, x] / 255.0
        x_norm = x / W
        y_norm = y / H
        features.append([R, G, B, x_norm, y_norm])
features = np.array(features)

rgb_weight = 1.0    
xy_weight = 0.4     
weights = np.array([rgb_weight, rgb_weight, rgb_weight, xy_weight, xy_weight])
features_weighted = features * weights

max_K = 10
wcss_scores = []
labels_list = []

for K in range(1, max_K + 1):
    kmeans = KMeans(n_clusters=K, n_init=10, max_iter=300)
    kmeans.fit(features_weighted)
    wcss_scores.append(kmeans.inertia_)
    labels_list.append(kmeans.labels_)

initial_drop = abs(wcss_scores[0] - wcss_scores[1])
optimal_K = max_K
for K in range(2, len(wcss_scores)):
    wcss_drop = abs(wcss_scores[K - 1] - wcss_scores[K])
    if initial_drop == 0 or wcss_drop / initial_drop < 0.10:
        optimal_K = K
        break

best_labels = labels_list[optimal_K - 1]
random_colors = np.random.randint(0, 256, (optimal_K, 3))
segmented_img = random_colors[best_labels].reshape(H, W, 3).astype(np.uint8)

cv2.imwrite('clustered_output_weighted.png', segmented_img)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].plot(range(1, len(wcss_scores) + 1), wcss_scores, marker='o')
axs[0].set_xlabel('K (Number of Clusters)')
axs[0].set_ylabel('Sum of Squared Distances')
axs[0].set_title('Elbow Method (Weighted Features)')
axs[0].grid(True)
axs[0].text(0.95, 0.95, f'Optimal K: {optimal_K}', verticalalignment='top', horizontalalignment='right',
            transform=axs[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
axs[1].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
axs[1].set_title(f"Clustered Image (K={optimal_K})")
axs[1].axis('off')
plt.tight_layout()
plt.show()
