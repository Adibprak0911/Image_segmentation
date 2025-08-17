import cv2
import numpy as np
import matplotlib.pyplot as plt

def euclidean_squared(a, b):
    return np.sum((a - b)**2)

def hard_kmeans(features, K, max_iter=10):
    N, D = features.shape
    centroids = features[np.random.choice(N, K, replace=False)]
    labels = np.zeros(N, dtype=int)
    for it in range(max_iter):
        for i in range(N):
            distances = [euclidean_squared(features[i], centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        for k in range(K):
            assigned = features[labels == k]
            if len(assigned) > 0:
                centroids[k] = np.mean(assigned, axis=0)
            else:
                centroids[k] = features[np.random.choice(N)]
    inertia = 0.0
    for i in range(N):
        inertia += euclidean_squared(features[i], centroids[labels[i]])
    return labels, centroids, inertia

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

# Assign weights: R, G, B have higher weights, x, y have lower
rgb_weight = 1.0
xy_weight = 0.4
weights = np.array([rgb_weight, rgb_weight, rgb_weight, xy_weight, xy_weight])
features_weighted = features * weights

max_K = 10
scores = []
best_labels = None
best_centroids = None
optimal_K = 1

for K in range(1, max_K+1):
    labels, centroids, inertia = hard_kmeans(features_weighted, K, max_iter=15)
    scores.append(inertia)
    if K > 2:
        initial_change = abs(scores[1] - scores[0])
        change = abs(scores[K-1] - scores[K-2])
        if initial_change != 0 and change < 0.10 * initial_change:
            optimal_K = K - 1
            best_labels = prev_labels
            best_centroids = prev_centroids
            break
    prev_labels, prev_centroids = labels, centroids

if best_labels is None or best_centroids is None:
    optimal_K = max_K
    best_labels, best_centroids = labels, centroids

random_colors = np.random.randint(0, 256, (optimal_K, 3))
segmented_img = random_colors[best_labels].reshape(H, W, 3).astype(np.uint8)

cv2.imwrite('clustered_output_hardcoded_weighted.png', segmented_img)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].plot(range(1, len(scores)+1), scores, marker='o')
axs[0].set_xlabel('K (Number of Clusters)')
axs[0].set_ylabel('Total L2 Score')
axs[0].set_title('K-means Elbow Method (Weighted Features)')
axs[0].grid(True)
axs[0].text(0.95, 0.95, f'Optimal K: {optimal_K}', verticalalignment='top', horizontalalignment='right',
            transform=axs[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
axs[1].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
axs[1].set_title(f"Clustered Image (K={optimal_K})")
axs[1].axis('off')
plt.tight_layout()
plt.show()
