# Watershed algorithm
Watershed for simple image segmentation implemented with OpenCV
This script demonstrates automated image segmentation using the watershed algorithm, a powerful tool in computer vision for separating overlapping or touching objects in an image. The watershed algorithm is inspired by topological flooding—imagining each pixel as terrain, and "water" filling up basins until it meets at ridge lines, which become segmentation boundaries. This method is especially useful for partitioning complex regions, such as coins touching in a photograph.

## 1. Image Loading

The algorithm begins by loading the input image (coin.jpeg) and converting it to both RGB and grayscale formats.
If the image cannot be loaded, the script reports an error and exits.

## 2. Grayscale Visualization

The grayscale image is displayed as the starting point, simplifying further processing compared to color images.

## 3. Thresholding

Binary inverse thresholding is applied to the gray image, transforming it into a binary mask that highlights foreground objects.
This step separates coins (or other objects) from the background.

## 4. Morphological Dilation

The binary mask is dilated using a small kernel to enhance foreground areas and close small gaps within objects.
This helps unify object regions that might otherwise be fragmented due to noise.

## 5. Distance Transform

The distance transform measures the distance from every pixel to the nearest background pixel.
This map identifies object centers for more accurate segmentation.

## 6. Thresholding Distance Map

A binary mask is created from the distance-transformed image, highlighting core object regions.
These marked regions will serve as seeds for the subsequent watershed process.

## 7. Connected Components (Labeling)

The core areas are labeled using connected component analysis, assigning unique identifiers to different objects.

## 8. Watershed Segmentation

The watershed algorithm is applied, using the labeled core regions as markers.
It expands markers outward until boundaries between touching objects are established.

## Result

<img width="1274" height="1080" alt="image" src="https://github.com/user-attachments/assets/c65a07d4-d4a0-48ac-9df3-e25647bc5b65" />
<img width="1274" height="1086" alt="image" src="https://github.com/user-attachments/assets/ffb64fae-05ea-4230-bad6-3de6a8acfc93" />

# K-means clustering 

This project applies the K-means clustering algorithm for image segmentation, leveraging a weighted 5-dimensional Euclidean space. In this method, each pixel is represented by its RGB color values and (x, y) spatial coordinates, forming a feature vector: [R, G, B, x, y].

By clustering pixels in this 5D space, the algorithm groups together regions that are similar in both color and location, resulting in segments that are coherent in appearance and spatial structure. Weights are assigned to each feature component: the RGB channels are given higher importance to prioritize color similarity, while the spatial positions (x, y) have slightly lower weights, ensuring that neighboring pixels are considered together but color still dominates the segmentation.

Segmentation is performed by minimizing the sum of squared Euclidean distances between each pixel and its assigned cluster centroid in this weighted space. This approach yields segmentations that are both color-consistent and spatially smooth—ideal for tasks like object separation

## Input 

<img width="800" height="830" alt="image" src="https://github.com/user-attachments/assets/ca7bfe63-4e04-4c3d-8782-adc34096f563" />

## Post segmentation 

<img width="800" height="830" alt="image" src="https://github.com/user-attachments/assets/43e7ca37-1398-4722-94f7-b912c8b3ecef" />







