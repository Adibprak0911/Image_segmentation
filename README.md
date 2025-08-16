# Watershed-algorithm
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



