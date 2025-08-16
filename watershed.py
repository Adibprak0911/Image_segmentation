import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def watershed():
    root = os.getcwd()
    img_path = os.path.join(root, 'watershed/coin.jpeg')
    img = cv.imread(img_path)
    if img is None:
        print("Image not found or cannot be loaded!")
        print("Tried to load:", img_path)
        return
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(231)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Gray Image")

    plt.subplot(232)
    _, imgThreshold = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY_INV)
    plt.imshow(imgThreshold, cmap='gray')
    plt.title("Thresholded")

    plt.subplot(233)
    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv.morphologyEx(imgThreshold, cv.MORPH_DILATE, kernel)
    plt.imshow(imgDilate, cmap='gray')
    plt.title("Dilated")

    plt.subplot(234)
    distTrans = cv.distanceTransform(imgDilate, cv.DIST_L2, 5)
    plt.imshow(distTrans, cmap='gray')
    plt.title("Distance Transform")

    plt.subplot(235)
    _, distThresh = cv.threshold(distTrans, 15, 255, cv.THRESH_BINARY)
    plt.imshow(distThresh, cmap='gray')
    plt.title("Dist. Thresh.")

    plt.subplot(236)
    distThresh_uint8 = np.uint8(distThresh)
    _, labels = cv.connectedComponents(distThresh_uint8)
    plt.imshow(labels, cmap='jet')
    plt.title("Connected Components (labels)")

    plt.figure()
    plt.subplot(121)
    labels = np.int32(labels)
    ws_labels = cv.watershed(imgRGB.copy(), labels)
    plt.imshow(ws_labels, cmap='jet')
    plt.title("Watershed Labels")

    plt.subplot(122)
    img_ws = imgRGB.copy()
    img_ws[ws_labels == -1] = [255, 0, 0]
    plt.imshow(img_ws)
    plt.title("Watershed Result")

    plt.show()

if __name__ == '__main__':
    watershed()
