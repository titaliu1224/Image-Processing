import cv2
import matplotlib.pyplot as plt
import numpy as np


# mouse click to find notch points
def add_notch_point(event, x, y, flags, img):
    # if button is clicked, mark the point
    if event == cv2.EVENT_LBUTTONDOWN:
        print("added nothch point at: ", x, y)
        # draw a circle
        cv2.circle(img, (x, y), 20, 0, -1)

# Read image
original_img = cv2.imread(".\\image4.png", cv2.IMREAD_GRAYSCALE)

# show original image
plt.subplot(2, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# get optimal size for dft
dft_M = cv2.getOptimalDFTSize(original_img.shape[0])
dft_N = cv2.getOptimalDFTSize(original_img.shape[1])
# add zero padding
dft_A = np.zeros((dft_M, dft_N), dtype=np.float32)
dft_A[:original_img.shape[0], :original_img.shape[1]] = original_img

# make two channel for dft (real and imaginary)
planes = [dft_A, np.zeros(dft_A.shape, np.float32)]
dft_A = cv2.merge(planes)
cv2.dft(dft_A, dft_A)
cv2.split(dft_A, planes)
#compute the magnitude
magnitude = cv2.magnitude(planes[0], planes[1])
magnitude = np.log(magnitude + 1)

# shift to center
cx = int(dft_A.shape[1] / 2)
cy = int(dft_A.shape[0] / 2)
q0 = magnitude[0:cy, 0:cx]
q1 = magnitude[0:cy, cx:dft_A.shape[1]]
q2 = magnitude[cy:dft_A.shape[0], 0:cx]
q3 = magnitude[cy:dft_A.shape[0], cx:dft_A.shape[1]]
# swap diagonal quadrants
tmp = np.copy(q0)
magnitude[0:cy, 0:cx] = q3
magnitude[cy:dft_A.shape[0], cx:dft_A.shape[1]] = tmp
tmp = np.copy(q1)
magnitude[0:cy, cx:dft_A.shape[1]] = q2
magnitude[cy:dft_A.shape[0], 0:cx] = tmp
cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)

# show frequency domain image after shift
plt.subplot(2, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title("Frequency Domain Image")
plt.axis("off")

# mouse click to find notch points
notch_points_img = np.ones(magnitude.shape, dtype=np.uint8)
cv2.namedWindow('Frequency Domain Image')
cv2.setMouseCallback('Frequency Domain Image', add_notch_point, notch_points_img)
cv2.imshow('Frequency Domain Image', magnitude)
cv2.waitKey(0)

# show notch points
plt.subplot(2, 2, 3)
plt.imshow(notch_points_img, cmap='gray')
plt.title("Notch Points")
plt.axis("off")

print(notch_points_img.shape)
# swap notch points to match dft_A
tmp = np.copy(notch_points_img[0:cy, 0:cx])
notch_points_img[0:cy, 0:cx] = notch_points_img[cy:dft_A.shape[0], cx:dft_A.shape[1]]
notch_points_img[cy:dft_A.shape[0], cx:dft_A.shape[1]] = tmp
tmp = np.copy(notch_points_img[0:cy, cx:dft_A.shape[1]])
notch_points_img[0:cy, cx:dft_A.shape[1]] = notch_points_img[cy:dft_A.shape[0], 0:cx]
notch_points_img[cy:dft_A.shape[0], 0:cx] = tmp
# apply notch filter
planes[0] = planes[0] * notch_points_img
planes[1] = planes[1] * notch_points_img
dftB = cv2.merge(planes)
# inverse dft_B
cv2.idft(dftB, dftB)
cv2.split(dftB, planes)
# get magnitude
inverse_img = cv2.magnitude(planes[0], planes[1])
# normalize to 0~255
cv2.normalize(inverse_img, inverse_img, 0, 255, cv2.NORM_MINMAX)
# convert to 8 bit unsigned integer
inverse_img = inverse_img.astype(np.uint8)

# show recovered image
plt.subplot(2, 2, 4)
plt.imshow(inverse_img, cmap='gray')
plt.title("Converted Image")
plt.axis("off")

plt.show()