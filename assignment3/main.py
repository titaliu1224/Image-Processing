import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
original_img = cv2.imread(".\\image3.png", cv2.IMREAD_GRAYSCALE)
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
plt.title("Frequency Domain Image After Shift")
plt.axis("off")

# get phase image
phase = cv2.phase(planes[0], planes[1], angleInDegrees=True)
cv2.normalize(phase, phase, 0, 1, cv2.NORM_MINMAX)
# convert to 8 bit unsigned integer
phase = phase * 255
phase = phase.astype(np.uint8)
# show phase image
plt.subplot(2, 2, 3)
plt.imshow(phase, cmap='gray')
plt.title("Phase Image")
plt.axis("off")

# get inverse dft
cv2.idft(dft_A, dft_A)
cv2.split(dft_A, planes)
# get inverse image
inverse_img = cv2.magnitude(planes[0], planes[1])
# normalize to 0~255
cv2.normalize(inverse_img, inverse_img, 0, 255, cv2.NORM_MINMAX)
# convert to 8 bit unsigned integer
inverse_img = inverse_img.astype(np.uint8)
# show inverse image
plt.subplot(2, 2, 4)
plt.imshow(inverse_img, cmap='gray')
plt.title("Inverse Image")
plt.axis("off")

plt.show()


