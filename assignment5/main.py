import cv2
import matplotlib.pyplot as plt
import numpy as np

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']

image_count = 0
for image in images:

    print("Now processing: ", image)

    # read image
    original_img = cv2.imread(image)
    # show original image
    plt.subplot(3, 3, image_count * 3 + 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # convert to HSV color space
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    # convert to YCrCb color space
    ycbcr_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)

    # create skin upper and lower bounds
    lower_skin = np.array([0, 50, 0], dtype=np.uint8)
    upper_skin = np.array([17, 170, 255], dtype=np.uint8)
    lower_skin_ycbcr = np.array((0, 135, 85), dtype=np.uint8)
    upper_skin_ycbcr = np.array((255, 180, 135), dtype=np.uint8)

    # find skin color in the image
    skin_mask_hsv = cv2.inRange(hsv_img, lower_skin, upper_skin)
    skin_mask_ycbcr = cv2.inRange(ycbcr_img, lower_skin_ycbcr, upper_skin_ycbcr)
    skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycbcr)
    # show skin mask
    plt.subplot(3, 3, image_count * 3 + 2)
    plt.imshow(skin_mask, cmap="gray")
    plt.title("Skin Mask")
    plt.axis("off")

    # make skin region red
    skin_region = original_img.copy()
    skin_region[skin_mask != 0] = [0, 0, 255]
    # show skin region
    plt.subplot(3, 3, image_count * 3 + 3)
    plt.imshow(cv2.cvtColor(skin_region, cv2.COLOR_BGR2RGB))
    plt.title("Skin Region")
    plt.axis("off")

    image_count += 1

plt.show()
