## 功課要求

偵測輸入照片中的皮膚區域並將其標示出。
## 成果
![圖像處理](https://github.com/titaliu1224/Image-Processing/blob/main/assignment5/result.png?raw=true)
_程式完成後的執行結果，膚色區域以紅色標記_

## 開發環境

| OS         | Editor             | Language      | OpenCV       |
|------------|--------------------|---------------|--------------|
| Windows 10 | Visual Studio Code | Python 3.9.16 | OpenCV 4.5.4 |

## 實作
> [本次程式碼](https://github.com/titaliu1224/Image-Processing/blob/main/assignment4/main.py)

使用的 libraries 如下：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np
```

### 1/ 利用迴圈讀入三張圖片

建立一個儲存三張圖片路徑的 list ，使用迴圈搭配 `cv2.imread(filename)` 讀入圖片並顯示。 <br>
顯示圖片中有一點要注意， plt 使用的彩色圖片是 RGB ，而 OpenCV 讀入的圖片是以 BGR 編碼，所以必須透過 `cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)` 來轉換要顯示的圖片，不然就會出現三名藍色皮膚的人。

```py
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
```

### 2/ 轉換圖片色域並設定膚色範圍

RGB 色域容易受到光線等因素影響，導致難以判斷顏色是否為膚色，使用 `cv2.cvtColor(src, code)` 轉換至 HSV 色域後就能把色相、飽和度、明度分開看。

```py
# convert to HSV color space
hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
# convert to YCrCb color space
ycbcr_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)
```

創建 array 來儲存膚色的範圍，這是我採用的範圍：
  - H: 0 ~ 17
  - S: 50 ~ 170
  - V: 0 ~ 255
  - Y: 0 ~ 255
  - Cr: 135 ~ 180
  - Cb: 85 ~ 135

調整過後我仍舊無法避免有些非皮膚（頭髮、陰影等）處還是會被判別為膚色。

```py
 # create skin upper and lower bounds
lower_skin = np.array([0, 50, 0], dtype=np.uint8)
upper_skin = np.array([17, 170, 255], dtype=np.uint8)
lower_skin_ycbcr = np.array((0, 135, 85), dtype=np.uint8)
upper_skin_ycbcr = np.array((255, 180, 135), dtype=np.uint8)
```

### 3/ 提取膚色區域

使用 `cv2.inRange(src, lowerb, upperb)` 提取膚色區域，獲得一張和原尺寸相同大小的二值化 mask ，膚色區域為白色，其他地方為黑色，就如結果圖中第二個 column 顯示的那樣。

```py
# find skin color in the image
skin_mask_hsv = cv2.inRange(hsv_img, lower_skin, upper_skin)
skin_mask_ycbcr = cv2.inRange(ycbcr_img, lower_skin_ycbcr, upper_skin_ycbcr)
skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycbcr)
# show skin mask
plt.subplot(3, 3, image_count * 3 + 2)
plt.imshow(skin_mask, cmap="gray")
plt.title("Skin Mask")
plt.axis("off")
```

### 4/ 將原圖膚色區域標示為紅色

最後在 `skin_mask` 不為 0 的像素，更改原圖的顏色為紅色。

```py
# make skin region red
skin_region = original_img.copy()
skin_region[skin_mask != 0] = [0, 0, 255]
# show skin region
plt.subplot(3, 3, image_count * 3 + 3)
plt.imshow(cv2.cvtColor(skin_region, cv2.COLOR_BGR2RGB))
plt.title("Skin Region")
plt.axis("off")
```

迴圈的最後要 `+= 1` 和程式結束前要顯示圖片。

```py
  image_count += 1

plt.show()
```

## 參考資料

- [Human Skin Detection Using RGB, HSV and YCbCr Color Models](https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf)
- [OpenCV 探索之路（二十七）：皮膚檢測技術](https://www.cnblogs.com/skyfsm/p/7868877.html)
- [CHEREF-Mehdi/SkinDetection](https://github.com/CHEREF-Mehdi/SkinDetection)
