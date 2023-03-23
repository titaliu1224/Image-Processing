---
title: "Python 和 OpenCV 的影像邊緣偵測"
date: 2023-03-10 19:30:00 +0800

tags: 
  - Python
  - OpenCV
  - image processing
  - edge detection
  - Sobel operators
  - Gaussian blur

mermaid: true

categories: [Python | 影像處理]

img_path: ../../assets/img/posts/image_crop_and_rotate
---

這是學校選修課的功課紀錄，同步發布於 [該課程 Blogger](https://yzucs362hw.blogspot.com/2023/03/s1091444-2.html) <br>

## 功課要求
撰寫一個程式，以灰階模式讀取一張圖像  `imread(path, IMREAD_GRAYSCALE)`
1. 利用 Sobel Operators 偵測並輸出邊緣成分圖 
2. 設計一個類似素描線條的自畫像圖案。

## 成果
![邊緣偵測與素描線條展示](https://github.com/titaliu1224/Image-Processing/blob/main/assignment2/result.png?raw=true)

## 開發環境

| OS         | Editor             | Language      | OpenCV       |
|------------|--------------------|---------------|--------------|
| Windows 10 | Visual Studio Code | Python 3.9.16 | OpenCV 4.5.4 |

## 實作
> [本次程式碼](https://github.com/titaliu1224/Image-Processing/blob/main/assignment2/main.py)


使用的 library 如下：

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np
```

### 1/ 讀取灰階圖片

`colored_img` 用以展示彩色的原圖，而 `cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)` 會以灰階模式讀入一張圖。

```py
file_name = ".\\fig.jpg"
colored_img = cv2.imread(file_name)
gray_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
```

和[上一個程式](/posts/image_crop_and_rotate/)不同，這次的圖片輸出如下： <br>
使用 `result_img` 儲存所有要顯示的圖片，一個個將他們貼到 `plt` 中，這裡帶入了 `plt.axis("off")` 隱藏 matplotlib 預設的 x 軸和 y 軸的刻度。

```py
result_img = [colored_img]

fig = plt.figure()
def show_img():
    for i in range(0, len(result_img)):
        image_rgb = cv2.cvtColor(result_img[i], cv2.COLOR_BGR2RGB)
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(image_rgb)
        plt.axis("off")

    plt.show()
```

### 2/ 邊緣成分圖

首先利用高斯模糊 (Gaussian blur) 去除雜訊（噪聲），使邊界更好檢測： <br>

```py
# 高斯模糊協助過濾雜訊
gray_img = cv2.GaussianBlur(gray_img,(3, 3), 0)
result_img.append(gray_img)
```

再利用索伯算子 (Sobel operaters) 提取 x 方向和 y 方向的邊界，之後將兩者的絕對值相加，獲得完整的邊緣成分圖。

```py
# 提取 x 方向和 y 方向的邊緣
edge_x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, 3)
edge_Y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, 3)
# 轉換為 unit8 （提取絕對值）
abs_x = cv2.convertScaleAbs(edge_x) 
abs_y = cv2.convertScaleAbs(edge_Y)
# 將兩者取絕對值相加，獲得完整影像
edge_all = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
result_img.append(edge_all)
```

關於高斯模糊和索伯算子，日後我想寫篇文章給他們。

### 3/ 素描線條畫像

這裡簡單的使用 `bitwise_not()` 將邊緣成分圖黑白相反，使其看起來像素描：

```py
sketch_img = cv2.bitwise_not(edge_all)
result_img.append(sketch_img)
```

最後使用 `show_img()` 展示成果。

## 總結

本篇使用**高斯模糊**去除雜訊、**索伯算子**提取邊界、`bitwise_not()`進行黑白反轉。 <br>
邊界提取是很實用的東西，例如[分辨一張圖畫是寶可夢還是數碼寶貝](https://youtu.be/_j9MVVcvyZI?t=327) 等。

## 參考資料

- [Day12-當自動駕駛遇見AI-索伯算子(Sobel Operator)](https://ithelp.ithome.com.tw/articles/10205752)
- [邊緣偵測 - 索伯算子 ( Sobel Operator )](https://medium.com/%E9%9B%BB%E8%85%A6%E8%A6%96%E8%A6%BA/%E9%82%8A%E7%B7%A3%E5%81%B5%E6%B8%AC-%E7%B4%A2%E4%BC%AF%E7%AE%97%E5%AD%90-sobel-operator-95ca51c8d78a)
- [Opencv学习----Opencv宏定义(CV_8U、CV_8S、CV_16U...)](https://blog.csdn.net/charce_you/article/details/99616021)
- [程式語言-看盤版面(上)-圖框教學](https://medium.com/%E5%8F%B0%E8%82%A1etf%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8-%E7%A8%8B%E5%BC%8F%E9%A1%9E/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80-%E7%9C%8B%E7%9B%A4%E7%89%88%E9%9D%A2-%E4%B8%8A-%E5%9C%96%E6%A1%86%E6%95%99%E5%AD%B8-5d1baf57f5a7)
- [在 Matplotlib 中隱藏座標軸、邊框和空白](https://www.delftstack.com/zh-tw/howto/matplotlib/hide-axis-borders-and-white-spaces-in-matplotlib/)