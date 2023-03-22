import cv2
import matplotlib.pyplot as plt
import numpy as np

file_name = ".\\fig.jpg"
colored_img = cv2.imread(file_name)
gray_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
result_img = [colored_img]

fig = plt.figure()
def show_img():
    image_rgb = []
    for i in result_img:
        image_rgb.append(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))

    for i in range(0, len(result_img)):
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(image_rgb[i])
        plt.axis("off")
        
    plt.show()


# 高斯模糊協助過濾雜訊
gray_img = cv2.GaussianBlur(gray_img,(3, 3), 0)
result_img.append(gray_img)
# 提取 x 方向和 y 方向的邊緣
edge_x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, 3)
edge_Y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, 3)
# 轉換為 unit8 （提取絕對值）
abs_x = cv2.convertScaleAbs(edge_x) 
abs_y = cv2.convertScaleAbs(edge_Y)
# 將兩者取絕對值相加，獲得完整影像
edge_all = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
result_img.append(edge_all)

sketch_img = cv2.bitwise_not(edge_all)
result_img.append(sketch_img)

show_img()

