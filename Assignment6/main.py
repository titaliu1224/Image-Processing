import cv2, os
import matplotlib.pyplot as plt
import numpy as np

img_path = ["img1.bmp", "img2.bmp", "img3.bmp"]
compress_path = ["img1.dat", "img2.dat", "img3.dat"]

# 壓縮單張圖片
def compress(img_path, compress_file):
    img = cv2.imread(img_path)
    with open(compress_file, "w") as file:
        # 記下圖片大小
        img_size = [img.shape[0], img.shape[1]]
        file.write(str(img_size[0]) + ", " + str(img_size[1]) + "\n")

        # 三個 channel 分開跑
        for channel in range(0, 3):
            img_channel = img[:, :, channel]
            img_flat = img_channel.flatten()

            last_pixel = 0
            pixel_count = 0

            # 遍歷每個 pixel
            for pixel in img_flat:
                if pixel == last_pixel:
                    pixel_count += 1
                else:
                    file.write(str(last_pixel) + ", " + str(pixel_count) + ", ")
                    pixel_count = 1
                    last_pixel = pixel

            # 存入最後一筆資料
            file.write(str(last_pixel) + ", " + str(pixel_count) + "\n")
    
    original_size = os.path.getsize(img_path)
    compress_size = os.path.getsize(compress_file)
    compress_ratio = original_size / compress_size
    
    print(f"| {compress_file:10} | {str(original_size):7} bytes | {str(compress_size):8} bytes | {str(round(compress_ratio, 5)):17} |")

    return compress_ratio

# 解壓縮單張圖片
def decompress(compress_file):
    with open(compress_file, "r") as file:
        # 讀取圖片大小
        img_size = file.readline().rstrip('\n').split(", ")
        # 讀取剩下的資料
        row_data = file.read().split("\n")
        data = [row_data[i].split(", ") for i in range(0, len(row_data))]
        # print(type(data[0]))
        # print(data[0])
        # 創建畫布
        image_bgr = [np.zeros(int(img_size[0]) * int(img_size[1]), dtype=np.uint8), np.zeros(int(img_size[0]) * int(img_size[1]), dtype=np.uint8), np.zeros(int(img_size[0]) * int(img_size[1]), dtype=np.uint8)]

        size = int(img_size[0]) * int(img_size[1])
        
        for channel in range (0, 3):
            pixel_count = 0
            for index in range(0, len(data[channel]), 2):
                pixel_length = pixel_count + int(data[channel][index + 1])
                # print(f"Adding pixel in ({pixel_count}, {pixel_length})")
                for i in range(pixel_count, pixel_length):
                    image_bgr[channel][i] = np.uint8(data[channel][index])
                pixel_count = pixel_length
            print("channel " + str(channel) + " decompress done.")



        # 先跑出一個一維 array
        # pixel_count = 0
        # for i in range(0, len(data) - 1, 2):
        #     image[pixel_count : pixel_count + int(data[i + 1])] = np.uint8(int(data[i]))
        #     pixel_count += int(data[i + 1])

        

        # 把圖片從 [B, B, G, G, R, R] 變成 [[B, G, R], [B, G, R]]
        # arr = [0, 0, 0]
        # img_bgr = []
        
        # print(len(image))
        # for i in range(0, size):
        #     img_bgr.append([image[i * 3 : i * 3 + 3]])

        # img_bgr = [image[0 : size], image[size : size * 2], image[size * 2 : size * 3]]

        # 將三個通道合併
        image_b = image_bgr[0].reshape(int(img_size[0]), int(img_size[1]))
        image_g = image_bgr[1].reshape(int(img_size[0]), int(img_size[1]))
        image_r = image_bgr[2].reshape(int(img_size[0]), int(img_size[1]))
        result_img = cv2.merge([image_b, image_g, image_r])

        # 將 BGR 改成 RGB
        # image = np.reshape(np.array(img_bgr), (int(img_size[0]), int(img_size[1]), 3))
        image = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # return 解壓縮完的圖片
        return image

def main():
    # 跑過三張圖片
    compress_ratio = [0, 0, 0]
    print("| Image Name | Original Size  | Compress Size  | Compression Ratio |")
    print("| ---------- | -------------- | -------------- | ----------------- |")
    for i in range(0, len(img_path)):
        original_img = img_path[i]
        compress_file = compress_path[i]
        
        # compress_ratio[i] =  compress(original_img, compress_file)

        decompress_img = decompress(compress_file)
        plt.subplot(2, 2, i + 1)
        plt.imshow(decompress_img)
        plt.title(img_path[i])
        plt.axis("off")

    average_ratio = sum(compress_ratio) / len(compress_ratio)
    print("Average compression ratio: " + str(average_ratio) + ".")
        
    plt.show()

if __name__ == "__main__":
    main()
