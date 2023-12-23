# CVPDL_hw3 R12945072 鄒雨笙
# 主程式


import os
from util import processJson, processImageWith123Bbox
# 引入 image2Text() 和 text2Image() 函式
from model import image2Text27b, image2Text67b, text2Image


def main():

    # ------------------- 相關路徑設定 ------------------- 

    # 設定 json 路徑
    file_path = './hw1_dataset/annotations/train.json'

    # ------------------- 呼叫函式 ------------------- 

    # 讀取 json 檔，並篩選出只有一個 category，每個 category
    # 有多個 bbox 的 image 和其記錄，再依照 category
    # 各存成一個 json 檔，將這些 json 檔存入新建的 ./category 資料夾
    processJson(file_path)


    # 呼叫 image2Text27b() 函式
    # generated_text1 = image2Text27b(image_path, cache_dir)
    # 呼叫 image2Text67b() 函式
    # generated_text2 = image2Text67b(image_path, cache_dir)
    # 呼叫 text2Image() 函式
    # text2Image(generated_text)


if __name__ == '__main__':
    main()

