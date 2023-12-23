# CVPDL_hw3 R12945072 鄒雨笙
# 副程式


# 以下引入是為了 image2Text() 函式的使用
import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import os
# 以下引入是為了 text2Image() 函式的使用
import logging
from transformers import logging as hf_logging
from diffusers import StableDiffusionPipeline


os.environ["TRANSFORMERS_CACHE"] = "/mnt/lab/2.course/112-1/CVPDL_hw/hw3/cache/hub"
# 設定快取資料夾
cache_dir = "/mnt/lab/2.course/112-1/CVPDL_hw/cache/hub"


# 以下函式功能是使用 BLIP2 模型 用圖片產生提詞
def image2Text27b(image_path, cache_dir):
    # 讀取圖片 
    image = Image.open(image_path).convert('RGB')
    # 展示圖片 
    # plt.imshow(image)
    # plt.axis('off')  # Hide the axis
    # plt.show()

    # 載入 model "Salesforce/blip2-opt-2.7b" and processor
    # Specify the cache directory in the from_pretrained method
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir, torch_dtype=torch.float16)

    # 使用 GPU
    if torch.cuda.is_available(): 
        device = "cuda" 
        print("GPU is being used")
    else: 
        device = "cpu" 
        print("CPU is being used")

    model.to(device)

    # 用圖片產生提詞
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    return generated_text


# 以下函式功能是使用 BLIP2 模型 用圖片產生提詞
def image2Text67b(image_path, cache_dir):
    # 讀取圖片 
    image = Image.open(image_path).convert('RGB')
    # 展示圖片 
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()

    # 載入 model "Salesforce/blip2-opt-6.7b-coco" and processor
    # Specify the cache directory in the from_pretrained method
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco", cache_dir=cache_dir)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco", cache_dir=cache_dir, torch_dtype=torch.float16)

    # 使用 GPU
    if torch.cuda.is_available(): 
        device = "cuda" 
        print("GPU is being used")
    else: 
        device = "cpu" 
        print("CPU is being used")

    model.to(device)

    # 用圖片產生提詞
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    return generated_text


# 以下函式功能是使用 GLIGEN 模型 將文字產生圖片
def text2Image(generated_text):
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    hf_logging.set_verbosity_info()

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    cache_dir = "/mnt/lab/2.course/112-1/CVPDL_hw/hw3/cache/hub" 

    # 載入 pipeline from the specified cache directory
    # If the model isn't in the cache directory, it will be downloaded and saved there
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = generated_text
    # 生成圖片
    image = pipe(prompt).images[0]

    # 儲存生成出來的圖片 
    # image.save("astronaut_rides_horse.png")

    # 展示生成出來的圖片 
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.show()




