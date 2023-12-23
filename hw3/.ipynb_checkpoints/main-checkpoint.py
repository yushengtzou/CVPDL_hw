# CVPDL_hw3 R12945072

import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import os
os.environ["TRANSFORMERS_CACHE"] = "/mnt/lab/2.course/112-1/CVPDL_hw/hw3/cache/hub"


# Define the cache directory
cache_dir = "/mnt/lab/2.course/112-1/CVPDL_hw/cache/hub"

# Load image
url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
#display(image.resize((596, 437)))

plt.imshow(image.resize((596, 437)))
plt.axis('off')  # Hide the axis
plt.show()


# Load model and processor
# Specify the cache directory in the from_pretrained method
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir, torch_dtype=torch.float16)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Image Captioning
inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
