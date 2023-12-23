import logging
from transformers import logging as hf_logging

# Set logging level
logging.basicConfig(level=logging.INFO)
hf_logging.set_verbosity_info()

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
cache_dir = "/mnt/lab/2.course/112-1/CVPDL_hw/hw3/cache/hub"  # Specify your custom cache directory here

# Load the pipeline from the specified cache directory
# If the model isn't in the cache directory, it will be downloaded and saved there
pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
# Generate the image
image = pipe(prompt).images[0]

# Save the generated image
# image.save("astronaut_rides_horse.png")

# Display the image if you want to see it in the notebook
plt.imshow(image)
plt.axis('off')  # Hide the axis
plt.show()
