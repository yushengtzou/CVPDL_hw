# HW3 Report 

## Report 
### 1.Image Captioning
#### a.Compare the performance of different pre-trained models in generating captions

* I selected 2 pre-trained models in generating captions
  1. [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
  4. [Salesforce/blip-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
* CLIP Model ```ViT-B/32``` was used for performance evaluation. The process involved preprocessing each image and its corresponding text, encoding them to obtain features, and computing the cosine similarity. The average cosine similarity was used as the final score.
* After conducting ```CUDA_VISIBLE_DEVICES=0 python ImgCaption.py```, the final scores are shown below(Could also check ```./pretrained_compare_results.txt```):
  1. Salesforce/blip2-opt-2.7b:0.77609
  2. Salesforce/blip-flan-t5-xl:0.7757
* The ```Salesforce/blip2-opt-2.7b``` model demonstrated superior performance and was chosen for caption generation.

#### b.Design 2 templates of prompts for later generating comparison

* I designed three templates of prompts for later generating comparisons. All prompts have two parts: **predix** and **suffix**. The prefix among all prompts is the same: **a real photo of "Generated_Text", "label", width: 'width', height: 'height',  in the aquarium, undersea background, non-grayscale**. The difference lies in the suffix:
  1. First: **camera shake, standard definition**
  2. Second: **high definition, limited color palette**
  3. Third: **camera shake, standard definition, splashes, restricted color**

* These prompts will be utilized for two primary purposes: FID Score calculation and data augmentation. The first will use all training dataset images, while the second will select images based on specific criteria (no more than 6 bounding boxes and a single category)


### 2. Text-to-Image Generation
#### a. Generate Images from Text Grounding

* Images were generated using all three prompts, with results saved in ```2_prompts2GenerateImage```

### b. Generate Images from Image Grounding
* Following the same approach as in section 2a, results are stored in ```2_prompts2GenerateImage```

### 3. Compare FID

* Text Grounding

  * I would compare the performance of all three prompts created earliar.The FID scores are stored in ```2_prompts2GenerateImage``` and shown here:

    |Prompts|FID Score|
    |-------|---------|
    | First | 144.7596|
    | Second| 144.7691|
    | Third | 141.7163|

 * FID scores for all three prompts indicated minimal differences, suggesting the significance of species and background descriptions over other details in image generation with GLIGEN


* Image Grounding

  * Since the **third** prompt has the lowest FID result. We used the **third** prompt to conduct image generating by image grounding.

  * The **FID result** is 144.8596. 
    |Prompts|FID Score|
    |-------|---------|
    | Third | 144.8596|
  
 * The third prompt achieved the lowest FID score, surprisingly not surpassing the text grounding results. This suggests the choice of images for generation plays a critical role in outcome quality.


### 4. Improve Model

* Brief Description

  1. Data Augmentation Strategy: Given the prevalence of fish images in the dataset, I endeavored to balance the dataset by generating additional images for other species. The goal was to equalize the representation across all species. The final dataset comprised 1474 images, including 448 original and 1026 newly generated images.


  2. Approach to Bounding Boxes in Generated Images: The recommended practice for image generation is to use images with no more than six bounding boxes, each depicting a single category. This led to a pivotal question: Should the generated images maintain the same bounding box configurations as their original counterparts, or should we opt for a randomized selection within the advised parameters? To explore this, I experimented with four different datasets. In two of these datasets, the generated images mirrored the bounding box layouts of the original images. For the remaining two datasets, the bounding boxes for the generated images were randomly determined, adhering to the recommended guidelines. This approach allowed for a comprehensive assessment of the impact of bounding box configurations on model training and performance.


* The mAP Results are shown:
  |Data Augment|Bouding Boxes|Grounding|MAP|
  |-|-------------|---------|---|
  |X|Original|X|0.4695|
  |V|Original|Text Groudning|0.4549|
  |V|Original|Image Groudning|0.4544|
  |V|Random|Text Groudning|0.4655|
  |V|Random|Image Groudning|0.4729|

### 5. Visualization
* fish

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/fish/Combine.png'></div>

* jellyfish

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/jellyfish/Combine.png'></div>

* penguin

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/penguin/Combine.png'></div>

* puffin

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/puffin/Combine.png'></div>

* shark

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/shark/Combine.png'></div>

* starfish

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/starfish/Combine.png'></div>

* stingray

  <div align=center><img src='https://github.com/lycge20923/CVPDL_HW3/blob/main/result_5/stingray/Combine.png'></div>

## Excution

### Preliminary Action
#### 1. Download HW1 Dataset
1. Go to *NTU COOL*, download the ```hw1_dataset.zip``` and put it in the directory.
2. unzip the ```hw1_dataset.zip```: ``` unzip hw1_dataset.zip```

#### 2.Create a Virtual Environment

```
conda create --name your_name python=3.11 -y
source activate your_name
```

#### 3.Download the related packages

```
pip install numpy==1.23.5
pip install cython
pip install -r requirements.txt
pip uninstall -y torchtext
```

#### 4.Download ```GLIGEN``` and its relative pre-trained weights

* Download ```GLIGEN```
  ```
  git clone https://github.com/gligen/GLIGEN.git
  ```
* Download pre-trained weights
  ```
  wget https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin?download=true -O checkpoint_generation_text.pth
  wget https://huggingface.co/gligen/gligen-generation-text-image-box/resolve/main/diffusion_pytorch_model.bin?download=true -O checkpoint_generation_text_image.pth
  ```

### 5. Download ```DAB-DETR``` and its relative pre-trained weights

* Download ```DAB-DETR`` and setup.

  ```
  git clone https://github.com/IDEA-Research/DAB-DETR.git
  cd DAB-DETR/models/dab_deformable_detr/ops
  python setup.py build install
  # unit test (should see all checking is True)
  python test.py
  cd ../../../..
  ```

* Download the pre-trained weight: 
  1. Go to the website: https://drive.google.com/drive/folders/1vVxRu8wmNA7sxdEB46vcoYlXvJIy2-Ep
  2. Download the ```checkpoint.pth``` 
  3. Move the ```checkpoint.pth``` to the ```CVPDL_HW3``` directory.

### Run 
#### 1. Generate Image Caption

* Must add ```CUDA_VISIBLE_DEVICES=0``` in excution code, and change the **Device Number** to the GPU you could run.
* After running the following code, the results would all store in ```./result_1/```

  ```
  CUDA_VISIBLE_DEVICES=0 1_processJson.py 
  ```

### 2. Text-to-Image Generation & 3. Calculate FID

* Copy the file ```2_text2Image.py``` to the ```GLIGEN``` directory, move to it and execute.

  ```
  cp -r 2_text2Image.py GLIGEN/2_text2Image.py
  cd GLIGEN
  python 2_text2Image.py --batch_size 1
  cd ..
  ```

### 4. Improve Model
* To generate the needed images, running 
  ```
  cp -r 3_textGrounding.py GLIGEN/3_textGrounding.py
  cd GLIGEN
  python 3_textGrounding.py --batch_size 1
  cd ..
  ```

* After generating the images, we then prepared for the training datasets and annotations.

  1. Original Boxes(with Text Grounding)
    ```
    cp -r hw1_dataset result_4/originalboxes/hw1_dataset_TG
    mv result_4/originalboxes/hw1_dataset_TG/annotations/val.json result_4/originalboxes/hw1_dataset_TG/annotations/instances_val2017.json
    mv 'result_4/train_Text_Grounding(original).json' result_4/originalboxes/hw1_dataset_TG/annotations/instances_train2017.json
    mv result_4/originalboxes/hw1_dataset_TG/train result_4/originalboxes/hw1_dataset_TG/train2017
    mv result_4/originalboxes/hw1_dataset_TG/valid result_4/originalboxes/hw1_dataset_TG/val2017
    mv result_4/originalboxes/generation_box_text/* result_4/originalboxes/hw1_dataset_TG/train2017
    rm -r result_4/originalboxes/generation_box_text
    ```
  
  2. Original Boxes(with Image Grounding)
    ```
    cp -r hw1_dataset result_4/originalboxes/hw1_dataset_IG
    mv result_4/originalboxes/hw1_dataset_IG/annotations/val.json result_4/originalboxes/hw1_dataset_IG/annotations/instances_val2017.json
    mv 'result_4/train_Image_Grounding(original).json' result_4/originalboxes/hw1_dataset_IG/annotations/instances_train2017.json
    mv result_4/originalboxes/hw1_dataset_IG/train result_4/originalboxes/hw1_dataset_IG/train2017
    mv result_4/originalboxes/hw1_dataset_IG/valid result_4/originalboxes/hw1_dataset_IG/val2017
    mv result_4/originalboxes/generation_box_image/* result_4/originalboxes/hw1_dataset_IG/train2017
    rm -r result_4/originalboxes/generation_box_image
    ```  
  
  3. Random Boxes(with Text Grounding)
    ```
    cp -r hw1_dataset result_4/randomboxes/hw1_dataset_TG
    mv result_4/randomboxes/hw1_dataset_TG/annotations/val.json result_4/randomboxes/hw1_dataset_TG/annotations/instances_val2017.json
    mv 'result_4/train_Text_Grounding(random).json' result_4/randomboxes/hw1_dataset_TG/annotations/instances_train2017.json
    mv result_4/randomboxes/hw1_dataset_TG/train result_4/randomboxes/hw1_dataset_TG/train2017
    mv result_4/randomboxes/hw1_dataset_TG/valid result_4/randomboxes/hw1_dataset_TG/val2017
    mv result_4/randomboxes/generation_box_text/* result_4/randomboxes/hw1_dataset_TG/train2017
    rm -r result_4/randomboxes/generation_box_text
    ```
  
  4. Random Boxes(with Image Grounding)
    ```
    cp -r hw1_dataset result_4/randomboxes/hw1_dataset_IG
    mv result_4/randomboxes/hw1_dataset_IG/annotations/val.json result_4/randomboxes/hw1_dataset_IG/annotations/instances_val2017.json
    mv 'result_4/train_Image_Grounding(random).json' result_4/randomboxes/hw1_dataset_IG/annotations/instances_train2017.json
    mv result_4/randomboxes/hw1_dataset_IG/train result_4/randomboxes/hw1_dataset_IG/train2017
    mv result_4/randomboxes/hw1_dataset_IG/valid result_4/randomboxes/hw1_dataset_IG/val2017
    mv result_4/randomboxes/generation_box_image/* result_4/randomboxes/hw1_dataset_IG/train2017
    rm -r result_4/randomboxes/generation_box_image
    ```

* Train the models
  * First, change to the ```DAB-DETR``` directory
    ```cd DAB-DETR```
  
  * Then, train the models and put the result in the ```../result_4```
    1. Original Boxes(with Text Grounding)

      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/originalboxes/results_TG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/originalboxes/hw1_dataset_TG \
      --resume ../checkpoint.pth 
      ```
    
    2. Original Boxes(with Image Grounding)
    
      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/originalboxes/results_IG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/originalboxes/hw1_dataset_IG \
      --resume ../checkpoint.pth 
      ```
    3. Random Boxes(with Text Grounding)

      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/randomboxes/results_TG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/randomboxes/hw1_dataset_TG \
      --resume ../checkpoint.pth 
      ```
    4. Random Boxes(with Image Grounding)

      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/randomboxes/results_IG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/randomboxes/hw1_dataset_IG \
      --resume ../checkpoint.pth 
      ```
* Evaluation
  * First, output the prediction of validation dataset. We first copy ```4-2.py``` to ```DAB-DETR``` directory, then change the directory to ```DAB-DETR```
    ```
    cp 4-2.py DAB-DETR/
    cd DAB-DETR
    ```
  * Then, output the prediction json file

    1. Original Boxes(with Text Grounding)

      ```
      python 4_predict.py \
      ../result_4/originalboxes/results_TG/config.json \
      ../result_4/originalboxes/results_TG/checkpoint.pth \
      ../result_4/originalboxes/hw1_dataset_TG/val2017 \
      ../result_4/originalboxes/hw1_dataset_TG/annotations/instances_val2017.json \
      ../result_4/originalboxes/pred_TG.json
      ```

    2. Original Boxes(with Image Grounding)
      
      ```
      python 4_predict.py \
      ../result_4/originalboxes/results_IG/config.json \
      ../result_4/originalboxes/results_IG/checkpoint.pth \
      ../result_4/originalboxes/hw1_dataset_IG/val2017 \
      ../result_4/originalboxes/hw1_dataset_IG/annotations/instances_val2017.json \
      ../result_4/originalboxes/pred_IG.json
      ```

    3. Random Boxes(with Text Grounding)
      
      ```
      python 4_predict.py \
      ../result_4/randomboxes/results_TG/config.json \
      ../result_4/randomboxes/results_TG/checkpoint.pth \
      ../result_4/randomboxes/hw1_dataset_TG/val2017 \
      ../result_4/randomboxes/hw1_dataset_TG/annotations/instances_val2017.json \
      ../result_4/randomboxes/pred_TG.json
      ```

    4. Random Boxes(with Image Grounding)
      ```
      python 4_predict.py \
      ../result_4/randomboxes/results_IG/config.json \
      ../result_4/randomboxes/results_IG/checkpoint.pth \
      ../result_4/randomboxes/hw1_dataset_IG/val2017 \
      ../result_4/randomboxes/hw1_dataset_IG/annotations/instances_val2017.json \
      ../result_4/randomboxes/pred_IG.json
      ```
  
  * Evaluate the prediction: initially, we have to go back to ```CVPDL_HW3``` directory, and then conduct the evaluation
    ```
    cd ..
    ```

    1. Original Boxes(with Text Grounding)
      
      ```
      python 5_evaulate.py result_4/originalboxes/pred_TG.json \
      result_4/originalboxes/hw1_dataset_TG/annotations/instances_val2017.json
      ```

    2. Original Boxes(with Image Grounding)
      
      ```
      python 5_evaulate.py result_4/originalboxes/pred_IG.json \
      result_4/originalboxes/hw1_dataset_IG/annotations/instances_val2017.json
      ```

    3. Random Boxes(with Text Grounding)
      
      ```
      python 5_evaulate.py result_4/randomboxes/pred_TG.json \
      result_4/randomboxes/hw1_dataset_TG/annotations/instances_val2017.json
      ```

    4. Random Boxes(with Image Grounding)
      
      ```
      python 5_evaulate.py result_4/randomboxes/pred_IG.json \
      result_4/randomboxes/hw1_dataset_IG/annotations/instances_val2017.json
      ```
