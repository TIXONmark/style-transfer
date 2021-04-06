# Style transfer
Transferring styles from several images to others.

# Get started
```
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
deactivate
rm -rf venv
```

# Options
```
python main.py --best_image_name best.jpg \
               --best_image_saving_folder results \
               --beta_1 0.9 \
               --content_layers block5_conv2 \
               --content_paths images/content_image.jpg \
               --content_weight 1e4 \
               --denoise_shape 2 \
               --denoise_weight 1e0 \
               --display_interval 10 \
               --epsilon 2e-2 \
               --image_max_dim_size 1024 \
               --learning_rate 1.1 \
               --num_iterations 1000 \
               --save_during_training True \
               --saving_folder results \
               --style_layers block1_conv1 block2_conv1 block3_conv1 block4_conv1 block5_conv1 \
               --style_paths images/style_image.jpg \
               --style_weight 7e12 \
               --use_first_image True \
               --use_style_norm True
```
- best_image_name: best image saving name
- best_image_saving_folder: best image saving folder
- content_layers: list of content layers
- content_paths: list of content images paths
- content_weight: content weight
- denoise_shape: denoise shape
- denoise_weight: denoise weight
- display_interval: display interval
- epsilon: tensorflow epsilon param
- image_max_dim_size: max dimension size for image (it is worth using a power of two, less - faster)
- learning_rate: tensorflow learning_rate param
- num_iterations: num iterations
- save_during_training: save images during training
- saving_folder: folder for saving images
- style_layers: list of style layers
- style_paths: list of style images paths
- style_weight: style weight
- use_first_image: use first image as init
- use_style_norm: should we divide style loss by 4.0 * (channels ** 2) * (width * height) ** 2

# How it works
Use VGG19 model and optimize content, style and denoise loss by changing source image.

# Features
- [x] Using multiple images
- [x] Smoothing
- [ ] Make image from noise
- [ ] Video processing

# Requirements
```
tensorflow==2.4.1
numpy==1.19.5
Pillow==7.0.0
matplotlib==3.2.2
```

# Results
![](results/best.jpg)
