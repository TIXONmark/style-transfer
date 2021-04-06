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
python main.py --best_image_name=best.jpg \ # Best image saving name
               --best_image_saving_folder=results \ # Best image saving folder
               --beta_1=0.9 \ # Beta 1
               --content_layers=["block5_conv2"] \ # Content layers
               --content_paths=["images/content_image.jpg"] \ # Content images paths
               --content_weight=1e4 \ # Content weight
               --denoise_shape=2 \ # Denoise shape
               --denoise_weight=1e0 \ # Denoise weight
               --display_interval=10 \ # Display interval
               --epsilon=2e-2 \ # Epsilon
               --image_max_dim_size=1024 \ # Max dimension size for image (it is worth using a power of two)
               --learning_rate=1.1 \ # Learning rate
               --num_iterations=1000 \ # Num iterations
               --save_during_training=True \ # Save images during training
               --saving_folder=results \ # Folder for saving images
               --style_layers=["block1_conv1","block2_conv1","block3_conv1", \ 
                               "block4_conv1","block5_conv1"] \ # Style layers
               --style_paths=["images/style_image.jpg"] \ # Style images paths
               --style_weight=7e12 \ # Style weight
               --use_first_image=True \ # Use first image as init
               --use_style_norm=True \ # Should we divide style loss by 4.0 * (channels ** 2) * (width * height) ** 2
```

# How it works
Use VGG19 model and optimize content, style and denoise loss by changing source image.

# Features
- [x] Using multiple images;
- [x] Smoothing;
- [ ] Video processing.

# Requirements
```
tensorflow==2.4.1
numpy==1.19.5
Pillow==7.0.0
matplotlib==3.2.2
```

# Results
![](results/best.jpg)
