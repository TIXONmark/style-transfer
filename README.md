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
"--display_interval", type=int, default=10, help="Display interval"

"--image_max_dim_size",
        type=int,
        default=512,
        help="Max dimension size for image"
        
"--num_iterations", type=int, default=1000, help="Num iterations"

"--style_weight", type=float, default=7e12, help="Style weight"

--content_weight", type=float, default=1e4, help="Content weight"

"--denoise_weight", type=float, default=1e0, help="Denoise weight"

"--learning_rate", type=float, default=1.1, help="Learning rate"

"--beta_1", type=float, default=0.9, help="Beta 1"

"--epsilon", type=float, default=2e-2, help="Epsilon"

"--use_style_norm",
        type=bool,
        default=True,
        help="Should we divide style loss by 4.0 * (channels ** 2) * (width * height) ** 2"

"--save_during_training",
        type=bool,
        default=True,
        help="Save images during training"

"--saving_folder", type=str, default="results", help="Folder for saving images"

"--best_image_name", type=str, default="best.jpg", help="Best image saving name"

"--best_image_saving_folder", type=str, default="results", help="Best image saving name"

"--denoise_shape", type=int, default=2, help="Denoise shape"

"--use_first_image", type=bool, default=True, help="Use first image as init"

"--content_paths",
        type=str,
        nargs="+",
        default=["images/content_image.jpg"],
        help="Content images paths"

"--style_paths",
        type=str,
        nargs="+",
        default=["images/style_image.jpg"],
        help="Style images paths"

"--content_layers",
        type=str,
        nargs="+",
        default=["block5_conv2"],
        help="Content layers"

"--style_layers",
        type=str,
        nargs="+",
        default=[
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        help="Style layers"
```

# Results
![](results/best.jpg)