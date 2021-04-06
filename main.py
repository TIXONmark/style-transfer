from PIL import Image
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.preprocessing import image as keras_image

class ImagesProcessing:
    """Functions for images processing"""
    @staticmethod
    def get_image(image_path: str, args: dict) -> np.ndarray:
        """Get image from file"""
        # Read image
        img = Image.open(image_path)

        # Scale image
        scale = args["image_max_dim_size"] / max(img.size)
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)))

        # Transform to array
        img = keras_image.img_to_array(img)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def load_and_process_images_for_model(image_paths, args: dict) -> list:
        """Load image and process it for VGG19 model"""
        imgs = []
        for image_path in image_paths:
            # Get image
            img = ImagesProcessing.get_image(image_path, args)
            # Preprocess image for VGG19
            img = tf.keras.applications.vgg19.preprocess_input(img)
            # Add to images
            imgs.append(img)
        return imgs

    @staticmethod
    def process_image_for_saving(image: np.ndarray) -> np.ndarray:
        """Process image for saving to"""
        # Copy image
        img = image.copy()

        # Remove batch dimension
        img = np.squeeze(img, 0)

        # Add mean (from VGG19 model)
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68

        # BGR to RGB
        img = img[:, :, ::-1]

        # Clip values
        img = np.clip(img, 0, 255).astype("uint8")
        return img

class Losses:
    """Functions for computing losses"""
    @staticmethod
    def get_content_loss(base_content, target):
        """Content loss"""
        return tf.reduce_mean(tf.square(base_content - target))

    @staticmethod
    def gram_matrix(input_tensor):
        """Get Gram's matrix"""
        # Make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]

        # Gram matrix
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)
        
    @staticmethod
    def get_style_loss(base_style, gram_target, args: dict):
        """Style loss"""
        # Get Gram's matrix
        height, width, channels = base_style.get_shape()
        gram_style = Losses.gram_matrix(base_style)

        if args["use_style_norm"]:
            return tf.reduce_mean(tf.square(gram_style - gram_target)) / (
                4.0 * (channels ** 2) * (width * height) ** 2
            )
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    @staticmethod
    def get_denoise_loss(image: np.ndarray, args: dict):
        """Denoise loss"""
        # Get difference sum for all neighbours
        sum_of_diff = tf.reduce_sum(tf.abs(image[:, :, :, :] - image[:, :, :, :]))
        for i in range(0, args["denoise_shape"] + 1):
            for j in range(0, args["denoise_shape"] + 1):
                sum_of_diff += tf.reduce_sum(
                    tf.abs(
                        image[:, i:, j:, :]
                        - image[:, : image.shape[1] - i, : image.shape[2] - j, :]
                    )
                )
        return sum_of_diff / float((args["denoise_shape"] + 1) ** 2 - 1)


class Models:
    """Functions for getting models"""
    @staticmethod
    def get_model(args: dict):
        """Get VGG19 model"""
        # Load VGG19 model
        vgg_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
        vgg_model.trainable = False

        # Get outputs of model
        style_outputs = [vgg_model.get_layer(name).output for name in args["style_layers"]]
        content_outputs = [
            vgg_model.get_layer(name).output for name in args["content_layers"]
        ]
        model_outputs = style_outputs + content_outputs

        # Get model
        model = models.Model(vgg_model.input, model_outputs)
        return model


class Features:
    """Functions for getting features"""
    @staticmethod
    def get_feature_representations(model, args: dict) -> (list, list):
        """Get feature representations for all images"""
        # Load content and style images
        content_images = ImagesProcessing.load_and_process_images_for_model(args["content_paths"], args)
        style_images = ImagesProcessing.load_and_process_images_for_model(args["style_paths"], args)

        # Compute content and style outputs
        styles_outputs = []
        for style_image in style_images:
            styles_outputs.append(model(style_image))
        contents_outputs = []
        for content_image in content_images:
            contents_outputs.append(model(content_image))

        # Get 0 because of 1 image in batch (1 output)
        styles_features = [
            [style_layer[0] for style_layer in style_outputs[: len(args["style_layers"])]]
            for style_outputs in styles_outputs
        ]
        contents_features = [
            [
                content_layer[0]
                for content_layer in content_outputs[len(args["style_layers"]) :]
            ]
            for content_outputs in contents_outputs
        ]
        return styles_features, contents_features


class Computing:
    """Functions for computing"""

    @staticmethod
    def compute_loss(
        args: dict,
        model,
        loss_weights: tuple,
        init_image: np.ndarray,
        gram_styles_features: list,
        contents_features: list,
    ) -> float:
        """Get loss for image"""
        # Get weights
        style_weight, content_weight, denoise_weight = loss_weights

        # Get model outputs
        model_outputs = model(init_image)

        # Get style and content outputs
        style_output_features = model_outputs[: len(args["style_layers"])]
        content_output_features = model_outputs[len(args["style_layers"]) :]

        # Init scores
        style_score = 0
        content_score = 0
        denoise_score = 0

        # Get scores for style features
        for gram_style_features in gram_styles_features:
            weight_per_style_layer = 1.0 / float(
                len(args["style_layers"]) * len(args["content_paths"])
            )
            for target_style, comb_style in zip(gram_style_features, style_output_features):
                style_score += weight_per_style_layer * Losses.get_style_loss(
                    comb_style[0], target_style, args
                )

        # Get scores for content features
        for content_features in contents_features:
            weight_per_content_layer = 1.0 / float(
                len(args["content_layers"]) * len(args["style_paths"])
            )
            for target_content, comb_content in zip(
                content_features, content_output_features
            ):
                content_score += weight_per_content_layer * Losses.get_content_loss(
                    comb_content[0], target_content
                )

        # Get score for denoise feature
        denoise_score += Losses.get_denoise_loss(init_image, args)

        # Multiply by weight
        style_score *= style_weight
        content_score *= content_weight
        denoise_score *= denoise_weight

        # Get total loss
        loss = style_score + content_score + denoise_score
        return loss, style_score, content_score, denoise_score

    @staticmethod
    def compute_grads(cfg: dict, args: dict):
        """Compute gradients"""
        # Compute loss using tensorflow
        with tf.GradientTape() as tape:
            all_loss = Computing.compute_loss(args, **cfg)

        # Get total loss
        total_loss = all_loss[0]

        # Return gradient and all losses (with total)
        return tape.gradient(total_loss, cfg["init_image"]), all_loss


def run_style_transfer(args: dict) -> (np.ndarray, float):
    """Main function. Style transfer"""
    # Get model
    model = Models.get_model(args)

    # Disable trainable
    for layer in model.layers:
        layer.trainable = False

    # Get features representations for styles and contents images
    styles_features, contents_features = Features.get_feature_representations(model, args)

    # Get Gram's matrix of features
    gram_styles_features = [
        [Losses.gram_matrix(style_feature) for style_feature in style_features]
        for style_features in styles_features
    ]

    # Init image is first content image or noise
    if args["use_first_image"]:
        init_image = ImagesProcessing.load_and_process_images_for_model(
            [args["content_paths"][0]], args
        )[0]
    else:
        init_image = ImagesProcessing.load_and_process_images_for_model(
            [args["content_paths"][0]], args
        )[0]
        init_image = np.round(np.random.rand(*init_image.shape) * 255)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Use Adam optimizer
    opt = tf.optimizers.Adam(
        learning_rate=args["learning_rate"],
        beta_1=args["beta_1"],
        epsilon=args["epsilon"],
    )

    # Variables for getting best image
    best_loss, best_img = float("inf"), None

    # Make config for training
    loss_weights = (
        args["style_weight"],
        args["content_weight"],
        args["denoise_weight"],
    )
    cfg = {
        "model": model,
        "loss_weights": loss_weights,
        "init_image": init_image,
        "gram_styles_features": gram_styles_features,
        "contents_features": contents_features,
    }

    # Depends on VGG19 model
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    # Start time
    start_time = time.time()

    # Iterations loop
    for i in range(args["num_iterations"]):
        # Compute gradients
        grads, all_loss = Computing.compute_grads(cfg, args)

        # Get losses
        loss, style_score, content_score, denoise_score = all_loss

        # Apply gradients
        opt.apply_gradients([(grads, init_image)])

        # Clip
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        # Save best image
        if loss < best_loss:
            best_loss = loss
            best_img = ImagesProcessing.process_image_for_saving(init_image.numpy())

        # Display interval and save images
        if i % args["display_interval"] == 0:
            if args["save_during_training"]:
                img_checkpoint = init_image.numpy()
                img_checkpoint = ImagesProcessing.process_image_for_saving(img_checkpoint)
                Image.fromarray(img_checkpoint).save(
                    os.path.join(
                        args["saving_folder"],
                        "result_" + str(i) + "_" + str(time.time()) + ".jpg",
                    )
                )
            print("Iteration: {}".format(i))
            print(
                "Total loss: {:.4e}, "
                "style loss: {:.4e}, "
                "content loss: {:.4e}, "
                "denoise loss: {:.4e}, "
                "time: {:.4f}s".format(
                    loss,
                    style_score,
                    content_score,
                    denoise_score,
                    time.time() - start_time,
                )
            )
    return best_img, best_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Style transfer")
    parser.add_argument(
        "--display_interval", type=int, default=10, help="Display interval"
    )
    parser.add_argument(
        "--image_max_dim_size",
        type=int,
        default=512,
        help="Max dimension size for image",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000, help="Num iterations"
    )
    parser.add_argument("--style_weight", type=float, default=7e12, help="Style weight")
    parser.add_argument(
        "--content_weight", type=float, default=1e4, help="Content weight"
    )
    parser.add_argument(
        "--denoise_weight", type=float, default=1e0, help="Denoise weight"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.1, help="Learning rate"
    )
    parser.add_argument("--beta_1", type=float, default=0.9, help="Beta 1")
    parser.add_argument("--epsilon", type=float, default=2e-2, help="Epsilon")
    parser.add_argument(
        "--use_style_norm",
        type=bool,
        default=True,
        help="Should we divide style loss by 4.0 * (channels ** 2) * (width * height) ** 2",
    )
    parser.add_argument(
        "--save_during_training",
        type=bool,
        default=True,
        help="Save images during training",
    )
    parser.add_argument(
        "--saving_folder", type=str, default="results", help="Folder for saving images"
    )
    parser.add_argument(
        "--best_image_name", type=str, default="best.jpg", help="Best image saving name"
    )
    parser.add_argument(
        "--best_image_saving_folder", type=str, default="results", help="Best image saving name"
    )
    parser.add_argument("--denoise_shape", type=int, default=2, help="Denoise shape")
    parser.add_argument(
        "--use_first_image", type=bool, default=True, help="Use first image as init"
    )
    parser.add_argument(
        "--content_paths",
        type=str,
        nargs="+",
        default=["images/content_image.jpg"],
        help="Content images paths",
    )
    parser.add_argument(
        "--style_paths",
        type=str,
        nargs="+",
        default=["images/style_image.jpg"],
        help="Style images paths",
    )
    parser.add_argument(
        "--content_layers",
        type=str,
        nargs="+",
        default=["block5_conv2"],
        help="Content layers",
    )
    parser.add_argument(
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
        help="Style layers",
    )
    args = parser.parse_args().__dict__
    best_img, best_loss = run_style_transfer(args)
    best_img = Image.fromarray(best_img)
    best_img.save(os.path.join(args["best_image_saving_folder"], args["best_image_name"]))
