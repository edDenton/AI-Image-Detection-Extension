"""

@author: Edward Denton
"""
import random as rand
import pandas as pd
from PIL import Image
from pathlib import Path


# Process and separate ArtiFact
# Split is 70-15-15 (train-val-test) for all models being used in training
# CycleGAN, LaMa, StyleGAN2, Taming Transformer, VQ Diffusion saved purely for testing
def split_artifact(raw_path, processed_path, val_fraction=0.15, test_fraction=0.15, seed=42):
    rand.seed(seed)

    print("ArtiFact is being processed")
    held_out_models = ["cycle_gan", "lama", "stylegan2", "taming_transformer", "vq_diffusion"]
    raw_path = Path(raw_path)

    for model in raw_path.iterdir():
        if model.name == ".complete":
            continue

        elif model.is_dir() and model.name not in held_out_models:
            for file in model.iterdir():
                if file.is_file() and file.name == "metadata.csv":
                    df = pd.read_csv(file)

                    # target == 0 is real, any non-zero value is AI-generated
                    real_images = [Path(p) for p in df["image_path"][df["target"] == 0]]
                    print(f"{model.name} real images: {len(real_images)}")
                    if len(real_images) > 0:
                        rand.shuffle(real_images)
                        val_count = int(val_fraction * len(real_images))
                        test_count = int(test_fraction * len(real_images))

                        val_images = real_images[:val_count]
                        test_images = real_images[val_count:test_count + val_count]
                        train_images = real_images[test_count + val_count:]

                        for img in val_images:
                            resize_img(raw_path / model.name / img, processed_path / "val" / "real" / img.name)

                        for img in train_images:
                            resize_img(raw_path / model.name / img, processed_path / "train" / "real" / img.name)

                        for img in test_images:
                            resize_img(raw_path / model.name / img,
                                       processed_path / "test" / model.name / "real" / img.name)

                    # target == 0 is real, any non-zero value is AI-generated
                    fake_images = [Path(p) for p in df["image_path"][df["target"] != 0]]
                    print(f"{model.name} fake images: {len(fake_images)}")
                    if len(fake_images) > 0:
                        rand.shuffle(fake_images)
                        val_count = int(val_fraction * len(fake_images))
                        test_count = int(test_fraction * len(fake_images))

                        val_images = fake_images[:val_count]
                        test_images = fake_images[val_count:test_count + val_count]
                        train_images = fake_images[test_count + val_count:]

                        for img in val_images:
                            resize_img(raw_path / model.name / img, processed_path / "val" / "fake" / img.name)

                        for img in train_images:
                            resize_img(raw_path / model.name / img, processed_path / "train" / "fake" / img.name)

                        for img in test_images:
                            resize_img(raw_path / model.name / img,
                                       processed_path / "test" / model.name / "fake" / img.name)

        elif model.is_dir() and model.name in held_out_models:
            for file in model.iterdir():
                if file.is_file() and file.name == "metadata.csv":
                    df = pd.read_csv(file)

                    # target == 0 is real, any non-zero value is AI-generated
                    real_images = [Path(p) for p in df["image_path"][df["target"] == 0]]
                    print(f"{model.name} real images: {len(real_images)}")
                    if len(real_images) > 0:
                        rand.shuffle(real_images)

                        for img in real_images:
                            resize_img(raw_path / model.name / img,
                                       processed_path / "test" / model.name / "real" / img.name)

                    # target == 0 is real, any non-zero value is AI-generated
                    fake_images = [Path(p) for p in df["image_path"][df["target"] != 0]]
                    print(f"{model.name} fake images: {len(fake_images)}")
                    if len(fake_images) > 0:
                        rand.shuffle(fake_images)

                        for img in fake_images:
                            resize_img(raw_path / model.name / img,
                                       processed_path / "test" / model.name / "fake" / img.name)

        print(f"{model.name} is done")

    print("ArtiFact has been processed")
    return


# Process and separate CIFAKE
# Using the provided split, train will be divided into 80-20 train-val
# test will be left alone for purely testing
# def split_cifake(raw_path, processed_path, val_fraction=0.20, seed=42):
#     rand.seed(seed)
#
#     print("CIFAKE is being processed")
#     for label, folder in [("real", "REAL"), ("fake", "FAKE")]:
#         test_images = list((raw_path / "test" / folder).glob("*.jpg"))
#         rand.shuffle(test_images)
#
#         # Put CIFAKE images in their own folder, so we can keep track off how our model performs
#         # on specific generators
#         for img in test_images:
#             resize_img(img, processed_path / "test" / "cifake" / label / img.name)
#
#         train_images = list((raw_path / "train" / folder).glob("*.jpg"))
#         rand.shuffle(train_images)
#
#         val_count = int(val_fraction * len(train_images))
#         val_images = train_images[:val_count]
#         train_images = train_images[val_count:]
#
#         for img in train_images:
#             resize_img(img, processed_path / "train" / label / img.name)
#
#         for img in val_images:
#             resize_img(img, processed_path / "val" / label / img.name)
#
#     print("CIFAKE has been processed")
#     return


def resize_img(img_path, output_path, size=(224, 224)):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"{img_path} does not exist")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as img:
        img = img.convert('RGB')
        img = img.resize(size, Image.Resampling.BILINEAR)
        img.save(output_path)


def main():
    # base_dir makes sure we can run this file from anywhere, __file__ gives path to preprocess.py (this file)
    # and .parent will travel up the project root similar to cd ../../
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw"
    processed_path = base_dir / "data" / "processed"

    split_artifact(raw_path / "ArtiFact", processed_path)
    # split_cifake(raw_path / "CIFAKE", processed_path), no longer using CIFAKE currently because of the img size


if __name__ == '__main__':
    main()
