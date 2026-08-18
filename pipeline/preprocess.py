"""

@author: Edward Denton
"""
import random
import pandas as pd
from PIL import Image
from pathlib import Path
from train import load_params
import shutil


def allocate(gen_counts, total):
    allocated_counts = {}

    gens = list(gen_counts.keys())
    share = total // len(gens)

    for gen in gens:
        allocated_counts[gen] = min(gen_counts[gen], share)

    difference = total - sum(allocated_counts.values())

    for gen in gens:
        if difference <= 0:
            break

        spare = gen_counts[gen] - allocated_counts[gen]
        extra = min(spare, difference)
        allocated_counts[gen] += extra
        difference -= extra

    return allocated_counts


def get_images(label_df, total):
    gen_dict = {}

    # Get the number of imgs for each generator
    for gen in label_df["generator"].unique():
        paths = list(label_df.loc[label_df["generator"] == gen, "image_path"])
        gen_dict[gen] = paths

    # generator: count of imgs it has in its dataset
    counts = {g: len(paths) for g, paths in gen_dict.items()}

    # Get an even amount of each generator unless one can't match the even amount
    allocated_counts = allocate(counts, total)

    selected = []
    for gen, take_n in allocated_counts.items():
        paths = gen_dict[gen]
        random.shuffle(paths)
        for path in paths[:take_n]:
            selected.append((gen, path))

    return selected


def split_data(images):
    gen_dict = {}
    TRAIN_FRACTION = 0.7
    VAL_FRACTION = 0.15

    # We are splitting by generator within the data to make sure we get an even amount of each generator in training
    for generator, path in images:
        gen_dict.setdefault(generator, []).append(path)

    train, val, test = [], [], []

    for gen, paths in gen_dict.items():
        random.shuffle(paths)

        val_count = int(len(paths) * VAL_FRACTION)
        train_count = int(len(paths) * TRAIN_FRACTION)

        train_paths = paths[:train_count]
        val_paths = paths[train_count:val_count + train_count]
        test_paths = paths[val_count + train_count:]

        train.extend((gen, path) for path in train_paths)
        val.extend((gen, path) for path in val_paths)
        test.extend((gen, path) for path in test_paths)

    return train, val, test


def create_processed(raw_path, metadata_path, processed_path):
    # config = load_params()
    # train_config = config["train"]
    # random.seed(train_config["seed"])
    #
    metadata = pd.read_csv(metadata_path)
    #
    # # Loop through each category, make sure there's an even amount of real and fake images for the category
    # for category in metadata["category"].unique():
    #     curr_category_df = metadata[metadata["category"] == category]
    #
    #     real = curr_category_df[curr_category_df["target"] == 0]
    #     fake = curr_category_df[curr_category_df["target"] != 0]
    #
    #     # Take the min to make sure one type of image doesn't outweigh the other
    #     total = min(len(real), len(fake))
    #
    #     real_imgs = get_images(real, total)
    #     fake_imgs = get_images(fake, total)
    #
    #     real_train, real_val, real_test = split_data(real_imgs)
    #     fake_train, fake_val, fake_test = split_data(fake_imgs)
    #
    #     # Add all imgs to the processed directory grouped by train, val, and test by category
    #     for gen, path in real_train:
    #         src = raw_path / gen / path
    #         resize_img(src, processed_path / "train" / "real" / Path(path).name)
    #
    #     for gen, path in real_val:
    #         src = raw_path / gen / path
    #         resize_img(src, processed_path / "val" / "real" / Path(path).name)
    #
    #     for gen, path in real_test:
    #         src = raw_path / gen / path
    #         resize_img(src, processed_path / "test" / category / "real" / Path(path).name)
    #
    #     for gen, path in fake_train:
    #         src = raw_path / gen / path
    #         resize_img(src, processed_path / "train" / "fake" / Path(path).name)
    #
    #     for gen, path in fake_val:
    #         src = raw_path / gen / path
    #         resize_img(src, processed_path / "val" / "fake" / Path(path).name)
    #
    #     for gen, path in fake_test:
    #         src = raw_path / gen / path
    #         resize_img(src, processed_path / "test" / category / "fake" / Path(path).name)
    #
    #     print(f"{category} is finished")
    #
    # print("All categories are finished")

    # Add all the generators unused in training to the testing dataset to see how the model performs against unseen generators
    used_gens = set(metadata["generator"].unique())

    for model in raw_path.iterdir():
        if model.name == ".complete" or model.name in used_gens:
            continue

        metadata_file = model / "metadata.csv"

        model_df = pd.read_csv(metadata_file)

        real_imgs = [Path(p) for p in model_df["image_path"][model_df["target"] == 0]]
        for img in real_imgs:
            resize_img(raw_path / model / img, processed_path / "test" / model.name / "real" / Path(img).name)

        fake_imgs = [Path(p) for p in model_df["image_path"][model_df["target"] != 0]]
        for img in fake_imgs:
            resize_img(raw_path / model / img, processed_path / "test" / model.name / "fake" / Path(img).name)

        print(f"{model.name} is finished")

    print("ArtiFact has been processed")


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
    raw_path = base_dir / "data" / "raw" / "ArtiFact"
    metadata_path = base_dir / "data" / "sorted" / "full_metadata.csv"
    processed_path = base_dir / "data" / "processed"

    create_processed(raw_path, metadata_path, processed_path)


if __name__ == '__main__':
    main()
