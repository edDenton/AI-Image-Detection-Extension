"""
Since ArtiFact has inconsistent metadata.csv categories and folder naming schemes,
I'm going to make one giant metadata.csv which contains:
 1. img_path
 2. generator
 3. category (type of image)
 4. label (0 - real, >=1 - fake)

Labels are all accurate but the categories are not

Proper labels:
    -afhq: metadata.csv is properly labeled (cat, dog, wild) - REAL
    -big_gan: (not properly labeled with no pattern of images)
    -celebahq: not properly labeled but all human faces - REAL
    -cips: metadata.csv is properly labeled (churches, human faces, landscapes)
    -coco: metadata.csv is made up of misc. images (could go into an "other" category) - REAL
    -cycle_gan: metadata.csv is properly labeled (horse2zebra, monet2photo, photo2monet) - SOME REAL
    -ddpm: metadata.csv is properly labeled (bedroom, church)
    -denoising_diffusion_gan: not properly labeled but all human faces
    -diffusion_gan: metadata.csv is properly labeled (take the ends of the categories)
    -face_synthetics: not properly labeled but all human faces
    -ffhq: not properly labeled but all human faces - REAL
    -gansformer: metadata.csv is properly labeled (bedroom and human faces)
    -gau_gan: (not properly labeled with no pattern of images)
    -generative_inpainting: (not properly labeled with no pattern of images)
    -glide: (not properly labeled with no pattern of images)
    -imagenet: not properly labeled but mostly made up of birds and fish - REAL
    -lama: metadata.csv is properly labeled (celeb (human faces) and places (landscapes))
    -landscape: not properly labeled but all backgrounds/landscapes - REAL
    -latent_diffusion: not properly labeled but mostly made up of birds and fish
    -lsun: metadata.csv is properly labeled (bedroom, car, cat, church, horse) - REAL
    -mat: metadata.csv is properly labeled (celeb (human faces) and landscapes)
    -metfaces: not properly labeled but all human faces - REAL
    -palette: not properly labeled but all human faces
    -pro_gan: metadata.csv is properly labeled (many labels)
    -projected_gan: metadata.csv is properly labeled (many labels)
    -sfhq: not properly labeled but all human faces
    -stable_diffusion: somewhat properly labeled, only stable-face is properly labeled with Female and Male but stable is random
    -star_gen: metadata.csv is properly labeled (human faces with different attributes)
    -stylegan1: not properly labeled but all faces
    -stylegan2: metadata.csv is properly labeled (car, cat, church, faces, horse)
    -stylegan3:  (not properly labeled with no pattern of images)
    -taming_transformer:  (not properly labeled with no pattern of images)
    -vq_diffusion: (not properly labeled with no pattern of images)

Strategy:
    1. All real data will be kept and used to keep a better balanced count of types of images between real and fake
    2. Not all fake generators will be kept simply because there's too many of them
    3. Many of these generators have images that are clearly nowhere near good enough to be possible to fool someone into
    believing they are real so ones that fit this will be removed as well
    4. Will also be prioritizing generators with well formatted metadata.csv files
    5. organize.py is going to handle putting all the images we want into folders (data/sorted/*) based on their labels and separated
    by real and fake. The real and fake folders will contain similar amounts of images.
    6. preprocess.py will take the folders under data/sorted/* and split them 70-15-15 train-val-test into data/processed. We can test
    our model with some of the generators we left out in training to see how it does on unseen data.

Real data:
    -afhq: metadata.csv is properly labeled (cat, dog, wild) - REAL
    -celebahq: not properly labeled but all human faces - REAL
    -coco: metadata.csv is made up of misc. images (could go into an "other" category) - REAL
    -cycle_gan: metadata.csv is properly labeled (horse2zebra, monet2photo, photo2monet) - SOME REAL
    -ffhq: not properly labeled but all human faces - REAL
    -imagenet: not properly labeled but mostly made up of birds and fish - REAL
    -landscape: not properly labeled but all backgrounds/landscapes - REAL
    -lsun: metadata.csv is properly labeled (bedroom, car, cat, church, horse) - REAL
    -metfaces: not properly labeled but all human faces - REAL

Generators getting kept:
    -cips: metadata.csv is properly labeled (churches, human faces, landscapes)
    -ddpm: metadata.csv is properly labeled (bedroom, church)
    -gansformer: metadata.csv is properly labeled (bedroom and human faces)
    -lama: metadata.csv is properly labeled (celeb (human faces) and places (landscapes))
    -mat: metadata.csv is properly labeled (celeb (human faces) and landscapes)
    -pro_gan: metadata.csv is properly labeled (many labels)
    -projected_gan: metadata.csv is properly labeled (many labels)

All categories that will be used: cat, dog, wild (all other animals), human/human faces, bedroom, churches, landscapes, car, other
"""

import pandas as pd
from pathlib import Path


def create_manifest(raw_path, sorted_path, config):
    metadata_pieces = []

    for model in raw_path.iterdir():
        if model.name not in config:
            continue

        generator_config = config[model.name]
        metadata = pd.read_csv(raw_path / model.name / "metadata.csv")

        if generator_config["mode"] == "hardcoded":
            metadata["category"] = generator_config["category"]

        elif generator_config["mode"] == "column":
            metadata["category"] = metadata["category"].map(generator_config["map"])

        narows = metadata[metadata["category"].isna()]
        if not narows.empty:
            print(narows)

        metadata = metadata.dropna(subset=["category"])
        metadata.insert(0, "generator", model.name)

        metadata_pieces.append(metadata)

    full_metadata = pd.concat(metadata_pieces)
    print(full_metadata.shape)

    full_metadata_path = sorted_path / "full_metadata.csv"
    full_metadata_path.parent.mkdir(exist_ok=True, parents=True)
    full_metadata.to_csv(full_metadata_path, index=False)


def main():
    GENERATOR_CONFIG = {
        "afhq": {
            "mode": "column",
            "column": "category",
            "map": {
                "cat": "cat",
                "dog": "dog",
                "wild": "wild"
            }
        },
        "celebahq": {
            "mode": "hardcoded",
            "category": "human"
        },
        "cips": {
            "mode": "column",
            "column": "category",
            "map": {
                "cips-churces": "church",
                "cips-ffhq": "human",
                "cips-landscape": "landscape"
            }
        },
        "coco": {
            "mode": "hardcoded",
            "category": "other"
        },
        "cycle_gan": {
            "mode": "column",
            "column": "category",
            "map": {
                "horse2zebra": "wild",
                "monet2photo": "landscape",
                "photo2monet": "landscape"
            }
        },
        "ddpm": {
            "mode": "column",
            "column": "category",
            "map": {
                "bedroom": "bedroom",
                "church": "church"
            }
        },
        "ffhq": {
            "mode": "hardcoded",
            "category": "human"
        },
        "gansformer": {
            "mode": "column",
            "column": "category",
            "map": {
                "bedrooms_images": "bedroom",
                "ffhq_images": "human"
            }
        },
        "imagenet": {
            "mode": "hardcoded",
            "category": "wild"
        },
        "lama": {
            "mode": "column",
            "column": "category",
            "map": {
                "celeb": "human",
                "places": "landscape"
            }
        },
        "landscape": {
            "mode": "hardcoded",
            "category": "landscape"
        },
        "lsun": {
            "mode": "column",
            "column": "category",
            "map": {
                "bedroom": "bedroom",
                "cat": "cat",
                "car": "car",
                "church": "church",
                "horse": "wild"
            }
        },
        "mat": {
            "mode": "column",
            "column": "category",
            "map": {
                "celebahq": "human",
                "landscape": "landscape"
            }
        },
        "metfaces": {
            "mode": "hardcoded",
            "category": "human"
        },
        "pro_gan": {
            "mode": "column",
            "column": "category",
            "map": {
                "airplane": "other",
                "bicycle": "other",
                "bird": "wild",
                "boat": "other",
                "bottle": "other",
                "bus": "other",
                "car": "car",
                "cat": "cat",
                "chair": "other",
                "cow": "wild",
                "diningtable": "other",
                "dog": "dog",
                "horse": "wild",
                "motorbike": "other",
                "person": "human",
                "pottedplant": "other",
                "sheep": "wild",
                "sofa": "other",
                "train": "other",
                "tvmonitor": "other"
            }
        },
        "projected_gan": {
            "mode": "column",
            "column": "category",
            "map": {
                "art_painting": "landscape",
                "bedroom": "bedroom",
                "church": "church",
                "cityscapes": "landscape",
                "ffhq": "human",
                "landscape": "landscape"
            }
        }
    }

    base_dir = Path(__file__).resolve().parent.parent
    artifact_path = base_dir / "data" / "raw" / "ArtiFact"

    sorted_path = base_dir / "data" / "sorted"
    sorted_path.parent.mkdir(exist_ok=True, parents=True)

    create_manifest(raw_path=artifact_path, sorted_path=sorted_path, config=GENERATOR_CONFIG)


if __name__ == "__main__":
    main()
