"""
Since ArtiFact has inconsistent metadata.csv categories and folder naming schemes,
I'm going to make one giant metadata.csv which contains:
 1. img_path
 2. generator
 3. category (type of image)
 4. label (0 - real, >=1 - fake)

Labels are all accurate but the categories are not

Proper labels:
    -afhq: metadata.csv is properly labeled (cat, dog, wild)
    -big_gan: (not properly labeled)
    -celebahq: not properly labeled but all human faces
    -cips: metadata.csv is properly labeled (churches, human faces, landscapes)
    -coco: metadata.csv is made up of misc. images (could go into an "other" category)
    -cycle_gan: metadata.csv is properly labeled (horse2zebra, monet2photo, photo2monet)
    -ddpm: metadata.csv is properly labeled (bedroom, church)
    -denoising_diffusion_gan: not properly labeled but all human faces
    -diffusion_gan: metadata.csv is properly labeled (take the ends of the categories)
    -face_synthetics: not properly labeled but all human faces
    -ffhq: not properly labeled but all human faces
    -gansformer: metadata.csv is properly labeled (bedroom and human faces)
    -gau_gan:
"""