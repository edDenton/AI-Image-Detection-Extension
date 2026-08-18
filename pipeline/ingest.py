"""
https://www.kaggle.com/datasets/awsaf49/artifact-dataset/data
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

@author: Edward Denton
"""

import kagglehub

print("ArtiFact started")
path = kagglehub.dataset_download("awsaf49/artifact-dataset", output_dir="../data/raw/ArtiFact")
print("ArtiFact finished, path to dataset is: ", path)

# print("CIFAKE started")
# path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images", output_dir="../data/raw/CIFAKE")
# print("CIFAKE finished, path to dataset is: ", path)

