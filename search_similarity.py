import numpy as np 
import pandas as pd
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
import h5py
import PIL
import faiss
# from utils import *
import utils as ut

#TODO: Creer une classe pour le modele ? 
#Load pretrained MobileNet model
model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

#directory of dataset images 
# dataset_list = pd.read_csv("./demo/test1.csv")
dataset_list = pd.read_csv("./data/test_playlist.csv")




#Extract features for all dataset images 
dataset_features = []
for img_url in dataset_list['url']:
    try:
        features = ut.extract_features(ut.get_img(img_url), model)
        dataset_features.append(features)
    except Exception as e:
        # Handle exceptions if any occur (e.g., network issues, invalid image URLs)
        print(f"Error processing image {img_url}: {e}")
        # You could also choose to append a default value if needed, e.g., an empty list or None
        # dataset_features.append(None)

# dataset_features = np.array(dataset_features)
dataset_features = np.array(dataset_features).astype('float32')

index_column = dataset_list['album']

normalized_dataset_features = ut.normalize_feature(dataset_features)

#Extract features for the external image
external_img = pd.read_csv("./data/demo/test2.csv")
external_img_path = external_img.iloc[1,1]
external_features = ut.extract_features(external_img_path, model).astype('float32')

normalized_external_features = ut.normalize_feature(np.expand_dims(external_features, axis=0))


#Compute similarities
similarities_cos = cosine_similarity([external_features], dataset_features)
similarities_corr = ut.correlation_similarity([external_features], dataset_features)
#compute cosine similarity using FAISS
faiss_index = faiss.IndexFlatIP(normalized_dataset_features.shape[1])
faiss_index.add(normalized_dataset_features)
k=5
distances, indices = faiss_index.search(normalized_external_features, k)

