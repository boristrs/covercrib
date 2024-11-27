import numpy as np 
import pandas as pd
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
import h5py
import PIL
# from PIL import Image
#Load pretrained MobileNet model
model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

#Function to extract features
def extract_features(img_path, model):
    img=image.load_img(img_path, target_size=(224,224)) #TODO: check size of album covers
    # img=img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


def correlation_similarity(vector1, vector2):
    """
    Calculate correlation similarity between two vectors.
    
    Args:
    Return:
    
    """
    # Ensure the vectors are NumPy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    
    correlation_scores = []
    
    # Calculate correlation coefficient
    if len(vector2) > 1:
        for vector in vector2:
            correlation_scores.append(np.corrcoef(vector1, vector)[0, 1])
    else:
        correlation_scores.append(np.corrcoef(vector1, vector2)[0, 1])
    return correlation_scores

#directory of dataset images 
dataset_list = pd.read_csv("./demo/test1.csv")
imgdata_paths = dataset_list['url']

#Extract features for all dataset images 
dataset_features = []
for img_path in imgdata_paths:
    features = extract_features(img_path, model)
    dataset_features.append(features)
    
dataset_features = np.array(dataset_features)

#Extract features for the external image
external_img = pd.read_csv("./demo/test2.csv")
external_img_path = external_img.iloc[1,1]
external_features = extract_features(external_img_path, model)

#Compute similarities
similarities_cos = cosine_similarity([external_features], dataset_features)
similarities_corr = correlation_similarity([external_features], dataset_features)

# Find the top 5 most similar images
top_indices = np.argsort(similarities[0])[::-1][:5]
print("Top 5 similar images:")
for idx in top_indices:
    print(f"Image: {dataset_images[idx]}, Similarity: {similarities[0][idx]}")











"""
Use Advanced Search Tools (e.g., FAISS):
For faster search across 8K images, use approximate nearest neighbors (e.g., FAISS) regardless of the similarity metric.
"""