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
import utils as ut

#TODO: Creer une classe pour le modele ? 
#Load pretrained MobileNet model
model = ut.load_model()


covers_features_path = "data/liked_tracks_cover_features.h5"
if os.path.exists(covers_features_path):
    print("Features file already exists. Loading...")
    with h5py.File(covers_features_path, "r") as h5file:
        #Load features
        lt_features = h5file["features"][:]
        #Load album names
        album_names = h5file["album_names"][:]
        album_names = [name.decode('utf-8') for name in h5file["album_names"][:]]
else:
    # Proceed with feature extraction
    breakpoint

#TODO: comparer le nom des albums avec la liste de get_tracks_list pour saovir s'il faut a mettre à jour 
# de combien  d'images: oui ou non 

normalized_dataset_features = ut.normalize_feature(lt_features)

#Extract features for the external image
external_img = pd.read_csv("./data/demo/test2.csv")
external_img_path = external_img.iloc[1,1]
external_features = ut.extract_features(external_img_path, model).astype('float32')

normalized_external_features = ut.normalize_feature(np.expand_dims(external_features, axis=0))


#Compute similarities
similarities_cos = pd.DataFrame({
    "album": album_names,
    "similarity": cosine_similarity([external_features], lt_features)[0]
})
similarities_cos = similarities_cos.sort_values(by="similarity", ascending=False)

similarities_corr = pd.DataFrame({
    "album": album_names,
    "similarity":  ut.correlation_similarity([external_features], lt_features)
})
similarities_corr = similarities_corr.sort_values(by="similarity", ascending=False)

#compute cosine similarity using FAISS
faiss_index = faiss.IndexFlatIP(normalized_dataset_features.shape[1])
faiss_index.add(normalized_dataset_features)
k = 5
distances, indices = faiss_index.search(normalized_external_features, k)
results_faiss = []
for query_idx, neighbor_indices in enumerate(indices):
    for rank, neighbor_idx in enumerate(neighbor_indices):
        results_faiss.append({
            "neighbor_album": album_names[neighbor_idx],
            "distance": distances[query_idx, rank]
        })
similarities_faiss = pd.DataFrame(results_faiss)


#load les 8k et une photo à mettre en story 
#test 1 