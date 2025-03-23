

import pandas as pd
import numpy as np
import os
import h5py
import utils_apk as ut

from sklearn.metrics.pairwise import cosine_similarity

DEBUG = os.getenv("DEBUG") == "True"
if DEBUG:
    print("Mode débogage activé")
    

def main(img_path = None):
    
    if img_name is None:
        raise error ("Please provide an image name")
    
    #Read saved features or connect tos spotify to extract them
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
        print("Features file does not exists. Creating...")
        sp = ut.connect_to_spotify()
        music_id_data = ut.extract_liked_song(sp, DEBUG)
        #TODO: Compute new features only for new liked tracks and update the h5 file
        lt_features, album_names = ut.extract_and_save_liked_music_covers_features(music_id_data)

    normalized_dataset_features = ut.normalize_feature(lt_features)

    #Extract features for the external image
    external_features = ut.get_image_features(external_img_path)
    normalized_external_features = ut.normalize_feature(np.expand_dims(external_features, axis=0))

    #Compute similarities
    print("Compute similarities...")
    similarities_cos = pd.DataFrame({
        "album": album_names,
        "similarity": cosine_similarity(normalized_external_features, normalized_dataset_features)[0]
        # "similarity": cosine_similarity([external_features], lt_features)[0]
    })
    similarities_cos = similarities_cos.sort_values(by="similarity", ascending=False)

    print(similarities_cos)


if __name__ == '__main__':
    
    img_name = "flo-horus.png"
    external_img_path = f"data/input_image/{img_name}"
    
    main(external_img_path)
    
    
#TODO: take the image as input from the user
#TODO: add a function to display the image and the top 5 similar albums
#TODO: Complete readme file
#TODO: Switch to Kivy for the GUI