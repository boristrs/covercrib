import pandas as pd
import numpy as np
import os
import h5py
import utils as ut
import sys
from sklearn.metrics.pairwise import cosine_similarity

DEBUG = os.getenv("DEBUG") == "True"
if DEBUG:
    print("Mode débogage activé")
    
    
def main(img_path=None, tf_lite=False):
    """
    Main function to process an image and compute its similarity with a dataset of album covers.
    Functionality:
    - If a precomputed features file exists (`data/liked_tracks_cover_features.h5`), it loads the features and album names.
    - If the features file does not exist, it connects to Spotify, extracts liked songs, and computes features for them.
    - Normalizes the dataset features.
    - Extracts and normalizes features for the external image provided via `img_path`.
    - Computes cosine similarity between the external image features and the dataset features.
    - Outputs a sorted DataFrame of album names and their similarity scores.
    Args:
        img_path (str, optional): Path to the external image to be processed. Defaults to None.
        tf_lite (bool, optional): Flag to indicate whether to use TensorFlow Lite for feature extraction. Defaults to False.
    Returns:
        similarities_cos (pd.DataFrame): DataFrame containing album names with the maximum of cosine similarity score
    """
    
    if img_path is None:
        raise ValueError("Please provide an image name")
    
    #Read saved features or connect to spotify to extract them
    if DEBUG:
        covers_features_path = "data/DEBUG_liked_tracks_cover_features.h5"
    else:
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
        lt_features, album_names = ut.extract_and_save_liked_music_covers_features(music_id_data, tf_lite)

    normalized_dataset_features = ut.normalize_feature(lt_features)

    #Extract features for the external image
    external_features = ut.get_test_image_features(external_img_path)
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
    
    DEBUG = os.getenv("DEBUG") == "True"
    if DEBUG:
        print("Mode débogage activé")

    #USE OF TF LITE for mobile deplyment
    tf_lite = False    
    
    # #TODO: take the image as input from the user
    # external_img_path = sys.argv[1]
    # lite = sys.argv[2] if len(sys.argv) > 2 else False
    img_name = "flo-horus.png"
    external_img_path = f"data/input_image/{img_name}"
    
    main(external_img_path, tf_lite)