import pickle
# import lmdb
import spotipy
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet



# def save_img(image_list, directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     for img in image_list :
    

# def store_image_lmbd(images_list, img_db_path):
#     map_size = len(images_list) * 300 * 300 * 10 
#     env = lmdb.open(img_db_path, map_size=map_size)
#     with env.begin(write=True) as txn:
#         for i, img in enumerate(images_list):
#             txn.put(f'image_{i}'.encode(), pickle.dumps(img))

def get_tracks_list(sp_rqrmt, url_playlist="https://open.spotify.com/collection/tracks", debug_mode=False):
    all_tracks = []
    limit = 50 if url_playlist == "https://open.spotify.com/collection/tracks" else 100
    offset = 0
    max_tracks = 100 if debug_mode else None
    
    # Choose the correct method based on URL
    is_liked_songs = url_playlist == "https://open.spotify.com/collection/tracks"
    fetch_tracks = sp_rqrmt.current_user_saved_tracks if is_liked_songs else sp_rqrmt.playlist_tracks
    playlist_uri = None if is_liked_songs else url_playlist.split("/")[-1].split("?")[0]
    
    # Retrieve tracks with pagination
    while True:
        # Fetch a batch of tracks
        if is_liked_songs:
            response = fetch_tracks(limit=limit, offset=offset)
        else:
            response = fetch_tracks(playlist_id=playlist_uri, limit=limit, offset=offset)
        
        # Extend all_tracks with the current batch
        all_tracks.extend(response["items"])
        
        # Break conditions for pagination
        if len(response['items']) < limit or (max_tracks and len(all_tracks) >= max_tracks):
            break
        
        # Update offset for next batch
        offset += limit
    
    # Trim to max_tracks if in debug mode
    if max_tracks:
        all_tracks = all_tracks[:max_tracks]

    return all_tracks


# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # size of album covers: 300*300
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


def normalize_feature(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features/norms


def get_img(url):  
    # Télécharger l'image depuis l'URL
    response = requests.get(url)
    if response.status_code == 200:  # Vérifie si la requête est réussie
        # Charger l'image avec Pillow
        image = Image.open(BytesIO(response.content))
        width, heigth = image.size
        print(f"largeur :{width} et hauteur: {heigth}")
        # image.show()  # Affiche l'image dans un visualiseur (selon votre système)
        # Optionnel : Sauvegarder l'image localement
        image.save("./temp/spotify_image.jpg")
        print(f"Image au lien {url} téléchargée ")
        return "temp/spotify_image.jpg"
    else:
        print(f"Erreur lors du téléchargement : {response.status_code}")
        return False

def load_model():
    model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
    return model